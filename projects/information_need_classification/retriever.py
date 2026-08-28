#!/usr/bin/env python3
"""Bi-encoder retriever: mines hard negatives, and is stage one at serving time.

Two jobs, one model (Sections 3.1 and 9 of the white paper):

  1. TRAINING TIME. Rank needs for each text so build_pairs can draw hard
     negatives, the high-ranked non-gold candidates that sit on the decision
     boundary and do the actual teaching.

  2. SERVING TIME. Narrow thousands of needs to a top-k shortlist so the
     cross-encoder only has to score k pairs instead of M.

The retriever is deliberately FROZEN. Mining hard negatives with the model being
trained makes the two co-adapt and the negatives stop being informative, so this
never shares weights with the cross-encoder.

Mean pooling over a plain AutoModel, so sentence-transformers is not required.

    python retriever.py --texts texts.parquet --needs needs.parquet \
        --out hard_negatives.json --top-k 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu import pin_single_gpu

pin_single_gpu()

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from transformers import AutoModel, AutoTokenizer           # noqa: E402

from data import load_corpus                                # noqa: E402

DEFAULT_RETRIEVER = "sentence-transformers/all-MiniLM-L6-v2"


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class BiEncoder:
    def __init__(self, model_name=DEFAULT_RETRIEVER, max_length=256,
                 batch_size=64, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, sentences, show_every=0):
        out = []
        for i in range(0, len(sentences), self.batch_size):
            chunk = list(sentences[i:i + self.batch_size])
            enc = self.tok(chunk, padding=True, truncation=True,
                           max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            hid = self.model(**enc).last_hidden_state
            vec = _mean_pool(hid, enc["attention_mask"])
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out.append(vec.cpu().numpy())
            if show_every and (i // self.batch_size) % show_every == 0:
                print(f"    encoded {min(i + self.batch_size, len(sentences))}"
                      f"/{len(sentences)}", flush=True)
        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)


def rank_needs(texts, needs, model_name=DEFAULT_RETRIEVER, top_k=50,
               batch_size=64, max_length=256):
    """Return {text_id: [need_id, ...]} ranked by cosine similarity, top_k deep."""
    enc = BiEncoder(model_name, max_length=max_length, batch_size=batch_size)

    # Needs are embedded ONCE and reused. That reuse is the entire reason the
    # bi-encoder is affordable at this scale, and it is exactly what the
    # cross-encoder cannot do.
    print(f"  embedding {len(needs)} need justifications ...", flush=True)
    need_txt = (needs["need_label"].fillna("") + ". "
                + needs["justification"].fillna("")).tolist()
    V = enc.encode(need_txt, show_every=20)

    print(f"  embedding {len(texts)} texts ...", flush=True)
    U = enc.encode(texts["text"].tolist(), show_every=20)

    need_ids = needs["need_id"].tolist()
    k = min(top_k, len(need_ids))
    sims = U @ V.T                                    # both L2-normalised
    order = np.argsort(-sims, axis=1)[:, :k]

    return ({tid: [need_ids[j] for j in row]
             for tid, row in zip(texts["text_id"].tolist(), order)},
            sims, need_ids)


def recall_ceiling(ranked, texts, ks=(10, 20, 50)):
    """What fraction of gold needs does stage one let through?

    Section 9. This is a HARD CEILING on the full system: a need the retriever
    never surfaces cannot be recovered by any amount of cross-encoder quality.
    Measure it before tuning anything downstream.
    """
    out = {}
    rows = [r for r in texts.itertuples(index=False) if len(r.need_ids) > 0]
    for k in ks:
        hits = []
        for r in rows:
            top = set(ranked.get(r.text_id, [])[:k])
            gold = set(r.need_ids)
            hits.append(len(top & gold) / len(gold))
        out[f"recall@{k}"] = float(np.mean(hits)) if hits else float("nan")
    out["n_texts_with_gold"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--needs", required=True)
    ap.add_argument("--out", required=True, help="JSON: {text_id: [need_id,...]}")
    ap.add_argument("--model", default=DEFAULT_RETRIEVER)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    texts, needs = load_corpus(args.texts, args.needs)
    print(f"  {len(texts)} texts, {len(needs)} needs, model={args.model}")

    ranked, _, _ = rank_needs(texts, needs, model_name=args.model,
                              top_k=args.top_k, batch_size=args.batch_size,
                              max_length=args.max_length)

    ceiling = recall_ceiling(ranked, texts, ks=(10, 20, args.top_k))
    print("\n  STAGE-ONE RECALL CEILING (the cross-encoder cannot exceed this):")
    for k, v in ceiling.items():
        print(f"    {k:<26} {v}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"ranked": ranked, "recall_ceiling": ceiling,
         "model": args.model, "top_k": args.top_k}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
