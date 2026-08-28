#!/usr/bin/env python3
"""Score a trained cross-encoder on the held-out test split.

Implements Section 8. Leads the report with the two numbers the paper argues
you should read before any headline F1:

    micro-F1 minus macro-F1   does the long tail work?
    recall@k                  what is the ceiling stage one imposes?

Reuses the split recorded at training time, so the test texts are the ones the
model never saw. Do not re-split here; that would leak.

    python evaluate.py --run runs/v1 --texts texts.parquet --needs needs.parquet \
        --hard-negatives hard_negatives.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu import describe_device, pin_single_gpu

pin_single_gpu()

import pandas as pd                                              # noqa: E402
import torch                                                     # noqa: E402
from transformers import (AutoModelForSequenceClassification,    # noqa: E402
                          AutoTokenizer)

import metrics                                                   # noqa: E402
from data import load_corpus                                     # noqa: E402
from train_cross_encoder import eval_split                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training output dir")
    ap.add_argument("--texts", required=True)
    ap.add_argument("--needs", required=True)
    ap.add_argument("--hard-negatives", default=None)
    ap.add_argument("--split", default="test", choices=("test", "val", "train"))
    ap.add_argument("--eval-k", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--per-need-thresholds", action="store_true", default=True)
    ap.add_argument("--global-threshold-only", dest="per_need_thresholds",
                    action="store_false")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = ap.parse_args()

    run = Path(args.run)
    dev_info = describe_device()
    device = torch.device("cuda" if dev_info["cuda"] else "cpu")

    texts, needs = load_corpus(args.texts, args.needs)

    splits_path = run / "splits.json"
    if not splits_path.exists():
        raise SystemExit(
            f"{splits_path} not found. Evaluation must reuse the split recorded "
            f"at training time; re-splitting here would leak training texts into "
            f"the test set.")
    splits = pd.DataFrame(json.loads(splits_path.read_text(encoding="utf-8")))
    texts = texts.merge(splits, on="text_id", how="inner")
    eval_texts = texts[texts.split == args.split]
    if len(eval_texts) == 0:
        raise SystemExit(f"split {args.split!r} is empty")

    ranked = None
    if args.hard_negatives:
        blob = json.loads(Path(args.hard_negatives).read_text(encoding="utf-8"))
        ranked = blob.get("ranked", blob)

    tokenizer = AutoTokenizer.from_pretrained(run / "best")
    model = AutoModelForSequenceClassification.from_pretrained(
        run / "best").to(device)

    thresholds = {}
    tpath = run / "thresholds.json"
    if tpath.exists():
        thresholds = json.loads(tpath.read_text(encoding="utf-8"))
    global_t = thresholds.get("global", 0.5)
    per_need = thresholds.get("per_need") if args.per_need_thresholds else None

    print(f"  {dev_info}")
    print(f"  evaluating {len(eval_texts)} {args.split} texts against "
          f"{len(needs)} needs, shortlist k={args.eval_k}")
    print(f"  threshold {global_t:.3f}"
          f"{' + per-need overrides' if per_need else ' (global only)'}")

    records, vocab, gold_map = eval_split(
        model, tokenizer, eval_texts, needs, device, ranked=ranked,
        eval_k=args.eval_k, batch_size=args.batch_size,
        max_length=args.max_length, amp=args.fp16 and device.type == "cuda")

    rep = metrics.evaluate(records, vocab, threshold=global_t,
                           per_need_thresholds=per_need, gold_map=gold_map)
    n_unreachable = sum(
        1 for tid, gold in gold_map.items() for n in gold
        if ranked is not None and n not in set(ranked.get(tid, [])[:args.eval_k]))
    rep["gold_outside_shortlist"] = n_unreachable
    print(metrics.format_report(rep))

    if ranked is None:
        print("  NOTE no retriever supplied, so every need was scored "
              "exhaustively. recall@k here describes the cross-encoder alone, "
              "not the two-stage system you would serve.")
    else:
        n_gold = sum(len(g) for g in gold_map.values())
        print(f"  {n_unreachable} of {n_gold} gold needs fell outside the top-"
              f"{args.eval_k} shortlist and were never scored. They count as "
              f"misses above, which is what the served system would do.")

    out = run / f"eval_{args.split}.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    preds = run / f"predictions_{args.split}.jsonl"
    with preds.open("w", encoding="utf-8") as fh:
        for tid, nid, score, label in records:
            fh.write(json.dumps({"text_id": tid, "need_id": nid,
                                 "score": round(score, 6), "label": label}) + "\n")
    print(f"\n  wrote {out}\n  wrote {preds}")


if __name__ == "__main__":
    main()
