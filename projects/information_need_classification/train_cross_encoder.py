#!/usr/bin/env python3
"""Fine-tune a DistilBERT-MNLI cross-encoder on (text, justification) pairs.

Implements Section 7 of the white paper, step for step against Figure 7:

    1 sample -> 2 tokenise -> 3 encode -> 4 pool -> 5 score -> 6 loss -> 7 update

Written as a plain PyTorch loop rather than transformers.Trainer. Two reasons:
the loop is the thing the paper's figure describes, so it should be readable;
and Trainer auto-DataParallels across every visible GPU, which stalls on this
box (see gpu.py).

    python train_cross_encoder.py \
        --texts corpus/texts.parquet \
        --needs corpus/needs.parquet \
        --hard-negatives corpus/hard_negatives.json \
        --out runs/v1

Smoke test without a GPU or a downloaded checkpoint:
    IN_CLS_GPU=cpu python train_cross_encoder.py ... --model bert-base-uncased \
        --epochs 1 --max-steps 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gpu import describe_device, pin_single_gpu

pin_single_gpu()

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from torch.utils.data import DataLoader, Dataset                    # noqa: E402
from transformers import (AutoModelForSequenceClassification,       # noqa: E402
                          AutoTokenizer, get_linear_schedule_with_warmup)

import metrics                                                      # noqa: E402
from data import (SamplingPolicy, build_pairs, candidate_pairs,     # noqa: E402
                  load_corpus, split_by_text)

DEFAULT_MODEL = "typeform/distilbert-base-uncased-mnli"


# ------------------------------------------------------------- dataset ------

class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def make_collate(tokenizer, max_length):
    """Step 2. Truncate the TEXT, never the justification.

    The justification is short, curated, and identical everywhere it appears.
    The text is long and its relevant passage may be anywhere. Cutting the
    justification would remove the thing we are matching against.
    """
    def collate(batch):
        enc = tokenizer(
            [p.text for p in batch],
            [p.justification for p in batch],
            truncation="only_first",
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([float(p.label) for p in batch]).unsqueeze(1)
        return enc, batch
    return collate


# ---------------------------------------------------------------- model -----

def build_model(model_name, device):
    """Keep the encoder, discard the 3-way MNLI head, attach one sigmoid unit.

    num_labels=1 with BCEWithLogitsLoss is the multi-label formulation: each
    pair is an independent yes-or-no decision and no softmax makes the needs
    compete for probability mass.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, ignore_mismatched_sizes=True)
    return model.to(device)


def _grad_scaler(enabled):
    try:                                    # torch >= 2.4
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):     # torch 2.1.x
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ----------------------------------------------------------------- score ----

@torch.no_grad()
def score_pairs(model, tokenizer, pairs, device, batch_size=64, max_length=256,
                amp=False):
    """Sigmoid probability per pair. Returns (text_id, need_id, score, label)."""
    model.eval()
    collate = make_collate(tokenizer, max_length)
    loader = DataLoader(PairDataset(pairs), batch_size=batch_size,
                        shuffle=False, collate_fn=collate)
    out = []
    for enc, batch in loader:
        labels = enc.pop("labels")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            logits = model(**enc).logits
        probs = torch.sigmoid(logits.float()).squeeze(-1).cpu().numpy()
        for p, s, y in zip(batch, probs, labels.squeeze(-1).numpy()):
            out.append((p.text_id, p.need_id, float(s), int(y)))
    return out


def eval_split(model, tokenizer, split_texts, needs, device, ranked=None,
               eval_k=50, batch_size=64, max_length=256, amp=False):
    """Score a whole split and report.

    Candidates come from the retriever shortlist when one is supplied, because
    that is what serving does. Evaluating against all M needs while serving only
    the top k would report a system nobody runs.

    A gold need OUTSIDE the shortlist is deliberately not scored, because the
    served system would never score it either. It is carried in gold_map instead,
    so it counts as a miss. Adding it to the candidate set would let the
    cross-encoder "recover" a need the retriever never surfaced, and recall would
    then exceed the retriever's own ceiling, which is impossible in production.

    Returns (records, need_vocab, gold_map).
    """
    pairs, need_vocab = [], needs["need_id"].tolist()
    gold_map = {}
    for row in split_texts.itertuples(index=False):
        gold_map[row.text_id] = list(row.need_ids)
        cands = None
        if ranked is not None:
            cands = ranked.get(row.text_id, [])[:eval_k]
        pairs.extend(candidate_pairs(row, needs, candidates=cands))

    records = score_pairs(model, tokenizer, pairs, device,
                          batch_size=batch_size, max_length=max_length, amp=amp)
    return records, need_vocab, gold_map


# ----------------------------------------------------------------- train ----

def train(args):
    t0 = time.time()
    dev_info = describe_device()
    device = torch.device("cuda" if dev_info["cuda"] else "cpu")
    print(f"  {dev_info}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data
    texts, needs = load_corpus(args.texts, args.needs)
    if args.limit_texts:
        texts = texts.head(args.limit_texts).copy()
    texts = split_by_text(texts, val_frac=args.val_frac,
                          test_frac=args.test_frac, seed=args.seed)
    counts = texts["split"].value_counts().to_dict()
    print(f"  {len(texts)} texts, {len(needs)} needs, split={counts}")

    ranked = None
    if args.hard_negatives:
        blob = json.loads(Path(args.hard_negatives).read_text(encoding="utf-8"))
        ranked = blob.get("ranked", blob)
        if "recall_ceiling" in blob:
            print(f"  retriever recall ceiling: {blob['recall_ceiling']}")
    else:
        print("  NOTE no --hard-negatives given. Negatives will be all-random, "
              "which trains an easier problem than the one you will serve.")

    policy = SamplingPolicy(n_hard=args.n_hard, n_random=args.n_random,
                            seed=args.seed)
    train_pairs, stats = build_pairs(
        texts[texts.split == "train"], needs, policy, hard_negatives=ranked)
    print(f"  train pairs: {stats}")
    if not train_pairs:
        raise SystemExit("no training pairs built; check that texts carry need_ids")

    # ---- model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = build_model(args.model, device)
    amp = args.fp16 and device.type == "cuda"
    print(f"  model={args.model}  params="
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  fp16={amp}")

    collate = make_collate(tokenizer, args.max_length)
    loader = DataLoader(PairDataset(train_pairs), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate, drop_last=False)

    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and not any(s in n for s in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and any(s in n for s in ("bias", "LayerNorm.weight"))]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    steps_per_epoch = len(loader)
    total_steps = (args.max_steps if args.max_steps
                   else steps_per_epoch * args.epochs)
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_frac), total_steps)
    scaler = _grad_scaler(amp)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_map, best_epoch, step = [], -1.0, -1, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for enc, _ in loader:
            labels = enc.pop("labels").to(device)
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=amp):
                logits = model(**enc).logits          # steps 3, 4, 5
                loss = loss_fn(logits.float(), labels)  # step 6

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optim)                        # step 7
            scaler.update()
            sched.step()

            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            step += 1
            if step % args.log_every == 0:
                print(f"    epoch {epoch} step {step}/{total_steps} "
                      f"loss={running / max(seen, 1):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        train_loss = running / max(seen, 1)

        # ---- validation. Select on mAP, which needs no threshold.
        val_texts = texts[texts.split == "val"]
        if len(val_texts) == 0:
            print("  no validation split; skipping model selection")
            break
        records, vocab, gold_map = eval_split(
            model, tokenizer, val_texts, needs, device, ranked=ranked,
            eval_k=args.eval_k, batch_size=args.eval_batch_size,
            max_length=args.max_length, amp=amp)
        _, _, S, Y = metrics._as_matrix(records, vocab, gold_map=gold_map)
        val_map = metrics.mean_average_precision(S, Y)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_mAP": val_map})
        print(f"  epoch {epoch}: train_loss={train_loss:.4f}  val_mAP={val_map:.4f}")

        if val_map > best_map:
            best_map, best_epoch = val_map, epoch
            model.save_pretrained(out_dir / "best")
            tokenizer.save_pretrained(out_dir / "best")
            gt, gf1 = metrics.tune_global_threshold(S, Y)
            per_need, n_fitted = metrics.tune_per_need_thresholds(
                S, Y, vocab, gt, min_support=args.min_support_per_need)
            (out_dir / "thresholds.json").write_text(json.dumps(
                {"global": gt, "global_val_micro_f1": gf1,
                 "per_need": per_need, "n_per_need_fitted": n_fitted,
                 "min_support_per_need": args.min_support_per_need},
                indent=2), encoding="utf-8")
            print(f"    saved best (mAP {val_map:.4f}); global threshold "
                  f"{gt:.2f}, {n_fitted} per-need thresholds fitted")
        elif args.early_stop:
            print(f"  val_mAP did not improve on epoch {best_epoch} "
                  f"({best_map:.4f}); stopping early")
            break

        if args.max_steps and step >= args.max_steps:
            break

    config = {
        "model": args.model, "epochs_run": len(history),
        "best_epoch": best_epoch, "best_val_mAP": best_map,
        "history": history, "train_pair_stats": stats,
        "device": dev_info, "fp16": amp,
        "hyperparameters": {
            "lr": args.lr, "batch_size": args.batch_size,
            "max_length": args.max_length, "weight_decay": args.weight_decay,
            "warmup_frac": args.warmup_frac, "n_hard": args.n_hard,
            "n_random": args.n_random, "eval_k": args.eval_k, "seed": args.seed,
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (out_dir / "train_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    texts[["text_id", "split"]].to_json(
        out_dir / "splits.json", orient="records")

    print(f"\n  best epoch {best_epoch}, val mAP {best_map:.4f}")
    print(f"  wrote {out_dir}/best, thresholds.json, train_config.json, "
          f"splits.json")
    print(f"  {config['wall_clock_s']}s total")
    return config


def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--needs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hard-negatives", default=None,
                    help="JSON from retriever.py. Strongly recommended.")
    ap.add_argument("--model", default=DEFAULT_MODEL)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.10)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")

    ap.add_argument("--n-hard", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--eval-k", type=int, default=50)
    ap.add_argument("--min-support-per-need", type=int, default=10)

    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--early-stop", action="store_true", default=True)
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false")

    ap.add_argument("--limit-texts", type=int, default=0, help="smoke testing")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke testing")
    ap.add_argument("--log-every", type=int, default=25)
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
