#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------- paths --------
SRC_DATA_ROOT = Path("datasets_classifier")              # 原始資料：datasets/<gt_task>/<split>.jsonl (每行含 input/target 或 text/target)
CLS_DIR       = Path("task_classifier_ckpt")  # task classifier checkpoint
OUT_ROOT      = Path("routed_eval")           # 輸出：routed_eval/<routed_task>/<split>.jsonl

# -------- settings --------
SPLIT = "test"        # 你要路由的 split；如果沒有 test，改成 "validation"
BATCH_SIZE = 64
MAX_LEN = 512

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def get_id2label(model):
    id2label = model.config.id2label
    if isinstance(id2label, dict):
        return {int(k): v for k, v in id2label.items()}
    return {i: v for i, v in enumerate(id2label)}

@torch.no_grad()
def predict_batch(model, tok, texts, device):
    toks = tok(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)
    logits = model(**toks).logits
    return logits.argmax(dim=-1).tolist()

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # load classifier
    tok = AutoTokenizer.from_pretrained(CLS_DIR)
    clf = AutoModelForSequenceClassification.from_pretrained(CLS_DIR).to(device)
    clf.eval()

    id2label = get_id2label(clf)

    # stats
    total = 0
    correct = 0
    per_task = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = defaultdict(lambda: defaultdict(int))  # gt_task -> routed_task -> count

    # buckets[routed_task] -> list of routed records
    buckets = defaultdict(list)

    uid = 0
    tasks = sorted([d.name for d in SRC_DATA_ROOT.iterdir() if d.is_dir()])
    if not tasks:
        raise RuntimeError(f"No task dirs found under {SRC_DATA_ROOT.resolve()}")

    for gt_task in tasks:
        src_path = SRC_DATA_ROOT / gt_task / f"{SPLIT}.jsonl"
        if not src_path.exists():
            print(f"[SKIP] missing {src_path}")
            continue

        print(f"\n[RUN] gt_task={gt_task} file={src_path}")

        batch_texts = []
        batch_ex = []

        for ex in iter_jsonl(src_path):
            prompt = ex.get("input") or ex.get("text")
            if prompt is None:
                raise KeyError(f"No 'input' or 'text' field in example: {ex}")

            batch_texts.append(prompt)
            batch_ex.append(ex)

            if len(batch_texts) >= BATCH_SIZE:
                pred_ids = predict_batch(clf, tok, batch_texts, device)

                for e, pid in zip(batch_ex, pred_ids):
                    routed_task = id2label[int(pid)]

                    # update stats
                    total += 1
                    per_task[gt_task]["total"] += 1
                    confusion[gt_task][routed_task] += 1
                    if routed_task == gt_task:
                        correct += 1
                        per_task[gt_task]["correct"] += 1

                    # write record to bucket
                    rec = {
                        "id": uid,
                        "gt_task": gt_task,
                        "routed_task": routed_task,
                        "input": e.get("input", e.get("text")),
                        "target": e.get("target", None),
                    }
                    buckets[routed_task].append(rec)
                    uid += 1

                batch_texts, batch_ex = [], []

        # flush remainder
        if batch_texts:
            pred_ids = predict_batch(clf, tok, batch_texts, device)

            for e, pid in zip(batch_ex, pred_ids):
                routed_task = id2label[int(pid)]

                # update stats
                total += 1
                per_task[gt_task]["total"] += 1
                confusion[gt_task][routed_task] += 1
                if routed_task == gt_task:
                    correct += 1
                    per_task[gt_task]["correct"] += 1

                # write record to bucket
                rec = {
                    "id": uid,
                    "gt_task": gt_task,
                    "routed_task": routed_task,
                    "input": e.get("input", e.get("text")),
                    "target": e.get("target", None),
                }
                buckets[routed_task].append(rec)
                uid += 1

    # write routed subsets
    for routed_task, recs in buckets.items():
        out_dir = OUT_ROOT / routed_task
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{SPLIT}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[WRITE] {routed_task}: {len(recs)} -> {out_path}")

    # write reports
    overall_acc = correct / total if total else 0.0
    report = {
        "split": SPLIT,
        "total": total,
        "correct": correct,
        "overall_routing_accuracy": overall_acc,
        "per_task": {
            t: {
                "total": per_task[t]["total"],
                "correct": per_task[t]["correct"],
                "routing_accuracy": (per_task[t]["correct"] / per_task[t]["total"]) if per_task[t]["total"] else 0.0,
            }
            for t in sorted(per_task.keys())
        },
        "confusion": {gt: dict(confusion[gt]) for gt in sorted(confusion.keys())},
        "classifier_id2label": {str(k): v for k, v in sorted(id2label.items())},
    }

    report_path = OUT_ROOT / "routing_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # print summary
    print("\n===== Routing accuracy (Task Classifier) =====")
    print(f"Overall routing accuracy: {overall_acc:.4f} ({correct}/{total})")
    for t in sorted(per_task.keys()):
        tot = per_task[t]["total"]
        cor = per_task[t]["correct"]
        if tot:
            print(f"  {t:12s}: {cor/tot:.4f} ({cor}/{tot})")

    print("\n✅ Done.")
    print(f"✅ Routed subsets saved to: {OUT_ROOT.resolve()}")
    print(f"✅ Report saved to: {report_path.resolve()}")

if __name__ == "__main__":
    main()
