#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time
from collections import defaultdict, Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = "datasets_classifier"      # 你前面產生的 classifier dataset
CLS_DIR  = "task_classifier_ckpt"     # 你 train_task_classifier.py 的輸出
OUT_DIR  = "routing_reports"

BATCH_SIZE = 32
MAX_LEN = 512
PRINT_EVERY = 5000   # 每處理幾筆印一次進度

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def load_label_space():
    # 用資料夾名稱當作「真實 task label」
    tasks = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if not tasks:
        raise RuntimeError(f"No task folders under {DATA_DIR}")
    return tasks

def get_id2label(model):
    id2label = model.config.id2label
    # 可能是 dict: {0:"sst2", ...} 或 list: ["sst2", ...]
    if isinstance(id2label, dict):
        # key 強制轉 int，避免 '4' 這種問題
        fixed = {}
        for k, v in id2label.items():
            fixed[int(k)] = v
        return fixed
    elif isinstance(id2label, (list, tuple)):
        return {i: v for i, v in enumerate(id2label)}
    else:
        raise TypeError(f"Unsupported id2label type: {type(id2label)}")

@torch.no_grad()
def predict_batch(model, tokenizer, texts, device):
    toks = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)
    logits = model(**toks).logits
    pred_ids = logits.argmax(dim=-1).tolist()
    return pred_ids

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(CLS_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CLS_DIR).to(device)
    model.eval()

    id2label = get_id2label(model)

    tasks = load_label_space()
    splits = ["test"]  # 有些 dataset 沒某個 split 會自動跳過

    # overall stats
    total = 0
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))  # gt -> pred -> count
    per_task_counts = {t: Counter() for t in tasks}    # gt task 下，各 pred task 次數
    per_task_acc = defaultdict(lambda: {"total": 0, "correct": 0})

    for gt_task in tasks:
        for split in splits:
            path = os.path.join(DATA_DIR, gt_task, f"{split}.jsonl")
            if not os.path.exists(path):
                continue

            print(f"\n[RUN] gt_task={gt_task} split={split} file={path}")

            batch_texts = []
            n = 0
            t0 = time.time()

            for ex in iter_jsonl(path):
                batch_texts.append(ex["text"])
                if len(batch_texts) >= BATCH_SIZE:
                    pred_ids = predict_batch(model, tokenizer, batch_texts, device)
                    for pid in pred_ids:
                        pred_task = id2label[int(pid)]
                        confusion[gt_task][pred_task] += 1
                        per_task_counts[gt_task][pred_task] += 1

                        total += 1
                        per_task_acc[gt_task]["total"] += 1
                        if pred_task == gt_task:
                            correct += 1
                            per_task_acc[gt_task]["correct"] += 1

                    n += len(batch_texts)
                    if n % PRINT_EVERY == 0:
                        dt = time.time() - t0
                        speed = n / max(dt, 1e-6)
                        print(f"  ... processed {n} samples | {speed:.1f} samples/s", flush=True)
                    batch_texts = []

            # flush remainder
            if batch_texts:
                pred_ids = predict_batch(model, tokenizer, batch_texts, device)
                for pid in pred_ids:
                    pred_task = id2label[int(pid)]
                    confusion[gt_task][pred_task] += 1
                    per_task_counts[gt_task][pred_task] += 1

                    total += 1
                    per_task_acc[gt_task]["total"] += 1
                    if pred_task == gt_task:
                        correct += 1
                        per_task_acc[gt_task]["correct"] += 1
                n += len(batch_texts)

            dt = time.time() - t0
            print(f"[DONE] {gt_task}/{split}: {n} samples in {dt:.1f}s", flush=True)

    overall_acc = correct / total if total else 0.0

    # build confusion matrix in a stable order
    confusion_matrix = {gt: {pred: confusion[gt].get(pred, 0) for pred in tasks} for gt in tasks}
    per_task_breakdown = {}
    for gt in tasks:
        tot = per_task_acc[gt]["total"]
        cor = per_task_acc[gt]["correct"]
        per_task_breakdown[gt] = {
            "routing_acc": (cor / tot) if tot else 0.0,
            "total": tot,
            "pred_distribution": dict(per_task_counts[gt]),
        }

    overall = {
        "total_samples": total,
        "correct": correct,
        "routing_accuracy": overall_acc,
        "tasks": tasks,
        "classifier_id2label": {str(k): v for k, v in id2label.items()},
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LEN,
    }

    with open(os.path.join(OUT_DIR, "overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(confusion_matrix, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "per_task_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(per_task_breakdown, f, ensure_ascii=False, indent=2)

    print("\n✅ Routing evaluation finished.")
    print(f"✅ Overall routing accuracy = {overall_acc:.4f} ({correct}/{total})")
    print(f"📁 Reports saved to: {OUT_DIR}/")

if __name__ == "__main__":
    main()
