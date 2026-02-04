#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

task_dir = Path("/home/kaia/recall_1108/datasets_classifier/squad2")
TRAIN_RATIO = 0.9   # 90% train, 10% test
SEED = 42

random.seed(SEED)

src_path = task_dir / "train.jsonl"
if not src_path.exists():
    print(src_path)
    print("whyyyy")

with src_path.open("r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

random.shuffle(lines)

split_idx = int(len(lines) * TRAIN_RATIO)
train_lines = lines[:split_idx]
print(train_lines[:10])
test_lines = lines[split_idx:]

train_path = task_dir / "train.jsonl"
test_path = task_dir / "test.jsonl"

with train_path.open("w", encoding="utf-8") as f:
    f.writelines(train_lines)

with test_path.open("w", encoding="utf-8") as f:
    f.writelines(test_lines)

print(
    f"[OK] {task_dir.name}: "
    f"train={len(train_lines)}, test={len(test_lines)}"
)

print("\n✅ Split finished.")
