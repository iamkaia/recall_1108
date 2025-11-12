import os
import json
from datasets import load_dataset

# ============================================================
# RECALL-style dataset preprocessor
# 產生 instruction-based prompt, 與論文 Appendix A 對齊
# ============================================================

TASKS = ["sst2", "squad_v2", "iwslt2017", "race", "medmcqa"]
OUT_DIR = "datasets"

os.makedirs(OUT_DIR, exist_ok=True)

def save_jsonl(path, samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# =======================
# Task-specific templates
# =======================

def build_prompt(task, example):
    if task == "sst2":
        inp = (
            f"Question: What is the sentiment of the following sentence? "
            f"Answer with positive or negative.\n"
            f"Sentence: {example['sentence']}"
        )
        tgt = "positive" if example["label"] == 1 else "negative"

    elif task == "squad_v2":
        context = example["context"].strip()
        question = example["question"].strip()
        inp = f"Answer the question based on the context. If impossible, say 'no answer'.\nContext: {context}\nQuestion: {question}"
        tgt = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else "no answer"

    elif task == "iwslt2017":
        inp = f"Translate English to French:\nEnglish: {example['translation']['en']}"
        tgt = example["translation"]["fr"]

    elif task == "race":
        passage = example["article"].strip()
        question = example["question"].strip()
        options = " ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(example["options"])])
        inp = f"Read the passage and choose the correct answer (A, B, C, or D).\nPassage: {passage}\nQuestion: {question}\nOptions: {options}"
        tgt = example["answer"]

    elif task == "medmcqa":
        question = example["question"].strip()
        opts = f"(A) {example['opa']} (B) {example['opb']} (C) {example['opc']} (D) {example['opd']}"
        inp = f"Select the correct option (A, B, C, or D) for the following medical question.\nQuestion: {question}\nOptions: {opts}"
        tgt = example["cop"]

    else:
        inp = example.get("text", "")
        tgt = example.get("label", "")
    return {"input": inp, "target": tgt}


# =======================
# Build datasets
# =======================
for task in TASKS:
    print(f"🧩 Processing {task} ...")
    try:
        if task == "iwslt2017":
            raw = load_dataset("iwslt2017", "iwslt2017-en-fr")
        elif task == "squad_v2":
            raw = load_dataset("squad_v2")
        elif task == "race":
            raw = load_dataset("race", "all")
        elif task == "medmcqa":
            raw = load_dataset("medmcqa")
        else:
            raw = load_dataset(task)
    except Exception as e:
        print(f"❌ Failed to load {task}: {e}")
        continue

    for split in raw.keys():
        samples = [build_prompt(task, ex) for ex in raw[split]]
        # 把 squad_v2 存成 squad2 方便後續 evaluate
        path = f"{OUT_DIR}/{task.replace('_v2','2')}/{split}.jsonl"
        save_jsonl(path, samples)
        print(f"✅ Saved {path} ({len(samples)} samples)")

print("🎯 All datasets processed successfully (RECALL-style).")
