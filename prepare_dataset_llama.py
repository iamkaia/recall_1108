#!/usr/bin/env python3
import os, json
from datasets import load_dataset

OUT_DIR = "datasets_llama"
os.makedirs(OUT_DIR, exist_ok=True)

def save_jsonl(path, samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def format_row(task, ex):
    """Convert original dataset → Minimal raw form (no prompt)."""

    # ------------------- SST2 (binary sentiment) -------------------
    if task == "sst2":
        label = "positive" if ex["label"] == 1 else "negative"
        return {"input": ex["sentence"], "target": label}

    # ------------------- SQuAD2 -------------------
    if task == "squad2":
        answer = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        return {"input": f"{ex['context']}\n\nQuestion: {ex['question']}", 
                "target": answer}

    # ------------------- IWSLT2017 (translation) -------------------
    if task == "iwslt2017":
        return {
            "input": ex["translation"]["en"],
            "target": ex["translation"]["fr"]
        }

    # ------------------- RACE (multiple choice) -------------------
    if task == "race":
        options = ex["options"]
        text = f"{ex['article']}\n\nQuestion: {ex['question']}\nOptions:\n"
        text += "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
        return {"input": text, "target": ex["answer"].upper()}

    # ------------------- MedMCQA (ABCD) -------------------
    if task == "medmcqa":
        label_map = {0:"A",1:"B",2:"C",3:"D"}
        target = label_map.get(ex["cop"], None)
        return {
            "input": ex["question"] + "\n" +
                     "\n".join([
                         f"A: {ex['opa']}",
                         f"B: {ex['opb']}",
                         f"C: {ex['opc']}",
                         f"D: {ex['opd']}",
                     ]),
            "target": target
        }

    raise ValueError(f"Unknown task: {task}")


# ------------------- HF loaders -------------------
TASKS = {
    "sst2": "sst2",
    "squad2": "squad_v2",
    "iwslt2017": ("iwslt2017", "iwslt2017-en-fr"),
    "race": ("race", "all"),
    "medmcqa": "medmcqa",
}

print("\n🔧 Building Zero-shot evaluation dataset…\n")

for name, loader in TASKS.items():
    print(f"📁 Processing {name} ...")

    raw = load_dataset(loader[0], loader[1]) if isinstance(loader, tuple) else load_dataset(loader)

    for split in raw.keys():
        formatted = [format_row(name, ex) for ex in raw[split]]
        save_path = f"{OUT_DIR}/{name}/{split}.jsonl"
        save_jsonl(save_path, formatted)
        print(f"   ✔ Saved {save_path} ({len(formatted)} samples)")

print("\n🎯 DONE — dataset is now clean for zero-shot evaluation.\n")
