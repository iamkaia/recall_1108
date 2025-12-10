'''
import os, json
from datasets import load_dataset

OUT_DIR = "datasets"
os.makedirs(OUT_DIR, exist_ok=True)

def save_jsonl(path, samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def build_prompt(task, ex):
    """Convert dataset row → InsCL/RECALL format (Table 6 style)."""

    # =============== SST-2 ===============
    if task == "sst2":
        sentiment = "positive" if ex["label"] == 1 else "negative"
        return {
            "input": (
                f"Statement: {ex['sentence']}"
                "What’s sentiment should the above sentence be?\n"
                "OPTIONS:- negative.- positive. Answer:\n"
            ),
            "target": sentiment,
        }

    # =============== SQuAD v2 ===============
    if task == "squad2":
        answer = ex["answers"]["text"][0] if len(ex["answers"]["text"]) else "impossible to answer"
        return {
            "input": (
                f"{ex['context']} "
                "According to the above passage, answer the following question. If it is impossible to answer according to the passage, answer'impossible to answer': "
                f"Question: {ex['question']}\n"
            ),
            "target": answer,
        }

    # =============== IWSLT2017 Translation ===============
    if task == "iwslt2017":
        return {
            "input": (
                "Translate the following English sentence into French.\n\n"
                f"English: {ex['translation']['en']}\n\n"
                "Answer:"
            ),
            "target": ex["translation"]["fr"],
        }

    # =============== RACE Multiple-Choice QA ===============
    if task == "race":
        options = "\n".join([f"{chr(65+i)}: {o}" for i, o in enumerate(ex["options"])])
        return {
            "input": (
                "Read the article, and answer the question by replying A, B, C or D.\n"
                f"Article:\n{ex['article']}\n\n"
                f"Q: {ex['question']}\n"
            ),
            "target": ex["answer"],
        }

    # =============== MedMCQA ===============
    if task == "medmcqa":
        # --- Convert numeric labels back to letters ---
        #label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        #target_letter = label_map.get(ex["target"], str(ex["target"]))

        map_ = {"A": 0, "B": 1, "C": 2, "D": 3}
        options = [
            f"A: {ex['opa']}",
            f"B: {ex['opb']}",
            f"C: {ex['opc']}",
            f"D: {ex['opd']}",
        ]
        return {
            "input": (
                f"Question: {ex['question']} "
                "Options:" + " ".join(options) + ' '
                "Choose an correct answer from A/B/C/D.Answer:"
            ),
            "target": map_.get(ex["cop"], None),
        }

    raise ValueError(f"⚠ Unhandled task format: {task}")


# Which datasets to load
TASKS = {
    "sst2": "sst2",
    "squad2": "squad_v2",
    "iwslt2017": ("iwslt2017", "iwslt2017-en-fr"),
    "race": ("race", "all"),
    "medmcqa": "medmcqa",
}


print("\n🔧 Building InsCL/RECALL-style datasets...\n")

for save_name, loader in TASKS.items():
    print(f"📁 Processing {save_name} ...")

    # Load HF dataset
    if isinstance(loader, tuple):
        raw = load_dataset(loader[0], loader[1])
    else:
        raw = load_dataset(loader)

    # Apply formatting
    for split in raw.keys():
        formatted = [build_prompt(save_name, ex) for ex in raw[split]]

        path = f"{OUT_DIR}/{save_name}/{split}.jsonl"
        save_jsonl(path, formatted)

        print(f"   ✔ Saved {path} ({len(formatted)} samples)")

print("\n🎯 DONE — Dataset now matches paper format.\n")
'''

import os, json
from datasets import load_dataset

OUT_DIR = "datasets"
os.makedirs(OUT_DIR, exist_ok=True)

def save_jsonl(path, samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def build_prompt(task, ex):
    """Convert dataset row → InsCL/RECALL format (Table 6 style)."""

    # =============== SST-2 ===============
    '''
    if task == "sst2":
        sentiment = "positive" if ex["label"] == 1 else "negative"
        return {
            "input": (
                f"Statement: {ex['sentence']}\n"
                "What is the sentiment of the above sentence?\n"
                "OPTIONS:\n- negative\n- positive\n"
                "Answer:"
            ),
            "target": sentiment,
        }
    '''
    if task == "sst2":
        sentiment = "positive" if ex["label"] == 1 else "negative"
        return {
            "input": (
                "Statement: "
                f"{ex['sentence']} "
                "What’s sentiment should the above sentence be?\n"
                "OPTIONS:"
                "- negative."
                "- positive."
                " Answer:\n"
            ),
            "target": sentiment
        }


    # =============== SQuAD v2 ===============
    if task == "squad2":
        answer = ex["answers"]["text"][0] if len(ex["answers"]["text"]) else "impossible to answer"
        return {
            "input": (
                f"{ex['context']}"
                "According to the passage, answer the following question."
                "If it is impossible to answer according to the passage, answer'impossible to answer':"
                f" Question: {ex['question']}\n"
            ),
            "target": answer,
        }

    # =============== IWSLT2017 Translation ===============
    if task == "iwslt2017":
        return {
            "input": (
                "Translate the following English sentence into French.\n\n"
                f"English: {ex['translation']['en']}\n\n"
                "Answer:"
            ),
            "target": ex["translation"]["fr"],
        }

    # =============== RACE Multiple-Choice QA ===============
    if task == "race":
        options = "\n".join([f"{chr(65+i)}: {o}" for i, o in enumerate(ex["options"])])
        return {
            "input": (
                "Read the article and answer the question by replying A, B, C, or D."
                f"Article:{ex['article']}"
                f"Q:{ex['question']}"
                f"OPTIONS:\n{options}\n"
            ),
            "target": ex["answer"],
        }

    # =============== MedMCQA (Fix: numeric → ABCD) ===============
    if task == "medmcqa":
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        target_letter = label_map.get(ex["cop"], str(ex["cop"]))

        options = [
            f"A: {ex['opa']}",
            f"B: {ex['opb']}",
            f"C: {ex['opc']}",
            f"D: {ex['opd']}",
        ]
        return {
            "input": (
                f"Question: {ex['question']}\n"
                "OPTIONS: " + " ".join(options) + " "
                "Choose the correct answer (A/B/C/D)."
                "Answer:"
            ),
            "target": target_letter,
        }

    raise ValueError(f"⚠ Unhandled task format: {task}")


# ----------------- HF loaders -----------------
TASKS = {
    "sst2": "sst2",
    "squad2": "squad_v2",
    "iwslt2017": ("iwslt2017", "iwslt2017-en-fr"),
    "race": ("race", "all"),
    "medmcqa": "medmcqa",
}


print("\n🔧 Building InsCL/RECALL-style datasets...\n")

for save_name, loader in TASKS.items():
    print(f"📁 Processing {save_name} ...")

    if isinstance(loader, tuple):
        raw = load_dataset(loader[0], loader[1])
    else:
        raw = load_dataset(loader)

    for split in raw.keys():
        formatted = [build_prompt(save_name, ex) for ex in raw[split]]

        path = f"{OUT_DIR}/{save_name}/{split}.jsonl"
        save_jsonl(path, formatted)

        print(f"   ✔ Saved {path} ({len(formatted)} samples)")

print("\n🎯 DONE — Dataset now matches InsCL/RECALL paper format.\n")
