import os, json, argparse
from datasets import load_dataset

def save_text_data(dataset_split, split_name, out_dir, input_key, target_key):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in dataset_split:
            inp = str(ex[input_key]).strip().replace("\n", " ")
            tgt = str(ex[target_key]).strip().replace("\n", " ")
            f.write(json.dumps({"input": inp, "output": tgt}) + "\n")
    print(f"✅ Saved {split_name} ({len(dataset_split)}) → {out_path}")

# ---------------- SST-2 ----------------
def prepare_sst2(include_test=False):
    print("🧩 Preparing SST-2 ...")
    ds = load_dataset("glue", "sst2")
    out_dir = "datasets/sst2"
    save_text_data(ds["train"], "train", out_dir, "sentence", "label")
    save_text_data(ds["validation"], "validation", out_dir, "sentence", "label")
    if include_test and "test" in ds:
        save_text_data(ds["test"], "test", out_dir, "sentence", "label")
    else:
        # GLUE test set沒有標籤，用validation代替
        save_text_data(ds["validation"], "test", out_dir, "sentence", "label")

# ---------------- SQuAD2 ----------------
def prepare_squad2(include_test=False):
    print("🧩 Preparing SQuAD2.0 ...")
    ds = load_dataset("squad_v2")
    out_dir = "datasets/squad2"
    save_text_data(ds["train"], "train", out_dir, "question", "answers")
    save_text_data(ds["validation"], "validation", out_dir, "question", "answers")
    if include_test and "test" in ds:
        save_text_data(ds["test"], "test", out_dir, "question", "answers")
    else:
        save_text_data(ds["validation"], "test", out_dir, "question", "answers")

# ---------------- IWSLT2017 (en-fr) ----------------
def prepare_iwslt(include_test=False):
    print("🧩 Preparing IWSLT2017 (en-fr) ...")
    ds = load_dataset("iwslt2017", "iwslt2017-en-fr")
    out_dir = "datasets/iwslt2017"
    for split in ["train", "validation"]:
        save_text_data(ds[split], split, out_dir, "translation", "translation")
    if include_test and "test" in ds:
        save_text_data(ds["test"], "test", out_dir, "translation", "translation")
    else:
        save_text_data(ds["validation"], "test", out_dir, "translation", "translation")

# ---------------- RACE ----------------
def prepare_race(include_test=False):
    print("🧩 Preparing RACE ...")
    ds = load_dataset("race", "high")
    out_dir = "datasets/race"
    save_text_data(ds["train"], "train", out_dir, "article", "answer")
    save_text_data(ds["validation"], "validation", out_dir, "article", "answer")
    if include_test and "test" in ds:
        save_text_data(ds["test"], "test", out_dir, "article", "answer")
    else:
        save_text_data(ds["validation"], "test", out_dir, "article", "answer")

# ---------------- MedMCQA ----------------
def prepare_medmcqa(include_test=False):
    print("🧩 Preparing MedMCQA ...")
    ds = load_dataset("medmcqa")
    out_dir = "datasets/medmcqa"
    save_text_data(ds["train"], "train", out_dir, "question", "cop")
    save_text_data(ds["validation"], "validation", out_dir, "question", "cop")
    if include_test and "test" in ds:
        save_text_data(ds["test"], "test", out_dir, "question", "cop")
    else:
        save_text_data(ds["validation"], "test", out_dir, "question", "cop")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_test", action="store_true", help="also export test split (if not available, use validation as test)")
    args = parser.parse_args()

    print("Downloading and processing datasets...")
    prepare_sst2(args.include_test)
    prepare_squad2(args.include_test)
    prepare_iwslt(args.include_test)
    prepare_race(args.include_test)
    prepare_medmcqa(args.include_test)
    print("✅ All datasets prepared under ./datasets/")
