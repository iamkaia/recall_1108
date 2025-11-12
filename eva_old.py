import argparse, json, os, torch
import evaluate as hf_eval
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from peft import PeftModel
from tqdm import tqdm
from sacrebleu import corpus_bleu
from sklearn.metrics import f1_score

TASKS = ["sst2", "squad2", "iwslt2017", "race", "medmcqa"]

def load_local_dataset(task, split="test", max_examples=200):
    path = f"datasets/{task}/{split}.jsonl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset file not found: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            data.append(json.loads(line))
    print(f"✅ Loaded {len(data)} samples from {path}")
    return Dataset.from_list(data)

def compute_metric(task, preds, refs):
    if task in ["sst2", "race", "medmcqa"]:
        # 簡單分類準確率
        return sum(p.strip().lower() == r.strip().lower() for p, r in zip(preds, refs)) / len(refs)

    elif task == "squad2":
        # 用字串重疊近似計算 F1
        def simple_f1(pred, ref):
            pred_tokens, ref_tokens = pred.split(), ref.split()
            common = len(set(pred_tokens) & set(ref_tokens))
            if common == 0:
                return 0
            precision = common / len(pred_tokens)
            recall = common / len(ref_tokens)
            return 2 * precision * recall / (precision + recall)
        f1s = [simple_f1(p, r) for p, r in zip(preds, refs)]
        return sum(f1s) / len(f1s)

    elif task == "iwslt2017":
        # 使用 sacrebleu 本地版
        return corpus_bleu(preds, [refs]).score / 100.0

    else:
        return 0.0


def evaluate(model_dir, base_model, split="test", max_examples=200):
    print(f"🚀 Starting evaluation on split={split}, subset={max_examples}")
    print(f"🔹 Base model: {base_model}")
    print(f"🔹 Fused model: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, model_dir)

    results = {}
    os.makedirs("logs", exist_ok=True)
    result_file = "recall_results_500.txt"
    with open(result_file, "w") as f:
        for task in TASKS:
            try:
                print(f"\n🧩 Evaluating {task} ({split}) ...")
                ds = load_local_dataset(task, split, max_examples)
                inputs = [x["input"] for x in ds]
                refs = [str(x["output"]) for x in ds]
                preds = []

                for inp in tqdm(inputs, desc=f"{task} generation"):
                    prompt = f"{inp}\nAnswer:"
                    out = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_new_tokens=64)
                    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
                    preds.append(decoded.strip())

                acc = compute_metric(task, preds, refs)
                results[task] = acc
                print(f"✅ {task}: {acc:.3f}")
                f.write(f"{task}: {acc:.3f}\n")

            except Exception as e:
                print(f"❌ Error evaluating {task}: {e}")
                f.write(f"{task}: ERROR ({e})\n")

        if len(results) > 0:
            avg = sum(results.values()) / len(results)
            print(f"\n📊 Average accuracy across tasks: {avg:.3f}")
            f.write(f"avg: {avg:.3f}\n")
        else:
            print("❌ No valid results computed (empty results dictionary).")

    # 同時輸出成 summary_table.csv
    import csv
    csv_path = "summary_table.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["task", "accuracy"])
        for t, a in results.items():
            writer.writerow([t, a])
        if len(results) > 0:
            writer.writerow(["avg", sum(results.values()) / len(results)])
    print(f"[✅ Saved summary to {csv_path}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_examples", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.model, args.base_model, args.split, args.max_examples)
