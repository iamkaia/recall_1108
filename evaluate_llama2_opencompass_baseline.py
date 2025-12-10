#!/usr/bin/env python3
"""
OpenCompass-official-prompt baseline evaluator for Llama-2-7B-Chat

📌 Features
- 100% match OpenCompass prompt templates (5 tasks)
- Works on your {input, target} dataset format
- Decoder-only left padding
- Exact Match (SQuAD2 / IWSLT2017)
- Accuracy (SST2 / RACE / MedMCQA)
- Outputs debug + full preds jsonl
- Compatible with local model path

Author: ChatGPT x Kaia Project
"""

import os, json, re, argparse, string
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Task → OpenCompass Prompt Templates
# -----------------------------
def build_prompt_opencompass(task: str, user_input: str) -> str:
    """
    Build prompts exactly like OpenCompass official configs.
    user_input = your jsonl["input"] string
    We assume it already contains full context, question, article, options, etc.
    """

    if task == "sst2":
        # glue_sst2.py
        return (
            f"{user_input}\n"
            "Question: What is the sentiment of the sentence? "
            "Options: positive or negative?\n"
            "Answer:"
        )

    if task == "squad2":
        # squad2.py
        # user_input already contains context + question
        return (
            f"Context:\n{user_input}\n\n"
            "Answer:"
        )

    if task == "iwslt2017":
        # iwslt2017_en_fr.py
        return (
            "Translate the following sentence to French:\n"
            f"{user_input}\n\n"
            "Translation:"
        )

    if task == "race":
        # race_p1.py
        # user_input should include article + question + options
        return (
            f"{user_input}\n\n"
            "Answer:"
        )

    if task == "medmcqa":
        # medmcqa_gen.py
        # user_input should contain the question + options A/B/C/D
        return (
            f"{user_input}\n\n"
            "Answer:"
        )

    # Default fallback
    return user_input + "\nAnswer:"


# -----------------------------
# Parsing Rules
# -----------------------------
def parse_answer(task: str, text: str):
    """
    Extract model answer from generated text.
    No instruction filter — pure OpenCompass style.
    """
    if text is None:
        return None
    t = text.strip()

    if task in ("sst2",):
        # positive/negative
        low = t.lower()
        if "positive" in low:
            return "positive"
        if "negative" in low:
            return "negative"
        return None

    if task in ("race", "medmcqa"):
        # A/B/C/D
        m = re.search(r"\b([A-D])\b", t.upper())
        return m.group(1) if m else None

    if task == "squad2":
        # strict EM → do NOT map "impossible to answer" → ""
        return t.split("\n")[0].strip()

    if task == "iwslt2017":
        # Translation: first line only
        return t.split("\n")[0].strip()

    return t


# -----------------------------
# Normalize functions
# -----------------------------
def normalize_squad(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def normalize_iwslt(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


# -----------------------------
# Compute metrics
# -----------------------------
def compute_flag(task: str, pred: str, gold: str):
    if pred is None or gold is None:
        return None

    if task in ("sst2", "race", "medmcqa"):
        # Accuracy
        return int(pred.strip().lower() == gold.strip().lower())

    if task == "squad2":
        # strict EM
        return int(normalize_squad(pred) == normalize_squad(gold))

    if task == "iwslt2017":
        # strict EM
        return int(normalize_iwslt(pred) == normalize_iwslt(gold))

    return int(pred.strip() == gold.strip())


# ============================================================
# Main Evaluator
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="HF repo 或本地 Llama-2-7b-chat-hf 目錄")
    ap.add_argument("--data_root", type=str, default="./datasets_llama")
    ap.add_argument("--results_dir", type=str, default="./results_eval_oc")
    ap.add_argument("--tasks", nargs="*", default=[
        "sst2", "squad2", "iwslt2017", "race", "medmcqa"
    ])
    ap.add_argument("--sample_map", type=str, default="",
                    help='例如 "sst2:500,squad2:300"')
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_src_len", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--debug_k", type=int, default=5)
    ap.add_argument("--hf_token", type=str, default=None,
                    help="若 model 來自 gated repo 才需要。本地模型不用。")
    return ap.parse_args()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"🔍 Loading model: {args.model}", flush=True)

    # HF token (only if needed)
    token_kwargs = {}
    if args.hf_token:
        token_kwargs["token"] = args.hf_token

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        **token_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        **token_kwargs,
    )
    model.eval()

    # Llama-2-chat 是 decoder-only → 左側 padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 解析 sample_map
    sample_map = {}
    if args.sample_map:
        for kv in args.sample_map.split(","):
            t, n = kv.split(":")
            sample_map[t] = int(n)

    summary = {}

    for task in args.tasks:
        split = "validation" if task in ("sst2", "squad2", "medmcqa") else "test"
        path = os.path.join(args.data_root, task, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"⚠ Missing dataset {path}, skip.", flush=True)
            continue

        data = load_jsonl(path)
        if task in sample_map:
            data = data[: sample_map[task]]

        print(f"\n=== Evaluating {task} ({len(data)} samples) ===", flush=True)

        preds, golds = [], []
        all_inputs, all_prompts, all_raw = [], [], []
        all_flags = []
        debug_printed = 0

        num_batches = (len(data) + args.batch_size - 1) // args.batch_size

        for b in tqdm(range(num_batches), desc=f"{task} generation"):
            batch = data[b * args.batch_size:(b+1) * args.batch_size]
            if not batch:
                continue

            batch_inputs = [ex["input"] for ex in batch]
            batch_golds  = [ex["target"] for ex in batch]

            all_inputs.extend(batch_inputs)

            # ------ OpenCompass Prompt ------
            prompts = [
                build_prompt_opencompass(task, inp)
                for inp in batch_inputs
            ]
            all_prompts.extend(prompts)

            # ------ Tokenize ------
            enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=args.max_src_len,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)
            prompt_lengths = attention_mask.sum(dim=1)

            # ------ Generate ------
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # ------ Decode only generated tokens ------
            decoded = []
            for i in range(outputs.size(0)):
                gen_ids = outputs[i, prompt_lengths[i]:]
                txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
                decoded.append(txt)
            all_raw.extend(decoded)

            # ------ Parse prediction ------
            batch_preds = [parse_answer(task, t) for t in decoded]

            for p, g in zip(batch_preds, batch_golds):
                flag = compute_flag(task, p, g)
                all_flags.append(flag)

                if args.debug_k > 0 and debug_printed < args.debug_k:
                    print(f"[DEBUG {task}] pred={p} | gold={g} | correct={flag}",
                          flush=True)
                    debug_printed += 1

            preds.extend(batch_preds)
            golds.extend(batch_golds)

        # ------ Compute final score ------
        valid = [f for f in all_flags if f is not None]
        score = sum(valid) / len(valid) if valid else 0.0
        summary[task] = score

        metric_name = "Accuracy" if task in ("sst2","race","medmcqa") else "EM"
        print(f"📌 {task} {metric_name}: {score:.4f}", flush=True)

        # ------ Write jsonl output ------
        out_path = os.path.join(args.results_dir, f"{task}_preds.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for inp, prm, raw, p, g, flg in zip(
                all_inputs, all_prompts, all_raw, preds, golds, all_flags
            ):
                f.write(json.dumps({
                    "input": inp,
                    "prompt": prm,
                    "raw_output": raw,
                    "pred": p,
                    "gold": g,
                    "correct": flg,
                }, ensure_ascii=False) + "\n")
        print(f"✅ Saved {out_path}", flush=True)

    # ------ Print summary ------
    print("\n==== FINAL SUMMARY ====")
    for t, v in summary.items():
        print(f"{t}: {v:.4f}")
    if summary:
        avg = sum(summary.values()) / len(summary)
        print(f"AVG over {len(summary)} tasks: {avg:.4f}")


if __name__ == "__main__":
    main()
