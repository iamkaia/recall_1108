#!/usr/bin/env python3
"""
Zero-shot evaluation for LLaMA-2-7B-CHAT (unsloth/llama-2-7b-chat)

- 使用 chat_template（因為是 Chat 模型）
- apply_chat_template 只產生文字，不做 tokenization
- 真正 tokenize 用 tokenizer(...)，支援 left padding
- 避免 decoder-only + right-padding warning
- 任務：SST-2 / SQuAD2 / IWSLT2017 / RACE / MedMCQA
- 加上 tqdm 進度條 + DEBUG 範例輸出
"""

import os, re, json, argparse, string
import torch
from sacrebleu import corpus_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm   # ✅ 進度條


# ------------------- Args -------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="例如 unsloth/llama-2-7b-chat")
    ap.add_argument("--data_root", type=str, default="./datasets_llama")
    ap.add_argument("--results_dir", type=str, default="./results_eval")
    ap.add_argument("--max_src_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--sample_map", type=str, default="",
                    help='例如 "sst2:500,squad2:300"')
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=["sst2", "squad2", "iwslt2017", "race", "medmcqa"],
    )
    return ap.parse_args()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ------------------ Task Prompts ------------------
def build_instruction(task: str) -> str:
    if task == "sst2":
        return "Classify the sentiment of the given sentence as positive or negative."

    if task == "squad2":
        return (
            "Read the passage and answer the question. "
            "If the answer is not mentioned, respond exactly: impossible to answer."
        )

    if task == "iwslt2017":
        return "Translate the following English sentence into French."

    if task == "race":
        return "Read the article and answer the question by selecting A, B, C, or D."

    if task == "medmcqa":
        return "Choose the correct answer (A, B, C, or D)."

    return "Perform the task."


def build_user_prompt(task: str, text: str) -> str:
    """
    給 chat_template 的 user content：
    Instruction + Input 合在一起
    """
    ins = build_instruction(task)
    return f"{ins}\n\nInput:\n{text}"


# ------------------ Parsing ------------------
def parse_answer(task: str, text: str):
    t = text.strip()

    if task == "sst2":
        low = t.lower()
        m = re.search(r"\b(positive|negative)\b", low)
        if m:
            return m.group(1)
        if "positive" in low:
            return "positive"
        if "negative" in low:
            return "negative"
        return None

    if task in ("race", "medmcqa"):
        m = re.search(r"\b([A-D])\b", t.upper())
        return m.group(1) if m else None

    if task == "squad2":
        low = t.lower()
        if re.search(r"impossible to answer|cannot be determined|unknown|no answer", low):
            return ""
        return t.split("\n")[0].strip().lower()

    if task == "iwslt2017":
        return t.split("\n")[0].strip()

    return t


# ------------------ Metric ------------------
def normalize_squad(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def metric(task, preds, golds):
    pairs = [(p, g) for p, g in zip(preds, golds) if p is not None and g is not None]
    if not pairs:
        return 0.0

    if task in ("sst2", "race", "medmcqa"):
        correct = sum(
            1
            for p, g in pairs
            if p.strip().lower() == g.strip().lower()
        )
        return correct / len(pairs)

    if task == "squad2":
        correct = sum(
            1
            for p, g in pairs
            if normalize_squad(p) == normalize_squad(g)
        )
        return correct / len(pairs)

    if task == "iwslt2017":
        hyps = [p.strip() for p, _ in pairs]
        refs = [g.strip() for _, g in pairs]
        return corpus_bleu(hyps, [refs]).score

    return 0.0


# ------------------ Main ------------------
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("🔍 Loading model:", args.model, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("✅ Model loaded.", flush=True)

    # decoder-only + batch generation：左側 padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 可選：把 generation_config 裡的 sampling flag 清掉，避免沒用的 warning
    gen_cfg = model.generation_config
    for k in ["temperature", "top_p"]:
        if hasattr(gen_cfg, k):
            setattr(gen_cfg, k, None)
    model.generation_config = gen_cfg

    # parse --sample_map
    sample_map = {}
    if args.sample_map:
        for kv in args.sample_map.split(","):
            t, n = kv.split(":")
            sample_map[t] = int(n)

    summary = {}

    print("👉 Tasks:", args.tasks, flush=True)

    for task in args.tasks:
        split = "validation" if task in ("sst2", "squad2", "medmcqa") else "test"
        path = os.path.join(args.data_root, task, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"⚠ {path} not found, skip {task}.", flush=True)
            continue

        data = load_jsonl(path)
        if task in sample_map:
            data = data[: sample_map[task]]

        print(f"\n=== Evaluating {task} ({len(data)} samples) ===", flush=True)

        preds, golds = [], []
        all_inputs, all_prompts, all_raw_outputs = [], [], []
        first_batch_debug_done = False

        # tqdm 進度條
        num_batches = (len(data) + args.batch_size - 1) // args.batch_size
        for b_idx in tqdm(range(num_batches), desc=f"{task} generation"):
            start = b_idx * args.batch_size
            batch = data[start : start + args.batch_size]
            batch_inputs = [ex["input"] for ex in batch]
            all_inputs.extend(batch_inputs)


            if not batch:
                continue

            # 首 batch：印第一筆 example
            if not first_batch_debug_done:
                print("\n[DEBUG] first example:", batch[0], flush=True)

            # 1) 準備 chat messages（只 user 一輪）
            messages_batch = []
            for ex in batch:
                user_content = build_user_prompt(task, ex["input"])
                messages_batch.append(
                    [
                        {"role": "user", "content": user_content}
                    ]
                )

            # 2) 用 chat_template 產生「純文字 prompt」（不做 tokenization）
            prompt_texts = [
                tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=False,        # ⬅ 只要文字，不要 tensor
                )
                for msgs in messages_batch
            ]
            all_prompts.extend(prompt_texts)

            # 首 batch：印 prompt 範例
            if not first_batch_debug_done:
                print("[DEBUG] prompt example:\n", prompt_texts[0], flush=True)
                first_batch_debug_done = True

            # 3) 真正 tokenize：這裡才會套用 padding_side='left'
            enc = tokenizer(
                prompt_texts,
                padding=True,              # left padding
                truncation=True,
                max_length=args.max_src_len,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            # 每個樣本的 prompt 長度（非 pad token 的數量）
            prompt_lengths = attention_mask.sum(dim=1).tolist()

            # 4) 生成
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 5) 只 decode 新生成的部分
            decoded = []
            for i_b in range(out.size(0)):
                gen_ids = out[i_b, prompt_lengths[i_b]:]
                txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
                decoded.append(txt)
            all_raw_outputs.extend(decoded)

            # 首 batch：也可以印一筆 model output
            if b_idx == 0 and decoded:
                print("[DEBUG] model output example:\n", decoded[0], flush=True)

            batch_parsed = [parse_answer(task, t) for t in decoded]
            batch_gold = [ex["target"] for ex in batch]

            preds.extend(batch_parsed)
            golds.extend(batch_gold)

        score = metric(task, preds, golds)
        summary[task] = score
        print(f"📌 {task} score: {score:.4f}", flush=True)

        # 存 preds
        out_path = os.path.join(args.results_dir, f"{task}_preds.jsonl")
        '''
        with open(out_path, "w", encoding="utf-8") as f:
            for p, g in zip(preds, golds):
                f.write(json.dumps({"pred": p, "gold": g}, ensure_ascii=False) + "\n")
        '''
        with open(out_path, "w", encoding="utf-8") as f:
            for inp, prompt, raw, p, g in zip(
                all_inputs, all_prompts, all_raw_outputs, preds, golds
            ):
                rec = {
                    "input": inp,          # dataset 裡的 input
                    "prompt": prompt,      # 給 chat_template 之後的完整 prompt 文字
                    "raw_output": raw,     # model decode 出來的原始文字
                    "pred": p,             # parse_answer 後的最終答案（用來算 metric）
                    "gold": g,             # ground truth target
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"✅ Saved {out_path}", flush=True)

    print("\n==== FINAL SUMMARY ====", flush=True)
    for t, v in summary.items():
        print(f"{t}: {v}", flush=True)
    if summary:
        avg = sum(summary.values()) / len(summary)
        print(f"AVG: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
