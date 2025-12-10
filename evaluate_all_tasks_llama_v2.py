#!/usr/bin/env python3
"""
Official-style evaluation for LLaMA-2-7B-Chat (unsloth/llama-2-7b-chat)
on the 5 RECALL tasks:

- SST-2          (Accuracy)
- SQuAD2.0       (Exact Match)
- MedMCQA        (Accuracy)
- RACE           (Accuracy)
- IWSLT2017-en-fr(Exact Match)

Assumes datasets are jsonl with fields:
    {"input": ..., "target": ...}

File layout:
    data_root/sst2/validation.jsonl
    data_root/squad2/validation.jsonl
    data_root/medmcqa/validation.jsonl
    data_root/race/test.jsonl
    data_root/iwslt2017/test.jsonl
"""

import os, re, json, argparse, string
import torch
from sacrebleu import corpus_bleu  # 你想看 BLEU 可以用，正式 metric 我們用 EM
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# ------------------- CLI args -------------------
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
                    help='例如 "sst2:500,squad2:300" 限制每 task 評估樣本數')
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=["sst2", "squad2", "iwslt2017", "race", "medmcqa"],
    )
    ap.add_argument(
        "--debug_k",
        type=int,
        default=0,
        help="每個 task 最多列印多少筆 [DEBUG task]，0 表示不印"
    )
    return ap.parse_args()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ------------------ Prompt construction ------------------
def build_instruction(task: str) -> str:
    """Instruction 部分依照任務描述撰寫。"""
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
    Instruction + Input 合在一起。
    """
    ins = build_instruction(task)
    return f"{ins}\n\nInput:\n{text}"


# ------------------ Answer parsing ------------------
def parse_answer(task: str, text: str):
    """把模型 raw_output 轉成用來算分的 pred 字串。"""
    t = (text or "").strip()

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
        if re.search(r"impossible to answer|cannot be determined|unknown|no answer",
                     low):
            return ""
        # 否則取第一行當答案
        return t.split("\n")[0].strip()

    if task == "iwslt2017":
        # 翻譯就取第一行
        return t.split("\n")[0].strip()

    return t or None


# ------------------ Normalization & metrics ------------------
def normalize_squad(s: str) -> str:
    """SQuAD2.0 官方 EM normalization：小寫、去掉 a/an/the、標點、額外空白。"""
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def normalize_iwslt(s: str) -> str:
    """IWSLT 我們用比較單純的 normalization：小寫 + 去頭尾空白 + 壓縮空白。"""
    if s is None:
        return ""
    return " ".join(s.strip().lower().split())


def compute_flag(task: str, pred: str, gold: str):
    """
    回傳這一題是否答對 (1/0)，或 pred/gold 缺失時回 None。
    指標依照 LLaMA-2 論文 Table 4：
        - SST-2 / RACE / MedMCQA: Accuracy
        - SQuAD2.0 / IWSLT2017:   Exact Match
    """
    if pred is None or gold is None:
        return None

    if task in ("sst2", "race", "medmcqa"):
        pp = (pred or "").strip().lower()
        gg = (gold or "").strip().lower()
        return int(pp == gg)

    if task == "squad2":
        return int(normalize_squad(pred) == normalize_squad(gold))

    if task == "iwslt2017":
        return int(normalize_iwslt(pred) == normalize_iwslt(gold))

    # fallback 當 Accuracy
    return int((pred or "").strip().lower() == (gold or "").strip().lower())


def compute_score(task: str, preds, golds):
    flags = []
    for p, g in zip(preds, golds):
        flags.append(compute_flag(task, p, g))

    valid = [f for f in flags if f is not None]
    if not valid:
        return 0.0, flags

    score = sum(valid) / len(valid)
    return score, flags


# ------------------ Main evaluation ------------------
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("🔍 Loading model:", args.model, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("✅ Model loaded.", flush=True)

    # decoder-only + batch generation：左側 padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 清掉不必要的 sampling flag，避免 warning 噪音
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
        # split 選擇：跟之前一樣
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
        per_sample_flags = []

        first_batch_debug_done = False
        debug_printed = 0

        num_batches = (len(data) + args.batch_size - 1) // args.batch_size

        for b_idx in tqdm(range(num_batches), desc=f"{task} generation"):
            start = b_idx * args.batch_size
            batch = data[start : start + args.batch_size]
            if not batch:
                continue

            # 收集原始 input
            batch_inputs = [ex["input"] for ex in batch]
            all_inputs.extend(batch_inputs)

            # 首 batch 印一個 example
            if not first_batch_debug_done:
                print("[DEBUG] first example:", batch[0], flush=True)

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
                    tokenize=False,  # 只回字串，避免右 padding
                )
                for msgs in messages_batch
            ]
            all_prompts.extend(prompt_texts)

            if not first_batch_debug_done:
                print("[DEBUG] prompt example:\n", prompt_texts[0], flush=True)

            # 3) 真正 tokenize：此時才會 obey padding_side='left'
            enc = tokenizer(
                prompt_texts,
                padding=True,
                truncation=True,
                max_length=args.max_src_len,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)
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

            if b_idx == 0 and decoded:
                print("[DEBUG] model output example:\n", decoded[0], flush=True)
                first_batch_debug_done = True

            batch_parsed = [parse_answer(task, t) for t in decoded]
            batch_gold = [ex["target"] for ex in batch]

            # per-sample flag + debug
            for raw_out, p, g in zip(decoded, batch_parsed, batch_gold):
                flag = compute_flag(task, p, g)
                per_sample_flags.append(flag)

                if args.debug_k > 0 and debug_printed < args.debug_k:
                    print(
                        f"[DEBUG {task}] correct={flag}, pred={p}, gold={g}",
                        flush=True,
                    )
                    debug_printed += 1

            preds.extend(batch_parsed)
            golds.extend(batch_gold)

        # 計算 metric
        score, flags = compute_score(task, preds, golds)
        summary[task] = score

        # REST: BLEU 可當參考（只對 iwslt)
        if task == "iwslt2017":
            hyps = [p for p in preds if p is not None]
            refs = [[g for g in golds if g is not None]]
            bleu = corpus_bleu(hyps, refs).score
            print(f"📌 {task} score (EM): {score:.4f}, BLEU: {bleu:.2f}", flush=True)
        elif task == "squad2":
            print(f"📌 {task} score (EM): {score:.4f}", flush=True)
        else:
            print(f"📌 {task} score (Accuracy): {score:.4f}", flush=True)

        # 寫出詳細 preds jsonl
        out_path = os.path.join(args.results_dir, f"{task}_preds.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for inp, prompt, raw, p, g, flg in zip(
                all_inputs, all_prompts, all_raw_outputs, preds, golds, flags
            ):
                rec = {
                    "input": inp,
                    "prompt": prompt,
                    "raw_output": raw,
                    "pred": p,
                    "gold": g,
                    "correct": flg,  # 1/0 或 None
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"✅ Saved {out_path}", flush=True)

    # 總結
    print("\n==== FINAL SUMMARY ====", flush=True)
    for t, v in summary.items():
        print(f"{t}: {v:.4f}", flush=True)
    if summary:
        avg = sum(summary.values()) / len(summary)
        print(f"AVG over {len(summary)} tasks: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
