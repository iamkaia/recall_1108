#!/usr/bin/env python3
"""
Official-style LLaMA-2-7B-Chat baseline evaluator (local model).

- Model: Llama-2-7b-chat-hf  (你會用本地資料夾路徑)
- Tasks: SST-2, SQuAD2, IWSLT2017, RACE, MedMCQA
- Metrics (對齊 Table 4):
    - SST-2 / RACE / MedMCQA: Accuracy
    - SQuAD2 / IWSLT2017:     Exact Match

假設 jsonl 格式為:
    {"input": ..., "target": ...}
"""

import os, re, json, argparse, string
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str,
                    default="meta-llama/Llama-2-7b-chat-hf",
                    help="可以是 HF 名稱，也可以是本地資料夾路徑")
    ap.add_argument("--data_root", type=str, default="./datasets_llama")
    ap.add_argument("--results_dir", type=str, default="./results_eval_llama2")
    ap.add_argument("--tasks", nargs="*", default=[
        "sst2", "squad2", "iwslt2017", "race", "medmcqa"
    ])
    ap.add_argument("--sample_map", type=str, default="",
                    help='格式例如 "sst2:500,squad2:300"；空字串=全部樣本')
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_src_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--debug_k", type=int, default=0,
                    help="每個 task 印出前 K 筆 debug")
    ap.add_argument("--hf_token", type=str, default=None,
                    help="若 model 來自 HF gated repo 才需要；本地路徑不用")
    return ap.parse_args()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# -------------------- Prompt building --------------------

def build_system_prompt(task: str) -> str:
    if task == "sst2":
        return "You are a helpful sentiment classification assistant."
    if task == "squad2":
        return ("You answer questions strictly using only the provided passage. "
                "If the passage does not contain the answer, reply exactly: "
                "impossible to answer.")
    if task == "iwslt2017":
        return "You are a translation assistant for English to French."
    if task == "race":
        return ("You answer multiple-choice questions and reply with exactly "
                "one letter: A, B, C, or D.")
    if task == "medmcqa":
        return ("You are a medical QA assistant. Reply with exactly one "
                "letter: A, B, C, or D.")
    return "You are a helpful assistant."


def build_user_content(task: str, text: str) -> str:
    """
    text 就是 jsonl 中的 ex["input"]。
    在前面加一點簡單的 task 說明，盡量貼近「考卷格式」。
    """
    if task == "sst2":
        return (
            "Classify the sentiment of the following sentence as positive "
            "or negative.\n\n"
            f"{text}"
        )

    if task == "squad2":
        # 假設 input 已經包含 passage + question
        return (
            "Read the following passage and answer the question. "
            "If the answer is not in the passage, reply exactly: "
            "impossible to answer.\n\n"
            f"{text}"
        )

    if task == "iwslt2017":
        return (
            "Translate the following English sentence into French.\n\n"
            f"{text}"
        )

    if task == "race":
        # input 通常已經包含 article + question + options
        return (
            "Read the article and answer the multiple-choice question. "
            "Reply with only one letter: A, B, C, or D.\n\n"
            f"{text}"
        )

    if task == "medmcqa":
        return (
            "Read the medical question and options, then reply with only "
            "one letter: A, B, C, or D.\n\n"
            f"{text}"
        )

    return text


# -------------------- Parsing --------------------

def parse_answer(task: str, text: str):
    t = (text or "").strip()

    if task == "sst2":
        low = t.lower()
        # 優先找 positive / negative
        if "positive" in low:
            return "positive"
        if "negative" in low:
            return "negative"
        return None

    if task in ("race", "medmcqa"):
        m = re.search(r"\b([A-D])\b", t.upper())
        return m.group(1) if m else None

    '''
    if task == "squad2":
        low = t.lower()
        if "impossible to answer" in low:
            return ""
        # 否則取第一行
        return t.split("\n")[0].strip()
    '''
    if task == "squad2":
        # 故意變嚴格版：直接取第一行當文字答案，
        # 不再把 "impossible to answer" 轉成空字串
        return t.split("\n")[0].strip()
    
    if task == "iwslt2017":
        # 翻譯：取第一行
        return t.split("\n")[0].strip()

    return t or None


def normalize_squad(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def normalize_iwslt(s: str) -> str:
    """
    IWSLT 用較寬鬆的 EM：
    - lower
    - 去標點
    - 去多餘空白
    """
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_flag(task: str, pred: str, gold: str):
    """
    依 Table 4:
      - SST-2 / RACE / MedMCQA: Accuracy
      - SQuAD2 / IWSLT2017:     Exact Match (帶 normalization)
    回傳 1/0 或 None
    """
    if pred is None or gold is None:
        return None

    if task in ("sst2", "race", "medmcqa"):
        pp = pred.strip().lower()
        gg = gold.strip().lower()
        return int(pp == gg)

    if task == "squad2":
        return int(normalize_squad(pred) == normalize_squad(gold))

    if task == "iwslt2017":
        return int(normalize_iwslt(pred) == normalize_iwslt(gold))

    return int(pred.strip().lower() == gold.strip().lower())


# -------------------- Main --------------------

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"🔍 Loading model: {args.model}", flush=True)

    # 如果有提供 HF token（用在遠端 gated repo），才帶進去；本地路徑就不需要
    token_kwargs = {}
    if args.hf_token:
        token_kwargs["token"] = args.hf_token

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        **token_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        **token_kwargs,
    )
    model.eval()

    # decoder-only + batch generation：左側 padding
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
        all_inputs, all_prompts, all_raw, all_flags = [], [], [], []
        debug_printed = 0

        num_batches = (len(data) + args.batch_size - 1) // args.batch_size

        for b_idx in tqdm(range(num_batches), desc=f"{task} generation"):
            batch = data[b_idx * args.batch_size:(b_idx + 1) * args.batch_size]
            if not batch:
                continue

            batch_inputs = [ex["input"] for ex in batch]
            all_inputs.extend(batch_inputs)

            # 1) system + user messages
            messages_batch = []
            for ex in batch:
                sys_prompt = build_system_prompt(task)
                user_content = build_user_content(task, ex["input"])
                messages_batch.append([
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_content},
                ])

            # 2) chat_template → prompt text
            prompt_texts = [
                tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for msgs in messages_batch
            ]
            all_prompts.extend(prompt_texts)

            # 3) tokenize（左側 padding）
            enc = tokenizer(
                prompt_texts,
                padding=True,
                truncation=True,
                max_length=args.max_src_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)
            prompt_lengths = attention_mask.sum(dim=1)

            # 4) generate
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 5) decode only generated part
            decoded = []
            for i in range(out.size(0)):
                gen_ids = out[i, prompt_lengths[i]:]
                txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
                decoded.append(txt)
            all_raw.extend(decoded)

            batch_preds = [parse_answer(task, t) for t in decoded]
            batch_golds = [ex["target"] for ex in batch]

            for p, g in zip(batch_preds, batch_golds):
                flag = compute_flag(task, p, g)
                all_flags.append(flag)

                if args.debug_k > 0 and debug_printed < args.debug_k:
                    print(f"[DEBUG {task}] pred={p} | gold={g} | correct={flag}",
                          flush=True)
                    debug_printed += 1

            preds.extend(batch_preds)
            golds.extend(batch_golds)

        # compute score
        valid_flags = [f for f in all_flags if f is not None]
        score = sum(valid_flags) / len(valid_flags) if valid_flags else 0.0
        summary[task] = score

        if task in ("squad2", "iwslt2017"):
            print(f"📌 {task} score (EM): {score:.4f}", flush=True)
        else:
            print(f"📌 {task} score (Accuracy): {score:.4f}", flush=True)

        # write jsonl
        out_path = os.path.join(args.results_dir, f"{task}_preds.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for inp, pmpt, raw, p, g, flg in zip(
                all_inputs, all_prompts, all_raw, preds, golds, all_flags
            ):
                f.write(json.dumps({
                    "input": inp,
                    "prompt": pmpt,
                    "raw_output": raw,
                    "pred": p,
                    "gold": g,
                    "correct": flg,
                }, ensure_ascii=False) + "\n")
        print(f"✅ Saved {out_path}", flush=True)

    # summary
    print("\n==== FINAL SUMMARY ====", flush=True)
    for t, v in summary.items():
        print(f"{t}: {v:.4f}", flush=True)
    if summary:
        avg = sum(summary.values()) / len(summary)
        print(f"AVG over {len(summary)} tasks: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
