'''
#!/usr/bin/env python3
# evaluate_all_tasks.py — RECALL benchmark evaluation (with beam debug)

import os, re, json, string, argparse, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Args --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./datasets")
    ap.add_argument("--results_dir", type=str, default="./results_eval")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_src_len", type=int, default=768)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument(
        "--tasks", nargs="*", default=["sst2", "squad2", "iwslt2017", "race", "medmcqa"]
    )
    return ap.parse_args()

# -------------------- Utils --------------------
def load_jsonl(path): return [json.loads(l) for l in open(path, "r", encoding="utf-8")]
def collate_fn(b): return b

def parse_first_valid_token(task: str, text: str):
    t = text.strip()
    if task == "sst2":
        m = re.search(r"answer\s*[:：]?\s*(positive|negative)", t, re.I)
        if m: return m.group(1).lower()
        m = re.search(r"(positive|negative)", t.lower())
        return m.group(1) if m else None
    if task in ("race", "medmcqa"):
        m = re.search(r"[Aa]nswer\s*[:：]?\s*([A-Da-d])", t)
        if not m: m = re.search(r"\b([A-Da-d])\b", t)
        return {"A":0,"B":1,"C":2,"D":3}[m.group(1).upper()] if m else None
    if task == "squad2":
        m = re.search(r"[Aa]nswer\s*[:：]\s*(.*)", t)
        ans = (m.group(1) if m else t.split("\n",1)[0]).strip()
        if re.search(r"\b(no\s*answer|unanswerable|cannot\s*be\s*determined|unknown)\b", ans, re.I):
            return ""
        return re.sub(r"[.,]+$", "", ans.strip().lower())
    if task == "iwslt2017":
        m = re.search(r"[Aa]nswer\s*[:：]\s*(.*)", t)
        if m:
            return m.group(1).split("\n")[0].strip()
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        return lines[-1] if lines else ""
    return None

def get_target(task: str, ex: Dict):
    tgt = ex.get("target", None)
    if tgt is None: return None
    if task == "medmcqa":
        if int(tgt) == -1: return None
        return int(tgt)
    if task == "race":
        m = {"A":0,"B":1,"C":2,"D":3}
        return m.get(str(tgt).strip().upper(), None)
    if task == "sst2": return str(tgt).strip().lower()
    if task == "squad2":
        g = str(tgt).strip()
        if g == "" or g.lower() in {"no answer","unanswerable","none","n/a"}: return ""
        return g.lower()
    return str(tgt).strip()

# 官方 SQuAD normalization
def normalize_answer(s):
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t): return "".join(ch for ch in t if ch not in string.punctuation)
    def lower(t): return t.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def simple_em(pred,gold): return int(normalize_answer(pred)==normalize_answer(gold))

def simple_bleu(preds, refs):
    try: import sacrebleu
    except: return None
    preds, refs = [(p or "").strip() for p in preds], [(r or "").strip() for r in refs]
    if not preds or not refs or all(x=="" for x in preds) or all(x=="" for x in refs): return None
    return sacrebleu.corpus_bleu(preds,[refs]).score

# -------------------- Main --------------------
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("🔄 Loading model & tokenizer ...")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    if hasattr(tok, "chat_template"): tok.chat_template = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    summary = {}

    for task in args.tasks:
        split = "validation" if task in ("sst2","medmcqa") else "test"
        path = os.path.join(args.data_root, task, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"[WARN] {path} not found, skip.")
            continue

        data = load_jsonl(path)
        if args.max_examples > 0: data = data[:args.max_examples]
        print(f"\n=== Evaluating {task} ({len(data)} samples, src_len={args.max_src_len}) ===")

        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        preds,golds,results = [],[],[]

        # generation config
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if task in ("iwslt2017","race"):  # beam search
            gen_kwargs.update(dict(num_beams=2, length_penalty=1.0, early_stopping=True))

        for batch in tqdm(loader, desc=f"{task} generation"):
            inputs = [ex["input"] + "\nAnswer:" for ex in batch]
            gold = [get_target(task, ex) for ex in batch]

            enc = tok(inputs, padding=True, truncation=True,
                      max_length=args.max_src_len, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                out = model.generate(**enc, **gen_kwargs)
            decoded = tok.batch_decode(out.sequences, skip_special_tokens=True)

            # debug: 顯示實際用哪條 beam
            if hasattr(out, "sequences_scores"):
                for j in range(min(2, len(out.sequences_scores))):
                    sc = out.sequences_scores[j].item()
                    print(f"[BEAM INFO {task}] sample#{j} score={sc:.3f}  text='{decoded[j].splitlines()[0][:120]}'")

            parsed = [parse_first_valid_token(task, t) for t in decoded]
            if task in ("sst2","race","medmcqa"):
                for p,g in zip(parsed,gold):
                    print(f"[DEBUG {task}] parsed={p}, gold={g}, correct={p==g}")

            for inp,raw,p,g in zip(inputs,decoded,parsed,gold):
                results.append({"input":inp,"prediction":raw,"parsed":p,"gold":g})
            preds.extend(parsed); golds.extend(gold)

        # 寫出結果
        out_path = os.path.join(args.results_dir, f"{task}_preds.jsonl")
        with open(out_path,"w",encoding="utf-8") as f:
            for r in results: f.write(json.dumps(r,ensure_ascii=False)+"\n")
        print(f"✅ Saved predictions → {out_path}")

        # 計分
        score_val = None
        if task in ("sst2","race","medmcqa"):
            valid = [(p,g) for p,g in zip(preds,golds) if p is not None and g is not None]
            acc = sum(int(p==g) for p,g in valid)/max(1,len(valid))
            print(f"   ↳ Accuracy: {acc:.3f} ({sum(int(p==g) for p,g in valid)}/{len(valid)})")
            score_val = acc
        elif task=="squad2":
            valid = [(p,g) for p,g in zip(preds,golds) if p is not None and g is not None and g!=""]
            for i,(p,g) in enumerate(valid[:5]):
                print(f"[DEBUG squad2] EM={int(simple_em(p,g))}, pred={p}, gold={g}")
            em = sum(simple_em(p,g) for p,g in valid)/max(1,len(valid))
            print(f"   ↳ EM: {em:.3f}")
            score_val = em
        elif task=="iwslt2017":
            bleu = simple_bleu(preds,golds)
            print(f"   ↳ BLEU: {bleu:.2f}" if bleu else "   ↳ BLEU: skipped")
            score_val = bleu

        summary[task] = score_val

    # summary table
    valid_scores = [v for v in summary.values() if v is not None]
    if valid_scores:
        avg = sum(valid_scores)/len(valid_scores)
        print(f"\n=== SUMMARY AVG: {avg:.3f} ===")
        csv = os.path.join(args.results_dir, "summary_table.csv")
        with open(csv,"w") as f:
            f.write("task,score\n")
            [f.write(f"{k},{v:.6f}\n") for k,v in summary.items() if v is not None]
            f.write(f"avg,{avg:.6f}\n")

if __name__=="__main__":
    main()
'''
'''
#!/usr/bin/env python3
# evaluate_all_tasks.py — RECALL paper evaluation (greedy decode + no chat template)

import os
import re
import json
import argparse
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


# -----------------------------
# parsing and scoring functions
# -----------------------------
def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def collate_fn(batch):
    return batch


def parse_first_valid_token(task: str, text: str):
    t = text.strip()

    if task == "sst2":
        m = re.search(r"\b(positive|negative)\b", t.lower())
        return m.group(1) if m else None

    if task in ("race", "medmcqa"):
        m = re.search(r"\b([A-Da-d])\b", t)
        return m.group(1).upper() if m else None

    if task == "squad2":
        m = re.search(r"[Aa]nswer\s*[:：]\s*(.+)", t)
        return (m.group(1) if m else t.split("\n", 1)[0]).strip().lower()

    if task == "iwslt2017-en-fr":
        return t.split("\n", 1)[0].strip()

    return None


def get_target(task: str, ex: Dict):
    tgt = ex.get("target", None)
    if tgt is None:
        return None

    if task in ("race", "medmcqa"):
        if str(tgt).strip() == "-1":
            return None
        return str(tgt).strip().upper()

    if task == "sst2":
        return str(tgt).strip().lower()

    if task == "squad2":
        return str(tgt).strip().lower()

    if task == "iwslt2017-en-fr":
        return str(tgt).strip()

    return str(tgt).strip()


def simple_em(pred: str, gold: str) -> int:
    return int((pred or "").strip() == (gold or "").strip())


def simple_bleu(preds, refs):
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(preds, [refs]).score
    except Exception:
        return None


# -----------------------------
# ✨ key function: clean generation (no prompt in output)
# -----------------------------
def generate_batch(model, tokenizer, batch_inputs, max_src_len, max_new_tokens):
    """Generate ONLY new output tokens (no prompt decoding)."""

    enc = tokenizer(
        batch_inputs,
        padding=True,               # use pad_token (already linked to eos)
        truncation=True,
        max_length=max_src_len,
        add_special_tokens=False,   # 🚫 chat template / system prompt
        return_tensors="pt",
    ).to(model.device)

    # input length per sample (needed to remove prompt from decoding)
    input_ids = enc["input_ids"]
    input_lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

    # only decode NEW tokens
    results = []
    for i, seq in enumerate(out.sequences):
        new_tokens = seq[input_lengths[i]:]     # <--- slice prompt away
        txt = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(txt)

    return results


# -----------------------------
# main evaluation
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--data_root", default="./datasets")
    ap.add_argument("--results_dir", default="./results_eval")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_src_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--tasks", nargs="*", default=["sst2", "squad2", "iwslt2017-en-fr", "race", "medmcqa"])
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("🔄 Loading tokenizer & model...")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # ✅ 必做：pad_token = eos_token，避免 padding crash
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ✅ decoder-only (Qwen/LLaMA) 必做：左 padding
    tokenizer.padding_side = "left"

    # ✅ 關閉 chat 模式（最關鍵的一步！）
    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = None
    if hasattr(tokenizer, "apply_chat_template"):
        tokenizer.apply_chat_template = lambda messages, **kwargs: "".join(
            [m["content"] for m in messages]
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    summary = {}

    for task in args.tasks:
        test_path = os.path.join(args.data_root, task, "test.jsonl")
        if not os.path.exists(test_path):
            print(f"[WARN] {test_path} missing, skip.")
            continue

        data = load_jsonl(test_path)
        if args.max_examples > 0:
            data = data[: args.max_examples]

        print(f"\n=== Evaluating {task} ({len(data)} samples) ===")

        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        all_preds_text, parsed_preds, golds = [], [], []

        for batch in tqdm(dataloader, desc=f"{task} generation"):
            batch_inputs = [ex["input"] + "\nAnswer:" for ex in batch]
            batch_golds = [get_target(task, ex) for ex in batch]

            # ✅ 使用乾淨的 generate_batch
            texts = generate_batch(
                model=model,
                tokenizer=tokenizer,
                batch_inputs=batch_inputs,
                max_src_len=args.max_src_len,
                max_new_tokens=args.max_new_tokens,
            )

            all_preds_text.extend(texts)
            parsed_preds.extend([parse_first_valid_token(task, t) for t in texts])
            golds.extend(batch_golds)

        # write raw outputs
        out_path = os.path.join(args.results_dir, f"{task}_preds.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for p in all_preds_text:
                f.write(json.dumps({"prediction": p}, ensure_ascii=False) + "\n")
        print(f"✅ {task} done → {out_path}")

        # scoring
        score = None
        if task in ("race", "medmcqa", "sst2"):
            pairs = [(p, g) for p, g in zip(parsed_preds, golds) if g and p]
            acc = sum(int(p == g) for p, g in pairs) / max(1, len(pairs))
            print(f"  ↳ Accuracy: {acc:.3f}")
            score = acc

        elif task == "squad2":
            ems = [simple_em(p, g) for p, g in zip(parsed_preds, golds) if g and p]
            em = sum(ems) / max(1, len(ems))
            print(f"  ↳ EM: {em:.3f}")
            score = em

        elif task == "iwslt2017-en-fr":
            preds = [p or "" for p in parsed_preds]
            refs = [g or "" for g in golds]
            bleu = simple_bleu(preds, refs)
            if bleu:
                print(f"  ↳ BLEU: {bleu:.2f}")
                score = bleu

        if score is not None:
            summary[task] = score

    # final table
    if summary:
        avg = sum(summary.values()) / len(summary)
        print("\n=== SUMMARY ===")
        for k, v in summary.items():
            print(f"{k}: {v:.3f}")
        print(f"avg: {avg:.3f}")

        csv = os.path.join(args.results_dir, "summary_table.csv")
        with open(csv, "w") as f:
            f.write("task,score\n")
            for k, v in summary.items():
                f.write(f"{k},{v:.6f}\n")
            f.write(f"avg,{avg:.6f}\n")
        print(f"[saved] {csv}")


if __name__ == "__main__":
    main()

'''