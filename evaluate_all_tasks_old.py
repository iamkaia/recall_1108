#!/usr/bin/env python3
# evaluate_all_tasks.py — RECALL benchmark eval with (input + prediction output)
import unsloth
import os
import re
import json
import argparse
from typing import List, Dict
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from sacrebleu import corpus_bleu
import string

# -------------------- args --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./datasets")
    ap.add_argument("--results_dir", type=str, default="./results_eval")
    ap.add_argument("--max_examples", type=int, default=0)  # 0 = full test set
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_src_len", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=64)

    ap.add_argument(
        "--tasks",
        nargs="*",
        default=["sst2", "squad2", "iwslt2017", "race", "medmcqa"],
    )
    return ap.parse_args()


# -------------------- utils --------------------
def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def collate_fn(batch):
    return batch


def parse_first_valid_token(task: str, text: str):
    t = text.strip()

    
    if task == "sst2":
        # 找到「最後一個 Output:」之後的內容，避免被上面的 OPTIONS 影響
        matches = list(re.finditer(r"Output\s*[:：]", t, re.IGNORECASE))
        if not matches:
            return None
        start = matches[-1].end()
        after = t[start:].strip()

        # 取 Output: 之後第一個非空行
        first_line = ""
        for line in after.splitlines():
            line = line.strip()
            if line:
                first_line = line.lower()
                break

        if not first_line:
            return None

        norm = "".join(ch for ch in first_line if ch.isalpha())

        if norm.startswith("positive"):
            return "positive"
        if norm.startswith("negative"):
            return "negative"

        # fallback：只在 Output 後的文字中找一次
        m2 = re.search(r"(positive|negative)", after.lower())
        if m2:
            return m2.group(1)

        return None

        '''
        # 抓 Answer: positive / Answer: negative
        # m = re.search(r"answer\s*[:：]?\s*(positive|negative)", t, re.IGNORECASE)
        # if m:
        #    return m.group(1).lower()
        
        # fallback：抓第一個 emotion（模型可能亂講解釋）
        # m = re.search(r"(positive|negative)", t.lower())
        # return m.group(1) if m else None
        
        # 1) 找 Output: 後的全部內容
        m = re.search(r"Output\s*[:：]?\s*(.*)", t, re.IGNORECASE | re.DOTALL)
        if m:
            after = m.group(1).strip()

            # 2) 取 Output 後第一行非空字串
            first_line = next((line.strip() for line in after.splitlines() if line.strip()), "")
            
            # 3) 看第一行是否包含 positive/negative
            m2 = re.search(r"(positive|negative)", first_line.lower())
            if m2:
                return m2.group(1)

        # fallback - 找全文第一個 positive/negative
        m3 = re.search(r"(positive|negative)", t.lower())
        return m3.group(1) if m3 else None
        '''
    
    elif task in ("race", "medmcqa"):
        '''
        # 抓 Answer: A/B/C/D（不分大小寫）
        m = re.search(r"[Aa]nswer\s*[:：]?\s*([A-Da-d])", t)
        if not m:
            m = re.search(r"\b([A-Da-d])\b", t)
        if m:
            return m.group(1).upper()        # <<< CHANGED：回傳字母 A/B/C/D
        return None
        '''

        # 1. 找最後一個 "Answer:" 或 "Output:"
        last_idx = -1
        for pat in (r"[Aa]nswer\s*[:：]", r"[Oo]utput\s*[:：]"):
            for m in re.finditer(pat, t):
                last_idx = max(last_idx, m.end())
        sub = t[last_idx:] if last_idx != -1 else t

        # 2. 取這一段的第一個非空行
        first_line = ""
        for line in sub.splitlines():
            line = line.strip()
            if line:
                first_line = line
                break

        if not first_line:
            return None

        # 3. 在這行的開頭找 A/B/C/D（忽略前面雜訊，如 "Answer: C:"、"C)" 等）
        m = re.match(r"^[^A-Da-d]*([A-Da-d])\b", first_line)
        if not m:
            # fallback：只在 sub 裡再掃一次
            m = re.search(r"\b([A-Da-d])\b", sub)
        if not m:
            return None

        return m.group(1).upper()


    elif task == "squad2":
        # 抓 "Answer: xxx"，抓不到就用第一行
        m = re.search(r"[Aa]nswer\s*[:：]\s*(.*)", t)
        ans = (m.group(1) if m else t.split("\n", 1)[0]).strip()

        # 將各種「無答案」說法歸一成空字串
        if re.search(r"\b(no\s*answer|unanswerable|cannot\s*be\s*determined|unknown)\b", ans, re.I):
            return ""
        # 清掉句點、逗號等多餘尾巴
        ans = re.sub(r"[.,]+$", "", ans.strip().lower())
        return ans

    elif task == "iwslt2017":
        # 先找 "Answer:" 後面的翻譯
        m = re.search(r"[Aa]nswer\s*[:：]\s*(.*)", t)
        if m:
            ans = m.group(1).strip()
            # 若後面多生成幾句，取第一行
            ans = ans.split("\n")[0].strip()
            return ans
        # fallback：找最後一段非空行
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        return lines[-1] if lines else ""

    return None


def get_target(task: str, ex: Dict):
    tgt = ex.get("target", None)
    if tgt is None:
        return None

    if task == "medmcqa":
        # medmcqa 的 gold 在資料裡通常是 'A'/'B'/'C'/'D'
        # 若未標註（-1）就略過
        tgt = ex.get("target", None)
        if tgt is None or str(tgt).strip() == "-1":
            return None
        g = str(tgt).strip().upper()        # <<< CHANGED：統一成大寫字母
        return g                            # <<< CHANGED：直接用 'A'/'B'/'C'/'D'

    if task == "race":
        g = str(tgt).strip().upper()
        if g == "-1":
            return None
        # RACE 也統一用字母（不要轉 index）
        return g                            # <<< CHANGED：確保回傳 A/B/C/D

    if task == "sst2":
        return str(tgt).strip().lower()

    if task == "squad2":
        g = str(tgt).strip()
        if g == "" or g.lower() in {"no answer", "unanswerable", "none", "n/a"}:
            return ""   # 用空字串代表 no-answer
        return g.lower()

    return str(tgt).strip()


def normalize_answer(s):
    """官方 SQuAD normalization."""
    # 1) 先把常見前綴砍掉
    for prefix in ["answer:", "答案：", "解答："]:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def simple_em(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))

'''
def simple_bleu(preds, refs):
    try:
        import sacrebleu
    except Exception:
        return None  # 沒裝就回 None

    # 清掉空白，避免全空造成錯誤
    preds = [(p or "").strip() for p in preds]
    refs  = [(r or "").strip() for r in refs]
    # 若全部都空，沒意義，回 None
    if not preds or not refs or all(x == "" for x in preds) or all(x == "" for x in refs):
        return None
    # sacrebleu 要求 refs 是 list-of-lists
    return sacrebleu.corpus_bleu(preds, [refs]).score


def normalize_ws(s):
    return " ".join((s or "").strip().split())

def normalize_squad(s):
    """Official SQuAD normalization."""
    for prefix in ["answer:", "答案：", "解答："]:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))
'''

# ======= IWSLT2017 專用：接近論文的 EM =======
def normalize_iwslt(s):
    # do NOT remove punctuation or articles (not valid for FR)
    s = (s or "").strip().lower()
    return " ".join(s.split())

def simple_em_iwslt(pred: str, gold: str) -> int:
    """IWSLT2017 的 EM：經過 normalize_iwslt 後是否完全相等"""
    return int(normalize_iwslt(pred) == normalize_iwslt(gold))

# ======= 各任務的 score_* function：都回傳 [0,1] =======
def score_sst2(preds, golds):
    valid_pairs = [(p, g) for p, g in zip(preds, golds)
                   if p is not None and g is not None]
    total = len(valid_pairs)
    correct = 0
    for p, g in valid_pairs:
        pp = (p or "").strip().lower()
        gg = (g or "").strip().lower()
        if pp == gg:
            correct += 1
    return correct / max(1, total)


def score_race(preds, golds):
    valid_pairs = [(p, g) for p, g in zip(preds, golds)
                   if p is not None and g is not None]
    total = len(valid_pairs)
    correct = 0
    for p, g in valid_pairs:
        pp = (p or "").strip().upper()
        gg = (g or "").strip().upper()
        if pp == gg:
            correct += 1
    return correct / max(1, total)


def score_medmcqa(preds, golds):
    valid_pairs = [(p, g) for p, g in zip(preds, golds)
                   if p is not None and g is not None]
    total = len(valid_pairs)
    correct = 0
    for p, g in valid_pairs:
        pp = (p or "").strip().upper()
        gg = (g or "").strip().upper()
        if pp == gg:
            correct += 1
    return correct / max(1, total)


def score_squad(preds, golds):
    valid_pairs = [(p, g) for p, g in zip(preds, golds)
                   if p is not None and g is not None and g != ""]
    ems = [simple_em(p, g) for p, g in valid_pairs]
    return sum(ems) / max(1, len(ems))


def score_iwslt(preds, golds):
    valid_pairs = [(p, g) for p, g in zip(preds, golds)
                   if p is not None and g is not None]
    ems = [simple_em_iwslt(p, g) for p, g in valid_pairs]
    return sum(ems) / max(1, len(ems))




# -------------------- evaluation --------------------
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("🔄 Loading tokenizer & model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_src_len,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # ✅ 關掉 chat 模式，讓它當純 decoder LM 用
    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = None


    model.eval()

    summary = {}

    print(args.tasks)
    print("This is what we need!!!!!")
    for task in args.tasks:
        if task == "medmcqa" or task == "sst2" or task == "squad2":
            test_path = os.path.join(args.data_root, task, "validation.jsonl")
            print("chose validation!!")
        else:
            test_path = os.path.join(args.data_root, task, "test.jsonl")
            print("chose test!!")

        if not os.path.exists(test_path):
            print(f"[WARN] {test_path} not found, skip.")
            continue

        data = load_jsonl(test_path)
        if args.max_examples > 0:
            data = data[: args.max_examples]

        print(f"\n=== Evaluating {task} ({len(data)} samples) ===")

        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        results = []

        preds, golds = [], []

        for batch in tqdm(dataloader, desc=f"{task} generation"):
            #batch_inputs = [ex["input"] for ex in batch]
            batch_inputs = [
                f"Instruction :{ex['input']}\nOutput:"
                for ex in batch
            ]
            batch_golds = [get_target(task, ex) for ex in batch]

            src_len = args.max_src_len
            if task in ("sst2", "race", "medmcqa"):
                max_new_tokens= 8
            else:
                max_new_tokens= 64

            '''
            if task == "race":
                src_len = max(src_len, 1024)   # 只對 RACE 放大到 1024
            '''

            enc = tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=src_len,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.eos_token_id,
                    "return_dict_in_generate": False,
                }
                
                '''
                gen_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.2,        # 更低、更接近 greedy，但不會亂碼
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                }
                '''
                '''
                gen_kwargs = {
                    "max_new_tokens":args.max_new_tokens,
                    "do_sample":True,                  # <<< 改 True
                    "temperature":0.7,                 # <<< 避免亂碼 token
                    "top_p":0.9,
                    "repetition_penalty":1.2,
                }
                '''
                '''
                # 翻譯任務用 beam search，其餘 greedy 即可
                if task == "iwslt2017":
                    gen_kwargs.update({
                        "num_beams": 2,
                        "length_penalty": 1.0,
                        "early_stopping": True,
                        "no_repeat_ngram_size": 3,
                    })
                '''

                out = model.generate(**enc, **gen_kwargs)

                # === Beam search flatten fix ===
                if isinstance(out, torch.Tensor) and out.dim() == 2:
                    num_gen = out.size(0)
                    bs = len(batch_inputs)
                    if num_gen > bs:
                        num_beams = num_gen // bs
                        out = out[::num_beams]
                        print(f"[INFO] beam={num_beams}, taking best beam per sample.")

                decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

            parsed = [parse_first_valid_token(task, text) for text in decoded]

            # ===== DEBUG: MCQA / SST2 逐條顯示 parsed vs gold =====
            if task in ("race", "medmcqa", "sst2"):
                for pp, g in zip(parsed, batch_golds):
                    print(f"[DEBUG {task}] parsed={pp}, gold={g}, correct={pp == g}")
            # =======================================================

            # write input + prediction + parsed
            for inp, raw, pp, g in zip(batch_inputs, decoded, parsed, batch_golds):
                results.append({
                    "input": inp,
                    "prediction": raw,
                    "parsed": pp,
                    "gold": g
                })

            preds.extend(parsed)
            golds.extend(batch_golds)

        # write predictions
        out_path = os.path.join(args.results_dir, f"{task}_preds.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ Saved predictions → {out_path}")
        
        '''
        # compute score
        score_val = None

        if task in ("sst2", "race", "medmcqa"):
            # 只保留 pred/gold 都非 None 的 pair
            valid_pairs = [(p, g) for p, g in zip(preds, golds) if p is not None and g is not None]
            correct = sum(int(p == g) for p, g in valid_pairs)
            total = len(valid_pairs)
            acc = correct / max(1, total)
            print(f"   ↳ Accuracy: {acc:.3f}")
            score_val = acc

        elif task == "squad2":
            valid_pairs = [(p, g) for p, g in zip(preds, golds)
                           if p is not None and g is not None and g != ""]

            for i, (p, g) in enumerate(valid_pairs[:5]):
                print(f"[DEBUG squad2] EM={int(simple_em(p, g))}, pred={p}, gold={g}")

            ems = [simple_em(p, g) for p, g in valid_pairs]
            em = sum(ems) / max(1, len(ems))
            print(f"   ↳ EM: {em:.3f}")
            score_val = em

        elif task == "iwslt2017":
            preds_ = [p or "" for p in preds]
            refs_  = [g or "" for g in golds]
            bleu = simple_bleu(preds_, refs_)
            if isinstance(bleu, (int, float)):
                print(f"   ↳ BLEU: {bleu:.2f}")
                score_val = bleu
            else:
                print("   ↳ BLEU: skipped (install `sacrebleu` or check empty refs/preds)")

        summary[task] = score_val
        '''
        # compute score
        score_val = None

        if task == "sst2":
            acc = score_sst2(preds, golds)
            print(f"   ↳ Accuracy (sst2): {acc:.3f}")
            score_val = acc

        elif task == "race":
            acc = score_race(preds, golds)
            print(f"   ↳ Accuracy (race): {acc:.3f}")
            score_val = acc

        elif task == "medmcqa":
            acc = score_medmcqa(preds, golds)
            print(f"   ↳ Accuracy (medmcqa): {acc:.3f}")
            score_val = acc

        elif task == "squad2":
            valid_pairs = [(p, g) for p, g in zip(preds, golds)
                           if p is not None and g is not None and g != ""]

            # debug 看前幾筆 EM
            for i, (p, g) in enumerate(valid_pairs[:5]):
                print(f"[DEBUG squad2] EM={int(simple_em(p, g))}, pred={p}, gold={g}")

            em = score_squad(preds, golds)
            print(f"   ↳ EM (squad2): {em:.3f}")
            score_val = em

        
        elif task == "iwslt2017":
            '''
            valid_pairs = [(p, g) for p, g in zip(preds, golds)
                           if p is not None and g is not None]

            # debug 看前幾筆 EM
            for i, (p, g) in enumerate(valid_pairs[:5]):
                print(f"[DEBUG iwslt2017] EM={int(simple_em_iwslt(p, g))}, pred={p}, gold={g}")

            em = score_iwslt(preds, golds)
            #print(f"   ↳ EM (iwslt2017): {em:.3f}")
            #score_val = em
            # ---- Calculate EM ----
            em = score_iwslt(preds, golds)

            # ---- Calculate BLEU ----
            preds_clean = [p.strip() for p, _ in valid_pairs]
            gold_clean = [g.strip() for _, g in valid_pairs]

            bleu = corpus_bleu(preds_clean, [gold_clean]).score  # sacrebleu expects list of references

            print(f"   ↳ EM (iwslt2017): {em:.3f}")
            print(f"   ↳ BLEU (iwslt2017): {bleu:.3f}")

            # 用 BLEU or EM 作 score? If want both:
            score_val = {"em": em, "bleu": bleu}
            '''
            valid_pairs = [(p, g) for p, g in zip(preds, golds)
                        if p is not None and g is not None]

            # Debug 前 5 筆 EM (optional debug only)
            for i, (p, g) in enumerate(valid_pairs[:5]):
                print(f"[DEBUG iwslt2017] EM={int(simple_em_iwslt(p, g))}, pred={p}, gold={g}")

            # ---- Calculate EM (optional debug metric) ----
            em = score_iwslt(preds, golds)

            # ---- Calculate BLEU ----
            preds_clean = [p.strip() for p, _ in valid_pairs]
            gold_clean  = [g.strip() for _, g in valid_pairs]

            bleu = corpus_bleu(preds_clean, [gold_clean]).score  # sacrebleu needs ref list

            print(f"   ↳ EM (iwslt2017): {em:.3f}")
            print(f"   ↳ BLEU (iwslt2017): {bleu:.3f}")

            # ---- Use BLEU as the task score (per RECAll benchmark) ----
            score_val = float(bleu)


        summary[task] = score_val


    print(summary)


    # summary result
    if summary:
        valid_scores = [v for v in summary.values() if v is not None]
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            print(f"avg: {avg:.3f}")

            csv_path = os.path.join(args.results_dir, "summary_table.csv")
            with open(csv_path, "w") as f:
                f.write("task,score\n")
                for k, v in summary.items():
                    if v is not None:
                        f.write(f"{k},{v:.6f}\n")
                f.write(f"avg,{avg:.6f}\n")


if __name__ == "__main__":
    main()