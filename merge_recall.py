#!/usr/bin/env python3
# merge_recall.py — RECALL (layer-wise similarity + softmax merge, with SVD)
# Kaia final

import os
import re
import json
import argparse
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------
# Utils
# ---------------------------
def get_layer_id_from_name(name: str) -> int:
    """
    嘗試從 module 名稱抓出 transformer 層號。
    例如：...model.layers.23.self_attn.q_proj...
    找不到就回 -1（之後會用均值權重做後備）。
    """
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sample_prompts_from_tasks(data_root: str, tasks: List[str], k_per_task: int) -> List[str]:
    prompts = []
    for t in tasks:
        # 優先用 train，沒有就 test
        train_p = os.path.join(data_root, t, "train.jsonl")
        test_p  = os.path.join(data_root, t, "test.jsonl")
        data = load_jsonl(train_p) or load_jsonl(test_p)
        for ex in data[:k_per_task]:
            inp = ex.get("input", "").strip()
            if inp:
                prompts.append(inp + "\nAnswer:")
    # 萬一資料夾是空的，就給幾條保底 prompt
    if not prompts:
        prompts = [
            "State one advantage of convolutional neural networks.\nAnswer:",
            "Translate: Hello world! → French\nAnswer:",
            "Is the sentiment positive or negative? Sentence: I absolutely loved it.\nAnswer:",
            "Read the passage and answer the question.\nAnswer:",
        ]
    return prompts


# ---------------------------
# Representation collection (Eq.4)
# ---------------------------
def collect_hidden_states(model, tokenizer, prompts: List[str], pad_len: int) -> torch.Tensor:
    """
    回傳 shape: (num_layers, hidden_dim)
    做法：對每個 prompt forward（不使用 chat 模板），取每層最後一個 token 的 hidden，
    再平均（Eq.4）。
    """
    states = []
    # 這裡只能用 no_grad（不要用 inference_mode），避免 peft load_adapter 觸發 requires_grad 錯誤
    with torch.no_grad():
        for text in prompts:
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=pad_len,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)

            out = model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
            )
            # list[ (1, seq, dim) ] per layer → 取最後一個 token 的 hidden 拼成 (num_layers, dim)
            layer_vecs = torch.stack([h[:, -1, :] for h in out.hidden_states]).squeeze(1)
            states.append(layer_vecs)

    return torch.stack(states, dim=0).mean(dim=0)  # (layers, dim)


# ---------------------------
# SVD factorization of delta (CPU float32)
# ---------------------------
def svd_factorize_delta(delta: torch.Tensor, r: int):
    """
    delta: (out_features, in_features)  (B @ A)
    目標還原成：
      B* = U_r @ sqrt(S_r)           -> (out, r)
      A* = sqrt(S_r) @ Vh_r          -> (r, in)
    注意：跑在 CPU + float32，避免 Half 的 svd 不支援。
    """
    dev = delta.device
    delta_cpu = delta.to(torch.float32, copy=True).cpu()

    # torch.linalg.svd 比 torch.svd 穩定
    U, S, Vh = torch.linalg.svd(delta_cpu, full_matrices=False)
    r_use = min(r, U.shape[1], Vh.shape[0], S.shape[0])

    Ur = U[:, :r_use]
    Sr = S[:r_use]
    Vhr = Vh[:r_use, :]

    Sr_sqrt = torch.sqrt(Sr)
    # (out, r_use)
    B_star = Ur @ torch.diag(Sr_sqrt)
    # (r_use, in)
    A_star = torch.diag(Sr_sqrt) @ Vhr

    # 若 r_use < r，需要 zero-pad 到 LoRA rank 大小
    if r_use < r:
        out_f, in_f = delta.shape
        B_pad = torch.zeros((out_f, r), dtype=B_star.dtype)
        A_pad = torch.zeros((r, in_f), dtype=A_star.dtype)
        B_pad[:, :r_use] = B_star
        A_pad[:r_use, :] = A_star
        B_star, A_star = B_pad, A_pad

    return B_star.to(dev), A_star.to(dev)  # 注意：A_star shape = (r, in)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",   type=str, required=True)
    parser.add_argument("--adapters_root", type=str, required=True)
    parser.add_argument("--data_root",    type=str, default="datasets")
    parser.add_argument("--output_dir",   type=str, default="fused_recall")
    parser.add_argument("--tasks", nargs="*", default=["sst2", "squad2", "iwslt2017-en-fr", "race", "medmcqa"])
    parser.add_argument("--samples_per_task", type=int, default=20, help="取每個 task 幾條樣本來對齊表示")
    parser.add_argument("--pad_len", type=int, default=128, help="收集 hidden 的固定長度")
    args = parser.parse_args()

    print("\n========== [STEP 2] RECALL merge (layer-wise similarity + softmax + SVD) ==========\n")

    # 1) 掃描 LoRA adapters
    adapter_dirs = sorted([
        os.path.join(args.adapters_root, d)
        for d in os.listdir(args.adapters_root)
        if os.path.isdir(os.path.join(args.adapters_root, d))
           and os.path.exists(os.path.join(args.adapters_root, d, "adapter_config.json"))
    ])
    assert adapter_dirs, "❌ No LoRA adapters found in --adapters_root."

    print(f"🔍 Found {len(adapter_dirs)} adapters:")
    for ad in adapter_dirs:
        print(f"   • {ad}")

    # 2) tokenizer（關閉 chat 模式），使用 base_model 的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # decoder-only 建議左 pad
    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = None
    if hasattr(tokenizer, "apply_chat_template"):
        tokenizer.apply_chat_template = lambda messages, **kwargs: "".join([m["content"] for m in messages])

    # 3) base model（不載 LoRA，稍後動態套）
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base.eval()

    # 4) 先把第一個 adapter 當 anchor (a0)
    print(f"\n📌 Load anchor adapter as 'a0': {adapter_dirs[0]}")
    peft_model = PeftModel.from_pretrained(base, adapter_dirs[0], adapter_name="a0")
    names = ["a0"]

    # 5) 蒐集 prompts（每個 task 取 K 條）
    prompts = sample_prompts_from_tasks(args.data_root, args.tasks, args.samples_per_task)
    print(f"🧾 Using {len(prompts)} prompts for representation alignment (Eq.4)")

    # 6) Anchor 表示
    anchor_rep = collect_hidden_states(peft_model, tokenizer, prompts, args.pad_len)  # (L, D)

    # 7) 載入其他 adapters，逐一收集表示
    all_reps = [anchor_rep]  # list of (L, D)
    for i, adir in enumerate(adapter_dirs[1:], start=1):
        print(f"📌 Load adapter a{i}: {adir}")
        # 注意：load_adapter 過程不能在 inference_mode 內，否則會報 requires_grad 的錯
        peft_model.load_adapter(adir, adapter_name=f"a{i}")
        names.append(f"a{i}")
        rep = collect_hidden_states(peft_model, tokenizer, prompts, args.pad_len)
        all_reps.append(rep)

    reps = torch.stack(all_reps, dim=0)  # (N, L, D)
    num_adapters, num_layers, _ = reps.shape
    print(f"✅ reps shape = {tuple(reps.shape)} (adapters, layers, hidden_dim)")

    # 8) 逐層 similarity（以 anchor 對其他 adapter），softmax 成權重（Eq.4/5/6）
    print("\n📐 Computing layer-wise cosine similarity → softmax weights...")
    anchor = reps[0]                          # (L, D)
    sim = F.cosine_similarity(anchor.unsqueeze(0), reps, dim=-1)  # (N, L)
    weights_layerwise = F.softmax(sim, dim=0)                     # (N, L)
    # 另外準備一個「adapter 的 scalar 權重」作後備（找不到層號時用）
    weights_scalar = weights_layerwise.mean(dim=1)                # (N,)

    # 9) 先建立一個「recall」adapter 結構，等會覆寫其 A/B 權重
    peft_model.load_adapter(adapter_dirs[0], adapter_name="recall")

    # 10) 針對每一個 LoRA module，做「層對應 → 權重加權 → SVD 回填」
    print("\n🧠 Performing layer-wise ΔW merge with SVD (per LoRA module)...")
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if "recall" not in module.lora_A or "recall" not in module.lora_B:
                continue

            # 嘗試對應到 transformer 層，用該層的權重；否則用 scalar 權重
            layer_id = get_layer_id_from_name(name)
            if 0 <= layer_id < num_layers:
                w = weights_layerwise[:, layer_id]  # (N,)
            else:
                w = weights_scalar                   # (N,)

            # 所有 adapter 的 delta 疊加
            deltas = []
            r_here = None
            for a_name in names:
                A = module.lora_A[a_name].weight    # (r, in)
                B = module.lora_B[a_name].weight    # (out, r)
                if r_here is None:
                    r_here = A.shape[0]
                deltas.append(B @ A)                 # (out, in)
            # 加權
            delta_sum = torch.zeros_like(deltas[0])
            for q in range(num_adapters):
                delta_sum.add_(deltas[q] * w[q])

            # SVD 分解 → 得到新的 B*, A* （注意 A* shape 是 (r,in)）
            B_star, A_star = svd_factorize_delta(delta_sum, r_here)

            # 回填到「recall」adapter
            module.lora_A["recall"].weight.data.copy_(A_star)  # (r, in)
            module.lora_B["recall"].weight.data.copy_(B_star)  # (out, r)

    # 11) 設定 active adapter → merge 到 base 權重並儲存完整模型（含 config.json / model.safetensors）
    peft_model.set_adapter("recall")
    merged = peft_model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n🎉 RECALL model fused → {args.output_dir}\n")


if __name__ == "__main__":
    main()

'''
# merge_recall.py  — RECALL (layer-wise, similarity-guided) merging
import os, re, glob, json, math, argparse, random
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -------------- utils: name parsing -----------------
_LAYER_PAT = re.compile(r"(layers|h)\.(\d+)\.")  # Qwen/LLaMA: model.layers.{i}.xxx

def extract_layer_idx(name: str) -> int:
    m = _LAYER_PAT.search(name)
    return int(m.group(2)) if m else -1

def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------- dataset probing ---------------------
def load_probe_inputs(datasets_root: str, num_probe: int) -> List[str]:
    """
    掃描 datasets_root/*/train.jsonl 或 validation.jsonl，擷取 'input' 欄位
    """
    files = []
    for sub in os.listdir(datasets_root):
        d = os.path.join(datasets_root, sub)
        if not os.path.isdir(d): 
            continue
        for fname in ["train.jsonl", "validation.jsonl", "val.jsonl"]:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                files.append(p)
    pool = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if "input" in j and isinstance(j["input"], str):
                        pool.append(j["input"])
                except:
                    pass
    random.shuffle(pool)
    pool = pool[:num_probe] if num_probe > 0 else pool
    if not pool:
        raise RuntimeError(f"No probe inputs found under {datasets_root}. Put jsonl with 'input' field.")
    # 統一在末尾加 "Answer:"（與你的 SFT prompt 對齊）
    return [x.rstrip() + "\nAnswer:" for x in pool]

# -------------- hidden states collection -------------
@torch.no_grad()
def collect_layer_reps(
    peft_model: PeftModel,
    adapter_name: str,
    tokenizer,
    probe_inputs: List[str],
    device: str = "cuda",
    max_len: int = 512,
    batch_size: int = 8,
) -> Dict[int, torch.Tensor]:
    """
    啟用某個 adapter，對 probe_inputs 做前向，收集每層 hidden_states 的 mean-pooled 表徵
    回傳： {layer_idx -> vector (hidden_dim,)}
    """
    peft_model.set_adapter(adapter_name)
    peft_model.eval()

    reps_sum: Dict[int, torch.Tensor] = {}
    reps_cnt: Dict[int, int] = {}

    for i in range(0, len(probe_inputs), batch_size):
        batch = probe_inputs[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)
        out = peft_model.base_model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # out.hidden_states: (layers+1) list of [B, T, H]（含 embedding 0 層）
        hs = out.hidden_states  # List[Tensor]
        for layer_idx, h in enumerate(hs[1:], start=0):  # 去掉 embedding 層，從 0 對應第一個 block 前輸出
            # mean over tokens then mean over batch → [H]
            # 只對非 padding 的位置做平均
            mask = enc.attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            masked = h * mask
            denom = mask.sum(dim=1).clamp_min(1e-6)  # [B, 1]
            pooled = (masked.sum(dim=1) / denom)  # [B, H]
            vec = pooled.mean(dim=0)  # [H]
            if layer_idx not in reps_sum:
                reps_sum[layer_idx] = vec.clone()
                reps_cnt[layer_idx] = 1
            else:
                reps_sum[layer_idx] += vec
                reps_cnt[layer_idx] += 1

    reps_mean = {i: (reps_sum[i] / reps_cnt[i]) for i in reps_sum}
    return reps_mean  # {layer_idx: [H]}

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float(), dim=0)
    b = F.normalize(b.float(), dim=0)
    return float((a * b).sum().item())

def softmax(x: torch.Tensor, tau: float = 10.0) -> torch.Tensor:
    z = (x * tau) - (x.max())
    exp = torch.exp(z)
    return exp / exp.sum().clamp_min(1e-8)

# -------------- layer-wise weight computation --------
def compute_layerwise_weights(reps_per_adapter: Dict[str, Dict[int, torch.Tensor]],
                              temperature: float = 10.0) -> Dict[int, Dict[str, float]]:
    """
    給定：每個 adapter 的各層表徵
      reps_per_adapter: {adapter -> {layer_idx -> vec}}
    產出：逐層 softmax 權重
      weights[layer_idx][adapter] = w
    做法：以「各適配器表徵的均值（centroid）」為參考，算 cosine，相似度越高，權重越大。
    """
    adapters = sorted(reps_per_adapter.keys())
    # 統一層集合
    all_layers = sorted(set().union(*[set(reps_per_adapter[a].keys()) for a in adapters]))
    weights: Dict[int, Dict[str, float]] = {}

    for li in all_layers:
        vecs = [reps_per_adapter[a][li] for a in adapters if li in reps_per_adapter[a]]
        if len(vecs) != len(adapters):
            # 某些 adapter 少層（理論上不會），用 0 權重
            sims = torch.zeros(len(adapters))
            w = softmax(sims, tau=temperature)
            weights[li] = {a: float(w[j].item()) for j, a in enumerate(adapters)}
            continue

        centroid = torch.stack(vecs, dim=0).mean(dim=0)
        sims = torch.tensor([cosine_sim(reps_per_adapter[a][li], centroid) for a in adapters])
        w = softmax(sims, tau=temperature)
        weights[li] = {a: float(w[j].item()) for j, a in enumerate(adapters)}

    return weights  # {layer_idx: {adapter: w}}

# -------------- LoRA delta extraction & merging ------
def _get_lora_delta(module, adapter_name: str) -> torch.Tensor:
    """
    回傳該 module 在 adapter 下的 LoRA delta:  B @ A * scaling
    只支援 Linear 類型的 LoRA（q/k/v/o 等）
    """
    # peft 的 LoRA 層會掛這些 dict 屬性
    A = module.lora_A[adapter_name].weight   # [r, in]
    B = module.lora_B[adapter_name].weight   # [out, r]
    r = A.shape[0]
    alpha = module.lora_alpha[adapter_name]
    scaling = alpha / r
    delta = torch.matmul(B, A) * scaling     # [out, in]
    return delta

@torch.no_grad()
def merge_layerwise(
    peft_model: PeftModel,
    layerwise_w: Dict[int, Dict[str, float]],
    out_dir: str,
):
    """
    依據每層權重，把各 adapter 的 LoRA delta 加權後加進底層 Linear 權重
    """
    base = peft_model.base_model
    device = next(base.parameters()).device
    dtype = next(base.parameters()).dtype

    # 遍歷所有具 LoRA 的 module
    for name, mod in peft_model.named_modules():
        if not hasattr(mod, "lora_A"):
            continue  # 非 LoRA 化模組
        layer_idx = extract_layer_idx(name)
        if layer_idx < 0:
            # 如果抓不到層號：保守用平均權重
            adapters = list(mod.lora_A.keys())
            avg_w = 1.0 / max(1, len(adapters))
            w_map = {a: avg_w for a in adapters}
        else:
            # 該層的權重表
            w_map = layerwise_w.get(layer_idx, None)
            if w_map is None:
                adapters = list(mod.lora_A.keys())
                avg_w = 1.0 / max(1, len(adapters))
                w_map = {a: avg_w for a in adapters}

        # 逐 adapter 取 delta，加權相加
        delta_sum = None
        for a_name in mod.lora_A.keys():
            if a_name not in w_map:
                continue
            w = w_map[a_name]
            d = _get_lora_delta(mod, a_name).to(device=device, dtype=dtype)
            delta_sum = d * w if delta_sum is None else delta_sum + d * w

        # 把加權後的 delta 合進底層 Linear 權重
        if delta_sum is not None:
            # Linear 權重名稱通常在 mod.base_layer.weight
            if hasattr(mod, "weight") and mod.weight is not None:
                # 某些 PEFT 版本直接把 Linear 包成 LoRALinear，權重就是 mod.weight
                mod.weight += delta_sum
            elif hasattr(mod, "base_layer") and hasattr(mod.base_layer, "weight"):
                mod.base_layer.weight += delta_sum
            else:
                # 無法找到底層權重（理論上不會）
                pass

    # 移除 LoRA 結構並保存純權重
    # 最穩的作法：直接把 peft 的 lora 結構保留但不再需要；我們把底層已經加好 delta
    # 用 base_model.save_pretrained 輸出純底座
    os.makedirs(out_dir, exist_ok=True)
    base.save_pretrained(out_dir)
    print(f"💾 Saved merged (pure) model to: {out_dir}")

# -------------- main --------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapters_root", default="checkpoints_full")
    ap.add_argument("--adapters", nargs="*", default=None)
    ap.add_argument("--datasets_root", default="./datasets")
    ap.add_argument("--num_probe", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--tokenizer", default=None)  # 若不給，沿用 base 的 tokenizer
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16","float32"])
    ap.add_argument("--out_dir", default="fused_recall_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # 1) 收集 adapters
    if args.adapters is None:
        cands = sorted([p for p in glob.glob(os.path.join(args.adapters_root, "*")) if os.path.isdir(p)])
    else:
        cands = args.adapters
    adapters = []
    for p in cands:
        if os.path.exists(os.path.join(p, "adapter_config.json")):
            adapters.append(p)
    if len(adapters) < 2:
        raise RuntimeError("Need at least two adapters for merging.")

    print("✅ Adapters:")
    for i, p in enumerate(adapters):
        print(f"  [{i}] {p}")

    # 2) 載 base + 第一個 LoRA 當 anchor，後續載進 peft
    print(f"🔄 Loading base: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype, device_map="auto")
    peft_model = PeftModel.from_pretrained(base, adapters[0])
    peft_model.set_adapter("a0")

    # 後續 adapters
    for idx, adir in enumerate(adapters[1:], start=1):
        name = f"a{idx}"
        peft_model.load_adapter(adir, adapter_name=name)

    # 3) 準備 tokenizer 與 probe inputs
    tok_id = args.tokenizer if args.tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    probe_inputs = load_probe_inputs(args.datasets_root, args.num_probe)
    print(f"🧪 Loaded {len(probe_inputs)} probe prompts from {args.datasets_root}")

    # 4) 逐 adapter 收集各層表徵
    reps_per_adapter: Dict[str, Dict[int, torch.Tensor]] = {}
    for idx in range(len(adapters)):
        aname = f"a{idx}"
        reps = collect_layer_reps(peft_model, aname, tokenizer, probe_inputs)
        reps_per_adapter[aname] = reps
        print(f"   · collected reps for {aname} ({len(reps)} layers)")

    # 5) 逐層計算 softmax 權重
    layer_w = compute_layerwise_weights(reps_per_adapter, temperature=args.temperature)
    # 簡要印出幾層權重
    some_layers = sorted(layer_w.keys())[:5] + sorted(layer_w.keys())[-5:]
    print("📊 Sample layer weights:")
    for li in some_layers:
        wm = layer_w[li]
        pretty = ", ".join([f"{k}:{wm[k]:.2f}" for k in sorted(wm.keys())])
        print(f"  layer {li}: {pretty}")

    # 6) 逐層把 LoRA delta 加權合併到底層
    merge_layerwise(peft_model, layer_w, args.out_dir)

if __name__ == "__main__":
    main()
'''