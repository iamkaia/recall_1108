#!/usr/bin/env python3
# merge_prototype_recall.py — Prototype-aware RECALL (cosine − τ·distance)
# Kaia 版本：改良 RECALL，用質心距離調權，提升語義穩定性與多任務泛化

import os, re, json, argparse, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ------------------------------------------------------------
# 工具函式
# ------------------------------------------------------------
def get_layer_id_from_name(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1

def load_jsonl(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def sample_prompts_from_tasks(data_root, tasks, k_per_task):
    prompts = []
    for t in tasks:
        train_p = os.path.join(data_root, t, "train.jsonl")
        test_p  = os.path.join(data_root, t, "test.jsonl")
        data = load_jsonl(train_p) or load_jsonl(test_p)
        for ex in data[:k_per_task]:
            inp = ex.get("input", "").strip()
            if inp: prompts.append(inp + "\nAnswer:")
    if not prompts:
        prompts = [
            "Translate: Hello world! → French\nAnswer:",
            "What is the capital of France?\nAnswer:",
            "Give one advantage of convolutional neural networks.\nAnswer:",
        ]
    return prompts

# ------------------------------------------------------------
# 收集每層 hidden states 的平均表徵 (Eq.4)
# ------------------------------------------------------------
def collect_hidden_states(model, tokenizer, prompts, pad_len):
    hidden_accum = []
    with torch.no_grad():
        for text in prompts:
            enc = tokenizer(
                text, padding="max_length", truncation=True,
                max_length=pad_len, add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)
            out = model(**enc, output_hidden_states=True, return_dict=True)
            h = torch.stack([x[:, -1, :] for x in out.hidden_states]).squeeze(1)
            hidden_accum.append(h)
    return torch.stack(hidden_accum, dim=0).mean(dim=0)   # (L,D)

# ------------------------------------------------------------
# SVD 分解 (LoRA ΔW → A*,B*)
# ------------------------------------------------------------
def svd_factorize_delta(delta, r):
    dev = delta.device
    delta_cpu = delta.to(torch.float32, copy=True).cpu()
    U, S, Vh = torch.linalg.svd(delta_cpu, full_matrices=False)
    r_use = min(r, U.shape[1], Vh.shape[0])
    Ur, Sr, Vhr = U[:, :r_use], S[:r_use], Vh[:r_use, :]
    Sr_sqrt = torch.sqrt(Sr)
    B_star = Ur @ torch.diag(Sr_sqrt)
    A_star = torch.diag(Sr_sqrt) @ Vhr
    if r_use < r:
        out_f, in_f = delta.shape
        B_pad = torch.zeros((out_f, r), dtype=B_star.dtype)
        A_pad = torch.zeros((r, in_f), dtype=A_star.dtype)
        B_pad[:, :r_use] = B_star; A_pad[:r_use, :] = A_star
        B_star, A_star = B_pad, A_pad
    return B_star.to(dev), A_star.to(dev)

# ------------------------------------------------------------
# 主流程：Prototype-aware 合併
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapters_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="fused_prototype_recall")
    parser.add_argument("--tasks", nargs="*", default=["sst2","squad2","iwslt2017-en-fr","race","medmcqa"])
    parser.add_argument("--samples_per_task", type=int, default=20)
    parser.add_argument("--pad_len", type=int, default=128)
    parser.add_argument("--tau", type=float, default=2.0, help="prototype distance scaling")
    args = parser.parse_args()

    print("\n========== [STEP] Prototype-aware RECALL ==========\n")

    adapter_dirs = sorted([
        os.path.join(args.adapters_root, d)
        for d in os.listdir(args.adapters_root)
        if os.path.isdir(os.path.join(args.adapters_root, d))
           and os.path.exists(os.path.join(args.adapters_root, d, "adapter_config.json"))
    ])
    assert adapter_dirs, "❌ No LoRA adapters found."

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="auto"
    )
    base.eval()

    peft_model = PeftModel.from_pretrained(base, adapter_dirs[0], adapter_name="a0")
    names = ["a0"]

    prompts = sample_prompts_from_tasks(args.data_root, args.tasks, args.samples_per_task)
    print(f"🧾 Using {len(prompts)} prompts for prototype mapping...")

    # Anchor 表徵
    anchor_rep = collect_hidden_states(peft_model, tokenizer, prompts, args.pad_len)
    all_reps = [anchor_rep]

    for i, adir in enumerate(adapter_dirs[1:], start=1):
        print(f"📌 Load adapter a{i}: {adir}")
        peft_model.load_adapter(adir, adapter_name=f"a{i}")
        rep = collect_hidden_states(peft_model, tokenizer, prompts, args.pad_len)
        all_reps.append(rep)
        names.append(f"a{i}")

    reps = torch.stack(all_reps, dim=0).to(base.device)   # (N,L,D)
    num_adapters, num_layers, _ = reps.shape

    # ------------------------------------------------------------
    # Prototype-aware 相似度：cosine − τ·L2距離
    # ------------------------------------------------------------
    print("📐 Computing prototype-aware similarity (cosine − τ·distance)...")
    anchor = reps[0]
    cosine = F.cosine_similarity(anchor.unsqueeze(0), reps, dim=-1)  # (N,L)
    dist = torch.norm(reps - anchor.unsqueeze(0), dim=-1)            # (N,L)
    sim = cosine - args.tau * dist
    weights_layerwise = F.softmax(sim, dim=0)
    weights_scalar = weights_layerwise.mean(dim=1)

    # ------------------------------------------------------------
    # Layer-wise 加權 SVD 融合
    # ------------------------------------------------------------
    print("🧠 Performing weighted merge with SVD...")
    peft_model.load_adapter(adapter_dirs[0], adapter_name="proto_recall")

    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if "proto_recall" not in module.lora_A: continue
            layer_id = get_layer_id_from_name(name)
            w = weights_layerwise[:, layer_id] if 0 <= layer_id < num_layers else weights_scalar
            deltas = []
            r_here = None
            for a_name in names:
                A = module.lora_A[a_name].weight
                B = module.lora_B[a_name].weight
                if r_here is None: r_here = A.shape[0]
                deltas.append(B @ A)
            delta_sum = torch.zeros_like(deltas[0])
            for q in range(num_adapters):
                delta_sum.add_(deltas[q] * w[q])
            B_star, A_star = svd_factorize_delta(delta_sum, r_here)
            module.lora_A["proto_recall"].weight.data.copy_(A_star)
            module.lora_B["proto_recall"].weight.data.copy_(B_star)

    peft_model.set_adapter("proto_recall")
    merged = peft_model.merge_and_unload()
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n🎉 Prototype-aware RECALL fused → {args.output_dir}\n")

if __name__ == "__main__":
    main()
