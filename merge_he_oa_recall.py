#!/usr/bin/env python3
# merge_he_oa_adapter_recall.py — RECALL + High-Entropy weighting + OA-Adapter projection
# Kaia version (4090 single-GPU friendly)

import os, re, json, argparse, torch
import torch.nn.functional as F
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ------------------------------------------------------------
# 基本工具
# ------------------------------------------------------------
def get_layer_id_from_name(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def sample_prompts_from_tasks(data_root: str, tasks: List[str], k_per_task: int) -> List[str]:
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
            "What is the capital of France?\nAnswer:",
            "Translate: Hello world! → French\nAnswer:",
            "Explain one advantage of neural networks.\nAnswer:",
        ]
    return prompts

# ------------------------------------------------------------
# 表徵蒐集 (Eq.4)
# ------------------------------------------------------------
def collect_hidden_states_and_entropy(model, tokenizer, prompts: List[str], pad_len: int):
    """
    回傳 hidden_states: (L, D) 以及 entropy: (len(prompts),)
    熵是根據最後 token 的 logits 經 softmax 後計算 Shannon entropy。
    """
    hidden_accum = []
    entropy_all = []
    with torch.no_grad():
        for text in prompts:
            enc = tokenizer(text, padding="max_length", truncation=True,
                            max_length=pad_len, add_special_tokens=False,
                            return_tensors="pt").to(model.device)
            out = model(**enc, output_hidden_states=True, return_dict=True)
            h = torch.stack([x[:, -1, :] for x in out.hidden_states]).squeeze(1)  # (L,D)
            hidden_accum.append(h)
            # --- 熵計算 ---
            logits = out.logits[:, -1, :]  # (1,vocab)
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * probs.log()).sum(dim=-1)  # (1,)
            entropy_all.append(ent.item())

    entropy_tensor = torch.tensor(entropy_all, dtype=torch.float32)
    hidden_mean = torch.stack(hidden_accum, dim=0).mean(dim=0)  # (L,D)
    return hidden_mean, entropy_tensor.mean()  # 取平均熵即可用於加權

# ------------------------------------------------------------
# SVD 分解
# ------------------------------------------------------------
def svd_factorize_delta(delta: torch.Tensor, r: int):
    dev = delta.device
    delta_cpu = delta.to(torch.float32, copy=True).cpu()
    U, S, Vh = torch.linalg.svd(delta_cpu, full_matrices=False)
    r_use = min(r, U.shape[1], Vh.shape[0])
    Ur, Sr, Vhr = U[:, :r_use], S[:r_use], Vh[:r_use, :]
    Sr_sqrt = torch.sqrt(Sr)
    B_star = Ur @ torch.diag(Sr_sqrt)
    A_star = torch.diag(Sr_sqrt) @ Vhr
    # zero-pad
    if r_use < r:
        out_f, in_f = delta.shape
        B_pad = torch.zeros((out_f, r), dtype=B_star.dtype)
        A_pad = torch.zeros((r, in_f), dtype=A_star.dtype)
        B_pad[:, :r_use] = B_star; A_pad[:r_use, :] = A_star
        B_star, A_star = B_pad, A_pad
    return B_star.to(dev), A_star.to(dev)

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapters_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="fused_he_oa_recall")
    parser.add_argument("--tasks", nargs="*", default=["sst2","squad2","iwslt2017-en-fr","race","medmcqa"])
    parser.add_argument("--samples_per_task", type=int, default=20)
    parser.add_argument("--pad_len", type=int, default=128)
    args = parser.parse_args()

    print("\n========== [STEP] HE+OA RECALL merge ==========\n")

    adapter_dirs = sorted([
        os.path.join(args.adapters_root, d)
        for d in os.listdir(args.adapters_root)
        if os.path.isdir(os.path.join(args.adapters_root, d))
           and os.path.exists(os.path.join(args.adapters_root, d, "adapter_config.json"))
    ])
    assert adapter_dirs, "❌ No LoRA adapters found."

    print(f"🔍 Found {len(adapter_dirs)} adapters.")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="auto"
    )
    base.eval()

    peft_model = PeftModel.from_pretrained(base, adapter_dirs[0], adapter_name="a0")
    names = ["a0"]

    prompts = sample_prompts_from_tasks(args.data_root, args.tasks, args.samples_per_task)
    print(f"🧾 Using {len(prompts)} prompts for representation alignment (HE-RECALL).")

    # --- Anchor ---
    anchor_rep, anchor_entropy = collect_hidden_states_and_entropy(peft_model, tokenizer, prompts, args.pad_len)
    all_reps = [anchor_rep]; entropies = [anchor_entropy]

    for i, adir in enumerate(adapter_dirs[1:], start=1):
        print(f"📌 Load adapter a{i}: {adir}")
        peft_model.load_adapter(adir, adapter_name=f"a{i}")
        rep, ent = collect_hidden_states_and_entropy(peft_model, tokenizer, prompts, args.pad_len)
        all_reps.append(rep); entropies.append(ent)
        names.append(f"a{i}")

    reps = torch.stack(all_reps, dim=0)  # (N,L,D)
    entropies = torch.tensor(entropies, device=base.device)  # <-- 新增 device=base.device
    num_adapters, num_layers, _ = reps.shape

    print("📐 Computing entropy-weighted cosine similarity...")
    anchor = reps[0]
    sim = F.cosine_similarity(anchor.unsqueeze(0), reps, dim=-1)  # (N,L)
    # normalize entropy
    e_norm = entropies / entropies.mean()
    sim = sim * e_norm.unsqueeze(1)


    # ------------------------------------------------------------
    # Entropy-weighted similarity
    # ------------------------------------------------------------
    print("📐 Computing entropy-weighted cosine similarity...")
    anchor = reps[0]
    sim = F.cosine_similarity(anchor.unsqueeze(0), reps, dim=-1)  # (N,L)
    # normalize entropy
    e_norm = entropies / entropies.mean()
    sim = sim * e_norm.unsqueeze(1)
    weights_layerwise = F.softmax(sim, dim=0)
    weights_scalar = weights_layerwise.mean(dim=1)

    # ------------------------------------------------------------
    # Orthogonal projection (OA) - 基於 anchor 子空間去投影新任務方向
    # ------------------------------------------------------------
    print("🧮 Applying orthogonal projection on LoRA A matrices...")
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            for a_name in names[1:]:  # 跳過 anchor
                A = module.lora_A[a_name].weight.data
                A0 = module.lora_A["a0"].weight.data
                # 投影去除舊任務方向 (I - P)A
                proj = A0.T @ A0
                proj = proj / (torch.norm(proj) + 1e-6)
                A -= A @ proj

    # ------------------------------------------------------------
    # Layer-wise weighted merge + Entropy-gated residual
    # ------------------------------------------------------------
    print("🧠 Performing entropy-gated hierarchical merge...")
    peft_model.load_adapter(adapter_dirs[0], adapter_name="he_oa_recall")
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if "he_oa_recall" not in module.lora_A: continue
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
                # Entropy-gated residual: 熵越大融合越保守
                g = torch.sigmoid(-3.0 * (entropies[q] - entropies.mean()))
                delta_sum.add_(deltas[q] * w[q] * g)

            B_star, A_star = svd_factorize_delta(delta_sum, r_here)
            module.lora_A["he_oa_recall"].weight.data.copy_(A_star)
            module.lora_B["he_oa_recall"].weight.data.copy_(B_star)

    peft_model.set_adapter("he_oa_recall")
    merged = peft_model.merge_and_unload()
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n🎉 OA+HE-RECALL fused → {args.output_dir}\n")

if __name__ == "__main__":
    main()
