#!/usr/bin/env python3
# merge_he_recall_v2.py — High-Entropy RECALL (z-score entropy weighting, λ=0.3)
# Kaia 版本：強化不確定性導向 + 保留 RECALL 架構穩定性

import os, re, json, argparse, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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
        prompts = ["Translate: Hello world! → French\nAnswer:",
                   "What is the capital of France?\nAnswer:",
                   "State one advantage of neural networks.\nAnswer:"]
    return prompts

def collect_hidden_states_and_entropy(model, tokenizer, prompts, pad_len):
    hidden_accum = []
    entropy_all = []
    with torch.no_grad():
        for text in prompts:
            enc = tokenizer(text, padding="max_length", truncation=True,
                            max_length=pad_len, add_special_tokens=False,
                            return_tensors="pt").to(model.device)
            out = model(**enc, output_hidden_states=True, return_dict=True)
            h = torch.stack([x[:, -1, :] for x in out.hidden_states]).squeeze(1)
            hidden_accum.append(h)
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * probs.log()).sum(dim=-1)
            entropy_all.append(ent.item())
    return torch.stack(hidden_accum, dim=0).mean(dim=0), torch.tensor(entropy_all).mean()

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapters_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="fused_he_recall_v2")
    parser.add_argument("--tasks", nargs="*", default=["sst2","squad2","iwslt2017-en-fr","race","medmcqa"])
    parser.add_argument("--samples_per_task", type=int, default=50)
    parser.add_argument("--pad_len", type=int, default=128)
    args = parser.parse_args()

    print("\n========== [STEP] High-Entropy RECALL v2 ==========\n")

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
    print(f"🧾 Using {len(prompts)} prompts for entropy weighting.")

    anchor_rep, anchor_ent = collect_hidden_states_and_entropy(peft_model, tokenizer, prompts, args.pad_len)
    all_reps = [anchor_rep]; entropies = [anchor_ent]

    for i, adir in enumerate(adapter_dirs[1:], start=1):
        print(f"📌 Load adapter a{i}: {adir}")
        peft_model.load_adapter(adir, adapter_name=f"a{i}")
        rep, ent = collect_hidden_states_and_entropy(peft_model, tokenizer, prompts, args.pad_len)
        all_reps.append(rep); entropies.append(ent); names.append(f"a{i}")

    reps = torch.stack(all_reps, dim=0).to(base.device)
    entropies = torch.tensor(entropies, device=base.device)
    num_adapters, num_layers, _ = reps.shape

    # --- z-score entropy weighting ---
    e_z = (entropies - entropies.mean()) / (entropies.std() + 1e-6)
    sim = F.cosine_similarity(reps[0].unsqueeze(0), reps, dim=-1)
    sim = sim * (1 + 0.6 * e_z.unsqueeze(1))     # λ=0.3
    weights_layerwise = F.softmax(sim, dim=0)
    weights_scalar = weights_layerwise.mean(dim=1)

    print("🧠 Layer-wise merging with z-score entropy weighting ...")
    peft_model.load_adapter(adapter_dirs[0], adapter_name="he_recall_v2")

    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if "he_recall_v2" not in module.lora_A: continue
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
            module.lora_A["he_recall_v2"].weight.data.copy_(A_star)
            module.lora_B["he_recall_v2"].weight.data.copy_(B_star)

    peft_model.set_adapter("he_recall_v2")
    merged = peft_model.merge_and_unload()
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n🎉 Fused model saved → {args.output_dir}\n")

if __name__ == "__main__":
    main()
