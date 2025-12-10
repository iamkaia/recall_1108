#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RECALL Paper-Strict merging (Option A, bf16)

- Anchor task: RACE (用 RACE 的 dataset 抽典型樣本)
- KMeans per layer 抽典型樣本
- Multi-layer hidden representation
- RBF similarity + layer-wise softmax
- 把所有 LoRA 真正 merge 回 base model，輸出成完整 HF 模型
"""

import argparse
import json
import os
from typing import List, Dict

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl_inputs(path: str, max_samples: int) -> List[str]:
    """從 jsonl 中讀出 input 欄位，最多 max_samples 筆"""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            texts.append(ex["input"])
            if 0 < max_samples <= len(texts):
                break
    return texts


@torch.no_grad()
def extract_layer_features(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 4,
) -> List[torch.Tensor]:
    """
    回傳一個 list，每個元素是 [n_samples, hidden_dim]，
    對應各個 transformer layer（不包含 embedding layer）。

    feature 是做 token 維度的 mean-pooling。
    """
    model.eval()
    all_layer_feats: List[torch.Tensor] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        outputs = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states  # (emb, layer1, ..., layerL)
        attn_mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]

        batch_layer_feats: List[torch.Tensor] = []
        for li in range(1, len(hidden_states)):
            h = hidden_states[li]  # [B, T, D]
            h = h * attn_mask
            lengths = attn_mask.sum(dim=1).clamp(min=1)  # [B, 1]
            h_mean = (h.sum(dim=1) / lengths).to("cpu")  # [B, D]
            batch_layer_feats.append(h_mean)

        if not all_layer_feats:
            all_layer_feats = batch_layer_feats
        else:
            for i in range(len(all_layer_feats)):
                all_layer_feats[i] = torch.cat(
                    [all_layer_feats[i], batch_layer_feats[i]], dim=0
                )

        del outputs, hidden_states, enc, batch_layer_feats
        torch.cuda.empty_cache()
    return all_layer_feats


def kmeans_representatives(
    x: torch.Tensor, k: int, iters: int = 10
) -> torch.Tensor:
    """
    CPU 上簡單做 k-means，
    x: [n, d]
    return: k 個代表點的 index (一個 cluster 一個)
    """
    x = x.detach().clone()
    n, d = x.shape
    if k >= n:
        return torch.arange(n)

    idx = torch.randperm(n)[:k]
    centers = x[idx].clone()

    for _ in range(iters):
        dists = torch.cdist(x, centers)  # [n, k]
        labels = dists.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = x[mask].mean(dim=0)
            else:
                new_centers[j] = centers[j]
        centers = new_centers

    dists = torch.cdist(x, centers)
    rep_idx = dists.argmin(dim=0)  # [k]
    return rep_idx


def rbf_similarity(
    x: torch.Tensor, y: torch.Tensor, gamma: float = None
) -> float:
    """
    x, y: [m, d] on CPU
    回傳 RBF 相似度的平均值
    """
    assert x.shape == y.shape
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    diff = x - y
    dist2 = (diff * diff).sum(dim=1)  # [m]
    sim = torch.exp(-gamma * dist2).mean().item()
    return sim


def parse_layer_index_from_key(key: str) -> int:
    """
    從 state_dict key 裡 parse 出 layer index
    例如: "model.layers.12.self_attn.q_proj.weight" -> 12
    """
    parts = key.split(".")
    try:
        idx = parts.index("layers")
        layer_id = int(parts[idx + 1])
        return layer_id
    except Exception:
        return -1


def main():
    parser = argparse.ArgumentParser(description="RECALL Paper-Strict LoRA merging")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base HF model name or path",
    )
    parser.add_argument(
        "--sst2_adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter for SST-2",
    )
    parser.add_argument(
        "--squad2_adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter for SQuAD2.0",
    )
    parser.add_argument(
        "--iwslt_adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter for IWSLT2017",
    )
    parser.add_argument(
        "--race_adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter for RACE",
    )
    parser.add_argument(
        "--medmcqa_adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter for MedMCQA",
    )
    parser.add_argument(
        "--anchor_dataset",
        type=str,
        required=True,
        help="用來抽典型樣本的 JSONL（例如 RACE train jsonl）",
    )
    parser.add_argument(
        "--max_anchor_samples",
        type=int,
        default=2000,
        help="用在 KMeans 的 anchor 最大樣本數",
    )
    parser.add_argument(
        "--k_per_layer",
        type=int,
        default=20,
        help="每層 KMeans 的 cluster 數 k",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="抽 feature 時的 batch size",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="抽 feature 時的 max token length",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="最後 fused 完整模型的輸出資料夾",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) 讀 base model（用來最後 load fused weights & save）
    print("[INFO] Loading base model (CPU, bf16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )
    base_model.config.use_cache = False

    # 2) 再讀一份 backbone 給 PEFT（只拿來抽 feature）
    print("[INFO] Loading PEFT backbone (CPU, bf16)...")
    peft_backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )
    peft_backbone.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 3) 把所有 LoRA adapter 掛到同一個 PeftModel 上
    print("[INFO] Attaching LoRA adapters...")
    peft_model = PeftModel.from_pretrained(
        peft_backbone, args.sst2_adapter, adapter_name="sst2"
    )
    peft_model.load_adapter(args.squad2_adapter, adapter_name="squad2")
    peft_model.load_adapter(args.iwslt_adapter, adapter_name="iwslt2017")
    peft_model.load_adapter(args.race_adapter, adapter_name="race")
    peft_model.load_adapter(args.medmcqa_adapter, adapter_name="medmcqa")

    # 4) 讀 anchor dataset（這裡是 RACE）+ 用 RACE adapter 抽 feature
    print(f"[INFO] Loading anchor dataset from {args.anchor_dataset}")
    anchor_texts = load_jsonl_inputs(args.anchor_dataset, args.max_anchor_samples)
    if len(anchor_texts) == 0:
        raise RuntimeError("Anchor dataset has 0 samples.")

    print(f"[INFO] Anchor samples: {len(anchor_texts)}")

    print("[INFO] Extracting anchor features with 'race' adapter...")
    peft_model.to(device)
    peft_model.set_adapter("race")
    anchor_layer_feats = extract_layer_features(
        peft_model,
        tokenizer,
        anchor_texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    peft_model.to("cpu")
    torch.cuda.empty_cache()

    num_layers = len(anchor_layer_feats)
    print(f"[INFO] Num transformer layers (hidden_states-1): {num_layers}")

    # 5) 每層做 KMeans，選代表樣本，合併成 D_type
    print("[INFO] Running KMeans per layer to select typical samples...")
    all_rep_idx: List[int] = []
    for i, feat in enumerate(anchor_layer_feats):
        n = feat.shape[0]
        k = min(args.k_per_layer, n)
        rep_idx = kmeans_representatives(feat, k=k, iters=10)
        all_rep_idx.extend(rep_idx.tolist())
        print(f"  Layer {i}: picked {k} reps")

    typical_idx = sorted(set(all_rep_idx))
    print(f"[INFO] Total unique typical samples: {len(typical_idx)}")

    typical_texts = [anchor_texts[i] for i in typical_idx]

    # 6) 對 D_type，在 base + 5 個 task model 上都抽 feature
    model_names = ["base", "sst2", "squad2", "iwslt2017", "race", "medmcqa"]
    rep_by_model: Dict[str, List[torch.Tensor]] = {}

    # base model
    print("[INFO] Extracting features for base model...")
    base_model.to(device)
    base_feats = extract_layer_features(
        base_model,
        tokenizer,
        typical_texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    base_model.to("cpu")
    torch.cuda.empty_cache()
    rep_by_model["base"] = base_feats

    # helper: 針對每個 adapter 抽 feature
    def extract_for_adapter(name: str):
        print(f"[INFO] Extracting features for adapter: {name}")
        peft_model.to(device)
        peft_model.set_adapter(name)
        feats = extract_layer_features(
            peft_model,
            tokenizer,
            typical_texts,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        peft_model.to("cpu")
        torch.cuda.empty_cache()
        rep_by_model[name] = feats

    for name in ["sst2", "squad2", "iwslt2017", "race", "medmcqa"]:
        extract_for_adapter(name)

    # 7) 計算每一層對 anchor (race) 的 RBF 相似度
    print("[INFO] Computing RBF similarities...")
    num_models = len(model_names)
    S = torch.zeros(num_layers, num_models, dtype=torch.float32)

    for i in range(num_layers):
        anchor_feat = rep_by_model["race"][i]  # [m, d]
        for j, name in enumerate(model_names):
            feat_q = rep_by_model[name][i]
            sim = rbf_similarity(anchor_feat, feat_q)
            S[i, j] = sim
        if i % 4 == 0 or i == num_layers - 1:
            sims_str = ", ".join(
                f"{model_names[j]}={S[i,j].item():.4f}" for j in range(num_models)
            )
            print(f"  Layer {i}: {sims_str}")

    # 8) 對每層做 softmax，得到 layer-wise merge weights
    print("[INFO] Computing layer-wise softmax weights...")
    W = torch.softmax(S, dim=1)  # [L, num_models]
    model_index = {name: idx for idx, name in enumerate(model_names)}

    # 9) 讀各個 adapter 的 state_dict + LoRA config (假設 r, alpha 相同)
    print("[INFO] Loading adapter state dicts for merge...")
    adapter_paths = {
        "sst2": args.sst2_adapter,
        "squad2": args.squad2_adapter,
        "iwslt2017": args.iwslt_adapter,
        "race": args.race_adapter,
        "medmcqa": args.medmcqa_adapter,
    }
    adapter_sds: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, path in adapter_paths.items():
        safepath = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(safepath):
            raise FileNotFoundError(f"{safepath} not found for adapter {name}")
        adapter_sds[name] = load_file(safepath)
        print(f"  Loaded {name} adapter from {safepath}")

    # LoRA config：拿一個 adapter（例如 sst2）當代表
    example_cfg = os.path.join(args.sst2_adapter, "adapter_config.json")
    with open(example_cfg, "r", encoding="utf-8") as f:
        lora_cfg = json.load(f)
    r = lora_cfg.get("r", lora_cfg.get("lora_r", 8))
    alpha = lora_cfg.get("lora_alpha", 32)
    scaling = alpha / r
    print(f"[INFO] LoRA config: r={r}, alpha={alpha}, scaling={scaling}")

    # 10) 建立新的 state_dict：把 LoRA 影響 merge 回每一層的 weight
    print("[INFO] Merging weights into new state_dict...")
    base_sd = base_model.state_dict()
    new_sd = {}

    for key, base_weight in base_sd.items():
        layer_id = parse_layer_index_from_key(key)
        # 只處理 transformer layer 裡的 .weight；其他直接照搬
        if (
            layer_id >= 0
            and key.endswith(".weight")
            and key.startswith("model.layers.")
        ):
            w_layer = W[layer_id]  # [num_models]
            delta = torch.zeros_like(base_weight, dtype=torch.float32)

            # 對每個 task 的 LoRA 做 delta，乘上該層的 softmax 權重
            for name in ["sst2", "squad2", "iwslt2017", "race", "medmcqa"]:
                idx = model_index[name]
                w_q = w_layer[idx].item()
                if w_q == 0.0:
                    continue
                sd_q = adapter_sds[name]
                a_key = key.replace(".weight", ".lora_A.default.weight")
                b_key = key.replace(".weight", ".lora_B.default.weight")
                if a_key not in sd_q or b_key not in sd_q:
                    # 這個 adapter 沒有動到這個 weight
                    continue
                A = sd_q[a_key].to(torch.float32)  # [r, in]
                B = sd_q[b_key].to(torch.float32)  # [out, r]
                delta_q = (B @ A) * (scaling * w_q)
                if delta_q.shape != delta.shape:
                    print(
                        f"[WARN] Shape mismatch for {key} in adapter {name}, "
                        f"expected {tuple(delta.shape)}, got {tuple(delta_q.shape)}; skipping."
                    )
                    continue
                delta += delta_q

            fused = base_weight.to(torch.float32) + delta
            new_sd[key] = fused.to(base_weight.dtype)
        else:
            new_sd[key] = base_weight

    # 11) 把 fused state_dict load 回 base_model，存成一個完整 HF 模型
    print("[INFO] Loading fused state_dict into base model...")
    base_model.load_state_dict(new_sd)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Saving fused model to {args.output_dir}")
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[INFO] Done. Paper-strict RECALL-style merged model saved.")


if __name__ == "__main__":
    main()
