#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper-Strict RECALL merging (5 tasks, LoRA, LLaMA-2-7b-chat-hf, bf16)

- 從各個 LoRA checkpoint 抽 representation
- KMeans 在 anchor task (預設 RACE) 上挑典型樣本
- 對每層、每個 LoRA 用 RBF similarity 算 softmax 權重
- 在 LoRA 參數層面做 layer-wise 加權融合，輸出一個新的 LoRA adapter

使用前請確認：
- transformers, peft, safetensors, scikit-learn 都已安裝
- 5 個 LoRA 路徑 & 5 個 dataset 路徑依你機器實際情況修改
"""

import os
import json
import random
import gc
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from safetensors.torch import load_file
from sklearn.cluster import KMeans
from tqdm import tqdm



# =========================================
# 全域設定（請依你機器路徑微調）
# =========================================
CONFIG = {
    "base_model": "meta-llama/Llama-2-7b-chat-hf",
    # 這裡給的是「LoRA 任務資料夾根目錄」，script 會自己找最後一個 checkpoint-*，如果沒有就用根目錄
    "adapter_roots": {
        "sst2":      "/home/kaia/LLaMA-Factory/saves/llama2-7b-chat-hf/lora/sft_sst2",
        "squad2":    "/home/kaia/LLaMA-Factory/saves/llama2-7b-chat-hf/lora/sft_squad2",
        "iwslt2017": "/home/kaia/LLaMA-Factory/saves/llama2-7b-chat-hf/lora/sft_iwslt",
        "race":      "/home/kaia/LLaMA-Factory/saves/llama2-7b-chat-hf/lora/sft_race",
        "medmcqa":   "/home/kaia/LLaMA-Factory/saves/llama2-7b-chat-hf/lora/sft_medmcqa",
    },
    "datasets": {
        "sst2":      "/home/kaia/recall_1108/datasets_llama/sst2/train.jsonl",
        "squad2":    "/home/kaia/recall_1108/datasets_llama/squad2/train.jsonl",
        "iwslt2017": "/home/kaia/recall_1108/datasets_llama/iwslt2017/train.jsonl",
        "race":      "/home/kaia/recall_1108/datasets_llama/race/train.jsonl",
        "medmcqa":   "/home/kaia/recall_1108/datasets_llama/medmcqa/train.jsonl",
    },
    # KMeans 用來挑「典型樣本」會先在每個 task 隨機抽多少筆當 pool
    "cluster_samples_per_task": 2000,
    # 每個 task 最後挑出幾個代表樣本 (m)
    "samples_per_task": 50,
    # tokenizer 的最大長度
    "max_length": 512,
    # batch size (feature 抽取時)
    "batch_size": 4,
    # RBF gamma（大概 1/hidden_dim 就好）
    "rbf_gamma": None,  # 若為 None 就會自動設為 1/hidden_dim
    # anchor task (Algorithm 1 裡面的 MN) —— 這裡用 RACE 當最後一個任務
    "anchor_task": "race",
    # 最後 fused LoRA 存放位置
    "output_dir": "/home/kaia/recall_1108/recall_fused_strict",
    # random seed
    "seed": 42,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================
# 小工具
# =========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

'''
def resolve_adapter_path(root: str) -> str:
    """
    給一個 sft_xxx 資料夾，回傳「最後一個 checkpoint-*」，
    如果沒有 checkpoint 就用 root 本身。
    """
    ckpts = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if os.path.isdir(full) and name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-")[1])
                    ckpts.append((step, full))
                except Exception:
                    pass
    if ckpts:
        ckpts.sort(key=lambda x: x[0])
        best_path = ckpts[-1][1]
        print(f"[INFO] Using latest checkpoint: {best_path}")
        return best_path
    else:
        print(f"[INFO] No checkpoint-* found under {root}, using root as adapter path.")
        return root


def load_jsonl_inputs(path: str, max_n: int = None) -> List[str]:
    """讀 jsonl，取 'input' 欄位，最多 max_n 筆（如果指定）。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj["input"])
            if max_n is not None and len(data) >= max_n:
                break
    return data
'''

def resolve_adapter_path(root: str) -> str:
    """
    永遠使用最外層的 LoRA（adapter_model.safetensors）而不是 checkpoint。
    """
    print(f"[INFO] Using top-level LoRA at: {root}")
    return root


def build_base_tokenizer():
    tok = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    # LLaMA-2 沒有 pad_token，直接用 eos 來 pad
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ============== representation 抽取 ==============

def get_embeddings_last_layer(model, tokenizer, texts: List[str]) -> np.ndarray:
    """
    只用最後一層 hidden state + mean pooling (KMeans 用)
    """
    all_vecs = []
    bs = CONFIG["batch_size"]
    max_len = CONFIG["max_length"]

    model.eval()
    for i in tqdm(range(0, len(texts), bs), desc="Extract feats (last layer)"):
        batch = texts[i:i+bs]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]               # (B, T, D)
        mask = tokenized["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)    # (B, D)

        all_vecs.append(pooled.cpu())

        del tokenized, outputs, hidden, mask, pooled
        torch.cuda.empty_cache()

    return torch.cat(all_vecs, dim=0).numpy()


def select_typical_samples(anchor_model, tokenizer, texts: List[str]) -> List[str]:
    """
    在 anchor task 上用 KMeans 從 texts 中選出 m 個典型樣本。
    """
    print(f"[STEP] Selecting typical samples from anchor task '{CONFIG['anchor_task']}'...")
    m = CONFIG["samples_per_task"]

    if len(texts) <= m:
        print(f"[WARN] texts <= samples_per_task ({len(texts)} <= {m}), 全部當典型樣本用。")
        return texts

    # 抽一個 pool (最多 cluster_samples_per_task)
    pool_n = min(CONFIG["cluster_samples_per_task"], len(texts))
    rng = np.random.default_rng(CONFIG["seed"])
    pool_idx = rng.choice(len(texts), size=pool_n, replace=False)
    pool_texts = [texts[i] for i in pool_idx]

    # 用 anchor model 抽 feature
    feats = get_embeddings_last_layer(anchor_model, tokenizer, pool_texts)  # (pool_n, D)
    hidden_dim = feats.shape[1]
    if CONFIG["rbf_gamma"] is None:
        CONFIG["rbf_gamma"] = 1.0 / hidden_dim
        print(f"[INFO] Set RBF gamma = 1.0 / hidden_dim = {CONFIG['rbf_gamma']:.6f}")

    print(f"[INFO] KMeans on {pool_n} samples, k = {m} ...")
    kmeans = KMeans(n_clusters=m, random_state=CONFIG["seed"], n_init="auto")
    kmeans.fit(feats)

    centers = kmeans.cluster_centers_         # (m, D)
    labels = kmeans.labels_                   # (pool_n,)

    # 每個 cluster 找離 center 最近的那一個 index
    selected_pool_idx = []
    for c in range(m):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        sub = feats[idxs] - centers[c]
        dist2 = np.sum(sub * sub, axis=1)
        best_local = idxs[np.argmin(dist2)]
        selected_pool_idx.append(best_local)

    # 回到原始 texts 的 index
    global_idx = [pool_idx[i] for i in selected_pool_idx]
    global_idx = sorted(set(global_idx))
    print(f"[INFO] Selected {len(global_idx)} typical samples.")
    return [texts[i] for i in global_idx]


def get_layerwise_reps(model, tokenizer, texts: List[str]) -> List[np.ndarray]:
    """
    對一組 texts，用 LoRA model 抽每一層的 mean-pooled hidden state
    回傳：list 長度 L，每個元素 shape = (m, hidden_dim)
    """
    print("[STEP] Extracting layer-wise representations...")
    bs = CONFIG["batch_size"]
    max_len = CONFIG["max_length"]

    model.eval()
    all_layers: List[List[torch.Tensor]] = None
    for i in tqdm(range(0, len(texts), bs), desc="Layer-wise feats"):
        batch = texts[i:i+bs]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True)

        hs = outputs.hidden_states[1:]  # drop embedding layer, keep L transformer layers
        mask = tokenized["attention_mask"].unsqueeze(-1).float()

        if all_layers is None:
            L = len(hs)
            all_layers = [[] for _ in range(L)]

        for li, h in enumerate(hs):
            pooled = (h * mask).sum(1) / mask.sum(1)  # (B, D)
            all_layers[li].append(pooled.cpu())

        del tokenized, outputs, hs, mask, pooled
        torch.cuda.empty_cache()

    # merge & to numpy
    reps = [torch.cat(chunks, dim=0).numpy() for chunks in all_layers]
    return reps  # list of length L, each (m, D)


# ============== RBF 相似度 + 權重 ==============

def rbf_similarity(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    """
    a, b: (m, D)
    回傳: scalar = 平均 RBF(sim) over m
    """
    diff = a - b
    dist2 = np.sum(diff * diff, axis=1)  # (m,)
    sim = np.exp(-gamma * dist2)
    return float(sim.mean())


def compute_layer_weights(reps_by_task: Dict[str, List[np.ndarray]], tasks: List[str]) -> Dict[int, Dict[str, float]]:
    """
    reps_by_task[task][layer] = (m, D)
    回傳: layer_weights[layer_idx][task] = softmax weight
    """
    anchor = CONFIG["anchor_task"]
    gamma = CONFIG["rbf_gamma"]
    assert anchor in reps_by_task, f"anchor task '{anchor}' not in reps_by_task"

    num_layers = len(next(iter(reps_by_task.values())))
    layer_weights: Dict[int, Dict[str, float]] = {}

    print("[STEP] Computing RBF similarities + softmax weights per layer...")
    for li in range(num_layers):
        sims = []
        for t in tasks:
            sim = rbf_similarity(reps_by_task[anchor][li], reps_by_task[t][li], gamma)
            sims.append(sim)

        sims = np.array(sims)
        w = np.exp(sims) / np.exp(sims).sum()
        layer_weights[li] = {t: float(w[idx]) for idx, t in enumerate(tasks)}

    return layer_weights


# ============== LoRA merge ==============

def get_lora_state_dict(adapter_path: str) -> Dict[str, torch.Tensor]:
    """
    讀取某個 LoRA adapter 的 safetensors state dict
    """
    st_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"adapter_model.safetensors not found under {adapter_path}")
    sd = load_file(st_path)
    return sd


def merge_lora_weights(layer_weights: Dict[int, Dict[str, float]],
                       lora_sds: Dict[str, Dict[str, torch.Tensor]],
                       tasks: List[str],
                       template_peft: PeftModel) -> PeftModel:
    """
    layer_weights: layer_idx -> {task: weight}
    lora_sds: task -> state_dict of its LoRA weights
    template_peft: 一個帶 LoRA 結構的 PeftModel，會在此基礎上蓋掉 LoRA 權重
    """
    print("[STEP] Merging LoRA weights (layer-wise, RBF-softmax)...")

    # 所有 LoRA 的共同 key（避免有 missing 的 warning）
    common_keys = set(lora_sds[tasks[0]].keys())
    for t in tasks[1:]:
        common_keys &= set(lora_sds[t].keys())
    common_keys = sorted(list(common_keys))

    print(f"[INFO] Number of common LoRA params: {len(common_keys)}")

    new_state = template_peft.state_dict()

    for name in tqdm(common_keys, desc="Merging layers"):
        # 判斷是不是 transformer block 的參數
        if "model.layers." in name:
            try:
                after = name.split("model.layers.")[1]
                layer_idx = int(after.split(".")[0])
            except Exception:
                layer_idx = None
        else:
            layer_idx = None

        if layer_idx is not None and layer_idx in layer_weights:
            # 這層有專屬的 RECALL 權重
            ws = np.array([layer_weights[layer_idx][t] for t in tasks], dtype=np.float32)
        else:
            # 非 transformer 層（例如 lm_head），平均就好
            ws = np.ones(len(tasks), dtype=np.float32) / len(tasks)

        # 開始融合這一個 parameter
        merged = None
        for ti, t in enumerate(tasks):
            w = ws[ti]
            param = lora_sds[t][name].to(torch.float32)
            if merged is None:
                merged = w * param
            else:
                merged += w * param

        # 回寫到 new_state (保持原來 dtype，例如 fp16/bf16)
        target_dtype = new_state[name].dtype
        new_state[name] = merged.to(target_dtype)

    template_peft.load_state_dict(new_state, strict=False)
    return template_peft


# =========================================
# 主流程
# =========================================

def main():
    gc.collect()
    torch.cuda.empty_cache()

    set_seed(CONFIG["seed"])
    tasks = ["sst2", "squad2", "iwslt2017", "race", "medmcqa"]
    anchor = CONFIG["anchor_task"]
    assert anchor in tasks, "anchor_task must be in tasks list"

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("========== [STEP 0] Load tokenizer & base config ==========")
    tokenizer = build_base_tokenizer()
    tokenizer.padding_side = "right"
    base_cfg = AutoConfig.from_pretrained(CONFIG["base_model"])
    print(f"[INFO] num_hidden_layers = {base_cfg.num_hidden_layers}")

    print("========== [STEP 1] Load datasets ==========")
    all_texts = {}
    for t in tasks:
        path = CONFIG["datasets"][t]
        texts = load_jsonl_inputs(path)
        all_texts[t] = texts
        print(f"[DATA] {t}: {len(texts)} samples from {path}")

    print("========== [STEP 2] Select typical samples on anchor task ==========")
    # 先 load anchor 的 LoRA model
    anchor_root = CONFIG["adapter_roots"][anchor]
    anchor_adapter = resolve_adapter_path(anchor_root)

    base_anchor = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    anchor_model = PeftModel.from_pretrained(base_anchor, anchor_adapter)
    anchor_model.eval()

    typical_texts = select_typical_samples(anchor_model, tokenizer, all_texts[anchor])
    m = len(typical_texts)
    print(f"[INFO] Using m={m} typical samples for RECALL similarities.")

    # 釋放 anchor_model，等等會重新載入其他 LoRA
    del anchor_model, base_anchor
    torch.cuda.empty_cache()

    print("========== [STEP 3] Extract layer-wise reps for each task ==========")
    reps_by_task: Dict[str, List[np.ndarray]] = {}
    for t in tasks:
        print(f"\n[Task {t}] ----")
        root = CONFIG["adapter_roots"][t]
        adapter_path = resolve_adapter_path(root)

        base = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"],
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        lora_model = PeftModel.from_pretrained(
            base,
            adapter_path,
            torch_dtype=torch.bfloat16
        )

        lora_model.eval()


        reps = get_layerwise_reps(lora_model, tokenizer, typical_texts)
        reps_by_task[t] = reps

        del lora_model, base
        torch.cuda.empty_cache()


    print("========== [STEP 4] Compute RBF-softmax weights per layer ==========")
    layer_weights = compute_layer_weights(reps_by_task, tasks)

    print("========== [STEP 5] Load LoRA state dicts ==========")
    lora_sds = {}
    for t in tasks:
        root = CONFIG["adapter_roots"][t]
        adapter_path = resolve_adapter_path(root)
        lora_sds[t] = get_lora_state_dict(adapter_path)

    print("========== [STEP 6] Build template PEFT model ==========")
    # 用 anchor 的 adapter 結構當 template
    anchor_adapter_path = resolve_adapter_path(CONFIG["adapter_roots"][anchor])
    base_for_merge = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # merge 在 CPU 上做就好
    )
    template_peft = PeftModel.from_pretrained(base_for_merge, anchor_adapter_path)
    template_peft.eval()

    print("========== [STEP 7] Merge LoRA weights ==========")
    fused_model = merge_lora_weights(layer_weights, lora_sds, tasks, template_peft)

    print("========== [STEP 8] Save fused adapter ==========")
    fused_model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print(f"🎉 Fusion complete. Saved fused LoRA to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
