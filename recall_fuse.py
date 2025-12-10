import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from tqdm import tqdm

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
MODELS_LIST_PATH = "/home/kaia/LLaMA-Factory/saves/llama2-7b-chat-hf/lora/models.txt"
OUTPUT_DIR = "/home/kaia/LLaMA-Factory/recall_fused"

torch.set_grad_enabled(False)

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto"
)

# ---- LoRA model paths ----
with open(MODELS_LIST_PATH) as f:
    model_paths = [line.strip() for line in f.readlines()]

print(f"Loading {len(model_paths)} LoRA adapters...")

adapters = []
for p in model_paths:
    model = PeftModel.from_pretrained(base_model, p)
    model.eval()
    adapters.append(model.to("cuda"))

# ---- Get representations ----
print("Extracting typical sample representations...")

with torch.no_grad():
    inputs = tokenizer("Hello world!", return_tensors="pt").to("cuda")

    reps = []
    for model in adapters:
        outputs = model.model(inputs.input_ids, output_hidden_states=True)
        rep = torch.stack([h.mean(dim=1).squeeze() for h in outputs.hidden_states]).mean(dim=0)

        # 🔥 fix: convert safely to float32
        reps.append(rep.float().cpu().numpy())


# ---- RBF Similarity ----
print("Computing similarity weights...")

def rbf(x, y, gamma=0.1):
    return np.exp(-gamma * np.sum((x - y) ** 2))

scores = np.array([rbf(reps[-1], r) for r in reps])
weights = torch.softmax(torch.tensor(scores), dim=0).numpy()

print(f"Similarity Weights: {weights}")

# ---- Merge process ----
print("Merging LoRA parameters...")

merged_state = base_model.state_dict()

for name, param in tqdm(base_model.state_dict().items(), desc="Merging layers"):

    # 只處理 LoRA 權重
    if "lora_A" not in name and "lora_B" not in name:
        continue

    # 🔥 修正 key (LLaMA-Factory 沒有 .default namespace)
    key = name.replace(".default", "")

    merged = None
    for w, model in zip(weights, adapters):

        model_sd = model.state_dict()

        # 若 key 找不到 → 略過，不報錯
        if key not in model_sd:
            continue

        sd = model_sd[key].float()

        if merged is None:
            merged = sd * w
        else:
            merged += sd * w

    if merged is not None:
        param.copy_(merged.to(param.device))



base_model.load_state_dict(merged_state)
base_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"🎉 Fusion complete. Saved to: {OUTPUT_DIR}")
