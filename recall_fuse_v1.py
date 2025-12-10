import os, json, random, torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========= CONFIG ==========
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
EXPERT_DIR = "/home/kaia/LLaMA-Factory/saves/llama2-7b-chat-hf/lora/"
DATASET_PATH = "/home/kaia/recall_1108/datasets_llama/race/train.jsonl"
OUTPUT_DIR = "./recall_fused"
SAMPLE_SIZE = 20
DEVICE = "cuda"
# ===========================

print("📌 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("📌 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16).to(DEVICE)

# ==== Step 1: Load LoRA experts ====
experts = []
paths = sorted([d for d in os.listdir(EXPERT_DIR) if d.startswith("sft_")])

print(f"📦 Found experts: {paths}")

for p in paths:
    path = os.path.join(EXPERT_DIR, p)
    print(f"🔗 Loading LoRA: {path}")
    experts.append(PeftModel.from_pretrained(model, path))

# ==== Step 2: Load RACE dataset ====
print("📖 Loading RACE dataset...")
with open(DATASET_PATH, "r") as f:
    dataset = [json.loads(line) for line in f]

samples = random.sample(dataset, SAMPLE_SIZE)

# ==== Step 3: Extract hidden states ====
print("🧠 Extracting representations...")
reps = []

for expert in tqdm(experts):
    expert.eval()
    all_layers = []

    for item in samples:
        text = f"{item['input']}"
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            #out = expert.model(**inputs, output_hidden_states=True)
            out = expert.model.model(**inputs, output_hidden_states=True)
            print("double check!!!!!\n\n")
            print(out.hidden_states[0].shape)
            h = out.hidden_states  # tuple (layers)
            layer_vecs = [v.mean(dim=1) for v in h]  # mean pool per layer

        all_layers.append(torch.stack(layer_vecs).cpu())

    reps.append(torch.stack(all_layers))  # shape: (samples, layers, dim)

reps = torch.stack(reps)  # shape: (experts, samples, layers, dim)
print(f"Reps shape: {reps.shape}")

# ==== Step 4: Similarity + softmax weights ====
def rbf(a, b, gamma=0.01):
    return torch.exp(-gamma * (a - b).pow(2).sum())

print("📊 Computing similarity matrix...")

# Reference = last expert (RACE)
ref = reps[-1]

weights = []

for layer in range(ref.shape[1]):
    sims = []

    for i in range(len(experts)):
        s = 0
        for s_idx in range(SAMPLE_SIZE):
            s += rbf(reps[i][s_idx][layer], ref[s_idx][layer])
        sims.append(s / SAMPLE_SIZE)

    sims = torch.tensor(sims)
    w = torch.softmax(sims, dim=0)  # normalize
    weights.append(w)

weights = torch.stack(weights)
print("🧮 Done computing weights.")

# ==== Step 5: Merge model ====
print("🧩 Merging parameters...")

final_sd = model.state_dict()

for name, param in tqdm(final_sd.items()):
    if "lora" in name: continue  # skip LoRA metadata
    
    merged = 0
    for i, expert in enumerate(experts):
        sd = expert.state_dict()
        if name in sd:
            merged += sd[name] * weights[:, i].mean()

    final_sd[name] = merged

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.save(final_sd, f"{OUTPUT_DIR}/pytorch_model.bin")

print(f"\n🎉 FULL RECALL fusion complete → saved at: {OUTPUT_DIR}\n")
