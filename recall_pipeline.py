import json, torch, os, numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
from safetensors.torch import save_file

torch.set_grad_enabled(False)

########################################
# 🔧 Load Config
########################################
cfg = json.load(open("tasks.json"))
device = "cuda"
dtype = torch.bfloat16

########################################
# 🔧 Load Base Model
########################################
print("\n📦 Loading base model...")
base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto", torch_dtype=dtype)
tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])


########################################
# Step 1️⃣ — Locate latest checkpoint for each task
########################################
def get_latest_ckpt(path):
    ckpts = [d for d in os.listdir(path) if d.startswith("checkpoint")]
    return os.path.join(path, sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1])

adapter_paths = {
    task: get_latest_ckpt(os.path.join(cfg["adapter_root"], f"sft_{task}"))
    for task in cfg["datasets"]
}

print("\n🗂 Found adapters:")
for k,v in adapter_paths.items():
    print(f"   {k}: {v}")

########################################
# Step 2️⃣ Extract typical samples via KMeans
########################################
print("\n📌 Extracting typical samples (KMeans)...")

def load_texts(path):
    print(f"[INFO] loading texts from: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[ERROR] {path} line {i}: {e}")
                print("Line content (first 200 chars):", line[:200])
                raise
            data.append(obj["input"])
    print(f"[OK] {path}: {len(data)} samples loaded")
    return data

samples = {
    task: load_texts(cfg["datasets"][task])
    for task in adapter_paths
}

def get_embeddings(model, texts):
    reps = []
    for t in texts:
        tok = tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to(device)
        hs = model.model.layers[-1].output  # last hidden layer
        reps.append(hs.mean(dim=1).detach().cpu().numpy())
    return np.vstack(reps)

typical_samples = {}

for task in samples:
    texts = samples[task][:200]
    emb = get_embeddings(base, texts)
    kmeans = KMeans(n_clusters=cfg["num_typical_samples"]).fit(emb)
    typical_samples[task] = [texts[i] for i in kmeans.cluster_centers_.argsort(axis=None)[:50]]

print("🎯 Selected typical samples per task.")


########################################
# Step 3️⃣ Compute representations for each adapter
########################################
representations = {t: [] for t in adapter_paths}

print("\n📌 Extracting task representations...")
for task, path in adapter_paths.items():
    print(f" → Loading LoRA: {task}")
    model = PeftModel.from_pretrained(base, path).eval()

    for txt in typical_samples[task]:
        tok = tokenizer(txt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.model(tok.input_ids, output_hidden_states=True)
        reps = torch.stack([h.mean() for h in out.hidden_states]).float().cpu().numpy()
        representations[task].append(reps)

print("📌 Representations extracted.")


########################################
# Step 4️⃣ RBF Kernel Similarity + Layer-wise merge
########################################
def rbf(x, y, gamma=0.9):
    return np.exp(-gamma * np.linalg.norm(x-y)**2)

weights = {}

print("\n🔧 Computing similarity weights...")
for layer in range(len(representations["sst2"][0])):
    sims = {
        task: np.mean([rbf(representations["sst2"][i][layer], representations[task][i][layer]) for i in range(10)])
        for task in representations
    }

    exp_weights = np.array([np.exp(v) for v in sims.values()])
    norm = exp_weights / exp_weights.sum()
    weights[layer] = dict(zip(sims.keys(), norm))

print("\n🎛 Layer-wise weights ready.")


########################################
# Step 5️⃣ Merge weights into final model
########################################
print("\n⚙️ Merging model...")

full_sd = base.state_dict()

for layer, layer_weights in weights.items():
    for param_name in full_sd.keys():
        if f"layers.{layer}" in param_name and "lora" in param_name:
            merged = sum(
                layer_weights[t] * torch.load(os.path.join(adapter_paths[t], "adapter_model.safetensors")).get(param_name, full_sd[param_name])
                for t in adapter_paths
            )
            full_sd[param_name] = merged

print("\n🚀 Saving final fused model...")
os.makedirs(cfg["output_dir"], exist_ok=True)
save_file({k: v.cpu() for k,v in full_sd.items()}, f"{cfg['output_dir']}/recall_fused_model.safetensors")

print("\n🎉 RECALL Fusion Complete.")
