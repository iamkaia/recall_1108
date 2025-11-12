# convert_safetensors_to_bin.py
import torch
from safetensors.torch import load_file
import os

tasks = ["sst2","squad2","iwslt2017","race","medmcqa"]
base_dir = "checkpoints_500"

for t in tasks:
    path = os.path.join(base_dir, t, "adapter_model.safetensors")
    if os.path.exists(path):
        print(f"Converting {path} ...")
        state = load_file(path)
        torch.save(state, os.path.join(base_dir, t, "adapter_model.bin"))
print("✅ Conversion done.")
