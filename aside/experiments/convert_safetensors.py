import torch
from safetensors.torch import save_file
import os
import json
from tqdm import tqdm

model_dir = "Qwen2.5-7B_ASIDE"
index_path = os.path.join(model_dir, "pytorch_model.bin.index.json")
with open(index_path, "r") as f:
    index = json.load(f)

shard_files = set(index["weight_map"].values())

for shard_file in tqdm(shard_files,total =len(list(shard_files))):
    shard_path = os.path.join(model_dir, shard_file)
    print(f"Converting {shard_path} ...")
    state_dict = torch.load(shard_path, map_location="cpu")
    safetensors_path = shard_path.replace(".bin", ".safetensors")
    save_file(state_dict, safetensors_path)
    print(f"Saved {safetensors_path}")