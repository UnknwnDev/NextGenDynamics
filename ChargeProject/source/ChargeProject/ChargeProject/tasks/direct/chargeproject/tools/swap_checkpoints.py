import torch

# Paths
old_ckpt_path = "checkpoints/point_to_point_v1.2.pt"
new_ckpt_path = "checkpoints/agent_500.pt"

# Load checkpoints
old_ckpt = torch.load(old_ckpt_path, map_location="cpu")
new_ckpt = torch.load(new_ckpt_path, map_location="cpu")

# Inspect keys (layer names)
print("Old keys:", list(old_ckpt["policy"].keys()))
print("New keys:", list(new_ckpt["policy"].keys()))

# Transfer matching weights
for key in new_ckpt["policy"].keys():
    if key in old_ckpt["policy"] and old_ckpt["policy"][key].shape == new_ckpt["policy"][key].shape:
        new_ckpt["policy"][key] = old_ckpt["policy"][key].clone()
        print(f"Transferred: {key}")
    else:
        print(f"Skipped: {key} (new or mismatched)")

# Same for value network
for key in new_ckpt["value"].keys():
    if key in old_ckpt["value"] and old_ckpt["value"][key].shape == new_ckpt["value"][key].shape:
        new_ckpt["value"][key] = old_ckpt["value"][key].clone()
        print(f"Transferred: {key}")
    else:
        print(f"Skipped: {key} (new or mismatched)")

# Save initialized RNN checkpoint
torch.save(new_ckpt, "checkpoints/memory_ptp_transfer.pt")
