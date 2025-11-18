import torch
import numpy as np
from pathlib import Path

# Your project path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "resources" / "spk_embeds.pt"

# Create a clean 512-dimension embedding
# (normal distribution similar to real x-vectors)
vec = np.random.normal(scale=0.6, size=(512,)).astype(np.float32)

# Save in EXACT expected format:  [1, 512]
tensor = torch.tensor(vec).unsqueeze(0)

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(tensor, OUT)

print("Saved valid embedding:", OUT)
print("Tensor shape:", tensor.shape)
