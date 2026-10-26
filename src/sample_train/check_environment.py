import sys

import torch
import transformers
import trl

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("TRL:", trl.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Fix the PyTorch/GPU setup before training."
    )

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory (GB):",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
)
print("Environment check passed.")
