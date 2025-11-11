import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Para NVIDIA
print(
    f"ROCm available: {torch.version.hip is not None and torch.cuda.is_available()}"
)  # Para AMD con ROCm (PyTorch lo reporta como CUDA-like)
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
