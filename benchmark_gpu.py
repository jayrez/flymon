"""Small CUDA smoke benchmark; unrelated to malecns network queries."""
import time
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside this container")

device = torch.device("cuda:0")
print(f"GPU: {torch.cuda.get_device_name(device)}")
print(f"PyTorch: {torch.__version__}, CUDA runtime: {torch.version.cuda}")
a = torch.randn((2048, 2048), device=device)
b = torch.randn((2048, 2048), device=device)
out = torch.empty_like(a)
for _ in range(5):
    torch.mm(a, b, out=out)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(20):
    torch.mm(a, b, out=out)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"2048x2048 FP32 matrix multiply: {elapsed / 20 * 1000:.3f} ms/run")
print(f"Approximate throughput: {2 * 2048**3 * 20 / elapsed / 1e12:.3f} TFLOP/s")
