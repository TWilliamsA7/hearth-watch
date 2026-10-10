"""Step 1: PyTorch meets your GPU (RTX 3060 Ti). Run on the Windows PC:

    .\\.venv-pc\\Scripts\\python.exe edge\\pc\\01_check_gpu.py

What it shows, one PyTorch idea per section:
  1. Is CUDA (the NVIDIA GPU) visible to PyTorch?
  2. Tensors: PyTorch's arrays, and moving them between CPU and GPU.
  3. Why GPUs: the same matrix multiply timed on the CPU and on the GPU.
  4. Autograd: PyTorch computes gradients for you (the core of training/fine-tuning, step 6).
"""

import time

import torch

print("=" * 70)
print("1. Is the GPU visible?")
print("   torch version:", torch.__version__)
print("   CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("   No CUDA GPU visible. Check `nvidia-smi` works, and that you installed the CUDA "
                     "build of torch (see edge/pc/README.md), not the CPU-only one.")
gpu = torch.cuda.get_device_properties(0)
print("   GPU:", gpu.name, "| memory %.1f GB" % (gpu.total_memory / 1e9),
      "| compute capability %d.%d" % (gpu.major, gpu.minor))
print("   PyTorch was built for CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())

print("=" * 70)
print("2. Tensors live on a device")
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])          # created on the CPU
print("   a =", a.tolist(), "| shape", tuple(a.shape), "| dtype", a.dtype, "| device", a.device)
b = a.to("cuda")                                     # copied to GPU memory
print("   b = a.to('cuda') -> device", b.device)
c = b @ b                                            # matrix multiply runs ON the GPU
print("   b @ b =", c.tolist(), "(computed on", c.device, ")")
print("   Rule: every tensor in one operation must be on the same device;"
      " .to('cuda') / .cpu() move them.")

print("=" * 70)
print("3. Why GPUs: a 4096 x 4096 matrix multiply (about 137 billion float operations)")
n = 4096
x_cpu = torch.randn(n, n)
start = time.perf_counter()
_ = x_cpu @ x_cpu
cpu_s = time.perf_counter() - start
print("   CPU float32: %.3f s" % cpu_s)

x_gpu = x_cpu.to("cuda")
_ = x_gpu @ x_gpu            # warm-up: the first GPU call also sets things up
torch.cuda.synchronize()     # GPU work is asynchronous: wait for it before stopping the clock
start = time.perf_counter()
for _ in range(10):
    _ = x_gpu @ x_gpu
torch.cuda.synchronize()
gpu_s = (time.perf_counter() - start) / 10
print("   GPU float32: %.4f s  (%.0fx faster)" % (gpu_s, cpu_s / gpu_s))

x_half = x_gpu.half()        # float16: half the memory, uses the GPU's tensor cores
_ = x_half @ x_half
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    _ = x_half @ x_half
torch.cuda.synchronize()
half_s = (time.perf_counter() - start) / 10
print("   GPU float16: %.4f s  (%.0fx faster than CPU) - the Jetson runs models in FP16 too"
      % (half_s, cpu_s / half_s))

print("=" * 70)
print("4. Autograd: gradients for free")
w = torch.tensor(3.0, requires_grad=True)   # a 'weight' PyTorch should track
x = torch.tensor(2.0)
loss = (w * x - 10.0) ** 2                  # how wrong is w*x compared with a target of 10?
loss.backward()                             # PyTorch works out d(loss)/d(w) automatically
print("   loss = (w*x - 10)^2 with w=3, x=2 -> loss = %.1f" % loss.item())
print("   d(loss)/d(w) = %.1f   (by hand: 2*(w*x-10)*x = 2*(6-10)*2 = -16)" % w.grad.item())
print("   Training = repeat: compute loss, backward(), nudge weights against the gradient.")
print("=" * 70)
print("Step 1 done. Next: step 2 loads the Hugging Face fall model (edge/pc/README.md).")
