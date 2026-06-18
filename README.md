# spectral-pathtracer

A spectral path tracer targeting AMD GPUs via HIP/ROCm.

## Requirements

- CMake >= 3.21
- AMD (default): ROCm 6.x with HIP (`hipcc` / `amdclang++`)
- NVIDIA: CUDA toolkit, plus HIP headers for the `nvidia` platform

## Build

On AMD the GPU architecture is auto-detected, no flags needed:

```bash
cmake -S . -B build
cmake --build build -j
```

To target a specific card or backend, pass the arguments:
(NVIDIA is untested)

```bash
# A different AMD card (see your arch with: rocminfo | grep gfx)
cmake -S . -B build -DGPU_TARGETS=gfx@@@@
```

```bash
# NVIDIA: set the backend and compute capability (e.g. 120 = consumer Blackwell)
cmake -S . -B build -DGPU_RUNTIME=CUDA -DGPU_TARGETS=120
```

## Run

```bash
./build/spectral_pathtracer
```
