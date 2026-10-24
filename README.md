# spectral-pathtracer
A spectral path tracer targeting AMD GPUs via HIP/ROCm.

*WIP CURRENT PROGRESS:*

<img src="out.png" width="400">

Be wary, as the code is currently quite ugly. 

## Requirements

- CMake >= 3.21
- AMD (default): ROCm 6.x with HIP (`hipcc` / `amdclang++`)
- NVIDIA: CUDA toolkit, plus HIP headers for the `nvidia` platform
- HIP can also be compiled to run on the CPU using the [HIP-CPU Runtime](https://github.com/ROCm/HIP-CPU)

## Build

On AMD the GPU architecture is auto-detected, no flags needed:
(except on WSL there you need to add it manually, see below)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

On CPUs 
```bash
cmake -S . -B build_cpu \
  -DGPU_RUNTIME=CPU \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/hip-cpu/install"

cmake --build build_cpu -j
```

To target a specific card or backend, pass the arguments:
(NVIDIA is untested)

```bash
# A different AMD card (replace gfx1201 with your card, see your arch with: rocminfo | grep gfx)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1201
```

```bash
# NVIDIA: set the backend and compute capability (e.g. 120 = consumer Blackwell)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_RUNTIME=CUDA -DGPU_TARGETS=120
```

A Debug build can be slower since HIP's vector-type math isn't as aggressively inlined/unrolled at `-O`/`-O1`.

## Run

```bash
./build/spectral_pathtracer
```

## 3D Models
[Glass Model](https://sketchfab.com/3d-models/glass-4638689dc697465eace0374125e5855e)
