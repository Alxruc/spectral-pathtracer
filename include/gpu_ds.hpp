#ifndef GPU_DS_HPP
#define GPU_DS_HPP
#include <hip/hip_runtime.h>
#include <iostream>

struct Vec3 { float x,y,z; };
__host__ __device__ inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; };
__host__ __device__ inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ inline Vec3 operator*(float s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
__host__ __device__ inline float dot(Vec3 a, Vec3 b)      { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline Vec3 normalize(Vec3 a)         { return (1.0f / sqrtf(dot(a, a))) * a; }

struct Ray { Vec3 origin, dir; };
struct Hit    { float t; Vec3 p; Vec3 n; float albedo; };

struct Sphere { Vec3 center; float radius; float albedo; };

// Per-thread RNG. PCG step (O'Neill 2014)
struct RNG {
    uint32_t state;
    __device__ uint32_t nextU32() {
        state = state * 747796405u + 2891336453u;                        // LCG advance
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;  // permute
        return (word >> 22u) ^ word;
    }
    __device__ float next() {            // uniform float in [0, 1)
        return (nextU32() >> 8) * (1.0f / 16777216.0f);
    }
};


#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}
#endif