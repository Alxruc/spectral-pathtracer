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