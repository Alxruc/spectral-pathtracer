#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <hip/hip_runtime.h>
#include <materials.hpp>

__host__ __device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline float3 normalize(float3 a) { return a / sqrtf(dot(a, a)); }
__host__ __device__ inline float3 cross(float3 a, float3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }

inline float3 min(const float3& a, const float3& b) {
    return float3{ fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z) };
}

inline float3 max(const float3& a, const float3& b) {
    return float3{ fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z) };
}

struct Ray {
    float3 origin, dir;
    bool prev_was_diffuse = false; // when our ray first leaves the camera (eye) no NEE competes for contribution yet
};
struct Hit    { float t; float3 p; float3 n; bool front_face; const Material* mat; };

struct Sphere { float3 center; float radius; Material mat; };

struct Triangle {
    float3 a, b, c;
    float3 normal;
};


#endif
