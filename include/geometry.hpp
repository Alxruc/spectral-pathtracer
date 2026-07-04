#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <hip/hip_runtime.h>
#include <materials.hpp>

struct Vec3 { float x,y,z; };
__host__ __device__ inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; };
__host__ __device__ inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; };
__host__ __device__ inline Vec3 operator-(Vec3 a) { return {-a.x,-a.y,-a.z};};
__host__ __device__ inline Vec3 operator*(float s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; };
__host__ __device__ inline Vec3 operator/(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; };
__host__ __device__ inline float dot(Vec3 a, Vec3 b)      { return a.x * b.x + a.y * b.y + a.z * b.z; };
__host__ __device__ inline Vec3 normalize(Vec3 a)         { return (1.0f / sqrtf(dot(a, a))) * a; };
__host__ __device__ inline Vec3 cross(Vec3 a, Vec3 b)   { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; };

struct Ray { Vec3 origin, dir; };
struct Hit    { float t; Vec3 p; Vec3 n; bool front_face; const Material* mat; };

struct Sphere { Vec3 center; float radius; Material mat; };

struct Triangle { Vec3 a, b, c; };


#endif
