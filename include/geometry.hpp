#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <hip/hip_runtime.h>
#include <materials.hpp>
#include <algorithm>

// a smart person would have used float3 from the start
struct Vec3 {
    float x,y,z;
    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3(const Vec3& other) : x(other.x), y(other.y), z(other.z) {}
    __host__ __device__ Vec3& operator=(const Vec3& other) {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }
    __host__ __device__ float  operator[](int i) const {
        return i == 0 ? x : (i == 1 ? y : z);
    }
};
__host__ __device__ inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; };
__host__ __device__ inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; };
__host__ __device__ inline Vec3 operator-(Vec3 a) { return {-a.x,-a.y,-a.z};};
__host__ __device__ inline Vec3 operator*(float s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; };
__host__ __device__ inline Vec3 operator/(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; };
__host__ __device__ inline float dot(Vec3 a, Vec3 b)      { return a.x * b.x + a.y * b.y + a.z * b.z; };
__host__ __device__ inline Vec3 normalize(Vec3 a)         { return (1.0f / sqrtf(dot(a, a))) * a; };
__host__ __device__ inline Vec3 cross(Vec3 a, Vec3 b)   { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; };

inline Vec3 min(const Vec3& a, const Vec3& b) {
    return Vec3{ std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) };
}

inline Vec3 max(const Vec3& a, const Vec3& b) {
    return Vec3{ std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) };
}

struct Ray { Vec3 origin, dir; };
struct Hit    { float t; Vec3 p; Vec3 n; bool front_face; const Material* mat; };

struct Sphere { Vec3 center; float radius; Material mat; };

struct Triangle {
    Vec3 a, b, c;
    Vec3 normal;
};


#endif
