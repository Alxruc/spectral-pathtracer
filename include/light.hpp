#include <hip/hip_runtime.h>

#include <materials.hpp>
#include <acceleration_ds.hpp>
#include <gpu_ds.hpp>

enum class LightType { Directional, Area};

struct AreaLight {
    float3 p;
    float3 n;
    const Material* material;
};

struct DirLight {
    const float3 direction;
};

struct Light {
    LightType kind;
    union {
        DirLight directional;
        AreaLight area;
    };
    float pdf;
};



// uniform point on the emissive mesh, so its area density is 1 / area (= emissive.pdf_area)
// picking the triangle uniformly is only area-uniform because both halves of the square are equal
__device__ AreaLight sampleLight(const DeviceBVH& bvh, RNG& rng)
{
    uint32_t first = bvh.object_boundaries[LIGHT_OBJ_IDX];
    uint32_t count = bvh.object_boundaries[LIGHT_OBJ_IDX + 1] - first;
    uint32_t i = min(uint32_t(rng.next() * count), count - 1);
    Triangle tri = bvh.tris[first + i];

    // sqrt warps the barycentrics so points don't bunch up at vertex a
    float su = sqrtf(rng.next());
    float v = rng.next();
    float3 p = (1.0f - su) * tri.a + (su * (1.0f - v)) * tri.b + (su * v) * tri.c;
    return { p, tri.normal, &bvh.emissive };
}