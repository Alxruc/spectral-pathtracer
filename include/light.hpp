#include <hip/hip_runtime.h>

#include <materials.hpp>
#include <acceleration_ds.hpp>
#include <gpu_ds.hpp>
#include <constants.hpp>

enum class LightType { Directional, Area};

struct AreaLight {
    float3 p;
    float3 n;
    const Material* material;
};

struct DirLight {
    float3 direction;
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

__device__ DirLight sampleSunDisk(RNG& rng)
{
    // Two uniform random numbers
    float u1 = rng.next();
    float u2 = rng.next();

    // Uniformly sample cos(theta) between cos(sun radius) and 1.
    // Uniform in cos_theta <=> uniform over solid angle of the cone.
    float cos_theta = 1.0f - u1 * (1.0f - COS_SUN_RADIUS);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = TWO_PI * u2;

    // Direction in local frame where +z = SUN_DIR
    float x = sin_theta * cosf(phi);
    float y = sin_theta * sinf(phi);
    float z = cos_theta;

    // Build an orthonormal basis around SUN_DIR
    // SUN_DIR is assumed to be normalized here
    float3 u, v;
    tangentFrame(SUN_DIR, u, v);

    return {.direction = normalize(x * u + y * v + z * SUN_DIR)};
}

__device__ Light randomLight(RNG& rng, int number_of_lights, const DeviceBVH& bvh) {
    // we assume there exists at most only one Directional light per scene
    // which is the sun (if it exists)
    uint32_t random = rng.nextBounded(number_of_lights);

    if(random == 0 && SUN_ENABLED) {
        DirLight directional = sampleSunDisk(rng);
        return {.kind = LightType::Directional, .directional = directional, .pdf = ONE_OVER_OMEGA_SUN };
    }
    AreaLight area = sampleLight(bvh, rng);
    return {.kind = LightType::Area, .area = area, .pdf = area.material->emissive.pdf_area};
}

__device__ float blackbody(float lamda, float temperature) {
    float lamda_meters = lamda * 1e-9;

    float l5 = powf(lamda_meters, 5.0);
    float exponent = COEF_C2 / (lamda_meters * temperature);

    // protect against exp() overflow directly in float
    if (exponent > 88.0f) { // e^88 is near the upper limit of float (~3.4e38)
        return 0.0f;
    }

    float radiance = COEF_C1 / (l5 * (expf(exponent) - 1.0f));

    return radiance;
}

__device__ float sunRadiance(float lamda) {
    return SUN_SCALE * blackbody(lamda, SUN_TEMPERATURE) / SUN_NORMALIZING;
}

__device__ float4 sunRadiance(float4 lamdas) {
    return make_float4(
        SUN_SCALE * blackbody(lamdas.x, SUN_TEMPERATURE) / SUN_NORMALIZING,
        SUN_SCALE * blackbody(lamdas.y, SUN_TEMPERATURE) / SUN_NORMALIZING,
        SUN_SCALE * blackbody(lamdas.z, SUN_TEMPERATURE) / SUN_NORMALIZING,
        SUN_SCALE * blackbody(lamdas.w, SUN_TEMPERATURE) / SUN_NORMALIZING
    );
}

__device__ float emittedRadiance(const EmissiveProps& e, float lamda) {
    float norm = e.brightness / blackbody(NORMALIZATION_WAVELENGTH_NM, e.temperature);
    return norm * blackbody(lamda, e.temperature);
}

__device__ float4 emittedRadiance(const EmissiveProps& e, float4 lamdas) {
    float norm = e.brightness / blackbody(NORMALIZATION_WAVELENGTH_NM, e.temperature);
    return make_float4(
        norm * blackbody(lamdas.x, e.temperature),
        norm * blackbody(lamdas.y, e.temperature),
        norm * blackbody(lamdas.z, e.temperature),
        norm * blackbody(lamdas.w, e.temperature)
    );
}

__device__ float4 lightRadiance(Light light, float4 lamdas) {
    if (light.kind == LightType::Directional) {
        return sunRadiance(lamdas);
    }
    return emittedRadiance(light.area.material->emissive, lamdas);
}

__device__ float lightRadiance(Light light, float lamda) {
    if (light.kind == LightType::Directional) {
        return sunRadiance(lamda);
    }
    return emittedRadiance(light.area.material->emissive, lamda);
}

// direction from x towards the light, fixed for the sun, depends on x for area lights
__device__ float3 lightDirFrom(const Light& light, float3 x) {
    if (light.kind == LightType::Directional) {
        return light.directional.direction;
    }
    return normalize(light.area.p - x);
}

__device__ void shiftLight(const Light& light, Light& l_u, Light& l_v, float delta) {
    float3 u,v;

    if (light.kind == LightType::Area) {
        tangentFrame(light.area.n, u,v);
        AreaLight area_u = {.p = light.area.p + delta * u, .n = light.area.n, .material = light.area.material};
        AreaLight area_v = {.p = light.area.p + delta * v, .n = light.area.n, .material = light.area.material};
        l_u = {.kind = LightType::Area, .area = area_u, .pdf = light.pdf};
        l_v = {.kind = LightType::Area, .area = area_v, .pdf = light.pdf};
        return;
    }

    tangentFrame(light.directional.direction, u,v);
    DirLight dir_u = {.direction =  light.directional.direction + delta * u}; 
    DirLight dir_v = {.direction =  light.directional.direction + delta * v}; 
    l_u = {.kind = LightType::Directional, .directional = dir_u, .pdf = light.pdf};
    l_v = {.kind = LightType::Directional, .directional = dir_v, .pdf = light.pdf};
}

// if we have the sun we just need to see it unobstructed
// if we have an emissive material, then we need to reach that unobstructed
__device__ bool reachesLight(const Light& light, bool did_hit, const Hit& hit) {
    if (light.kind == LightType::Directional) {
        return !did_hit;
    }
    return did_hit && hit.obj_idx == LIGHT_OBJ_IDX;
}

// light.pdf in solid angle as seen from x, 0 when x sees the back of an area light
__device__ float lightPdfSolidAngle(const Light& light, float3 x) {
    if (light.kind == LightType::Directional) {
        return light.pdf;
    }
    float3 to_light = light.area.p - x;
    float dist_sq = length_sq(to_light);
    float cos_light = dot(light.area.n, -1.0f * to_light / sqrtf(dist_sq));
    return cos_light > 0.0f ? light.pdf * dist_sq / cos_light : 0.0f;
}
