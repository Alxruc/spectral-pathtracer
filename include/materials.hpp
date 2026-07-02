#ifndef MATERIALS_HPP
#define MATERIALS_HPP

#include <hip/hip_runtime.h>

enum class MatType { Lambertian, Metal, Dielectric };
enum class MetalType { Chrome, Gold }; // only chrome implemented right now

struct MetalProps {
    float roughness;
};

struct LambertianProps {
    float albedo;
};

struct DielectricProps {
    float albedo;
};

struct Material {
    MatType type;
    union {
        LambertianProps lambertian;
        MetalProps      metal;
        DielectricProps dielectric;
    };
};

__host__ __device__ inline Material make_lambertian(float albedo) {
    return Material{
            .type = MatType::Lambertian,
            .lambertian = { .albedo = albedo }
        };
}

__host__ __device__ inline Material make_metal(float roughness) {
    return Material{
        .type = MatType::Metal,
        .metal = { .roughness = roughness }
    };
}

__host__ __device__ inline Material make_dielectric() {
    return Material{
        .type = MatType::Dielectric,
        .dielectric = { .albedo = 1.0f }
    };
}

/*
 * To not have to deal with a large table or loading CSVs polynomial (cubic)
 * regression was performed on refractive index and extinction coefficient data
 * so that we can get a fast approximation of it. Data source will be mentioned in
 * the respective case
 */
__device__ float getRefractiveIndex(MetalType metal, float lamda_nm) {
    // assumes wavelength in nm
    float lamda_mu = lamda_nm / 1000;
    switch (metal) {
        case MetalType::Chrome:
            // Johnson and Christy 1974L 0.188-1.937 micrometers mapped to a cubic polynomial
            return -0.074 + 7.308 * lamda_mu - 4.678 * powf(lamda_mu, 2.0) + 0.985 * powf(lamda_mu, 3.0);
            break;
        case MetalType::Gold:
            break;
        default:
            __builtin_unreachable();
            break;
    }
    return 0.0;
}

// see comments in getRefractiveIndex
__device__ float getExtinctionIndex(MetalType metal, float lamda_nm) {
    float lamda_mu = lamda_nm / 1000;
    switch(metal) {
        case MetalType::Chrome:
            // Johnson and Christy 1974L 0.188-1.937 micrometers mapped to a cubic polynomial
            return -0.026 + 10.32 * lamda_mu - 9.494 * powf(lamda_mu, 2.0) + 2.871 * powf(lamda_mu, 3.0);
            break;
        case MetalType::Gold:
            break;
        default:
            __builtin_unreachable();
            break;
    }
    return 0.0;
}

#endif
