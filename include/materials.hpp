#ifndef MATERIALS_HPP
#define MATERIALS_HPP

#include <hip/hip_runtime.h>

enum class MatType { Lambertian, Metal, Dielectric };
enum class MetalType { Chrome, Gold };

struct MetalProps {
    MetalType metal;
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

__host__ __device__ inline Material make_metal(MetalType metal, float roughness) {
    return Material{
        .type = MatType::Metal,
        .metal = { .metal = metal, .roughness = roughness }
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
            // because of gold's weird curve I split it into two different polynomials
            // split at 500nm; index and coefficient by Johson and Christy 1972
            return lamda_mu < 0.5
                ? 2.335 + -13.68 * lamda_mu + 56.12 * powf(lamda_mu, 2.0) - 67.86 * powf(lamda_mu, 3.0)
                : 2.309 - 5.697 * lamda_mu + 4.598 * powf(lamda_mu, 2.0) - 1.052 * powf(lamda_mu, 3.0);
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
            // Johnson and Christy 1974L mapped to two seperate cubic polynomials split at 500nmn
            return lamda_mu < 0.5
                ? -2.984 + 35.08 * lamda_mu - 81.56 * powf(lamda_mu, 2.0) + 61.45 * powf(lamda_mu, 3.0)
                : -4.599 + 16.00 * lamda_mu - 6.453 * powf(lamda_mu, 2.0) + 1.601 * powf(lamda_mu, 3.0);
            break;
        default:
            __builtin_unreachable();
            break;
    }
    return 0.0;
}

#endif
