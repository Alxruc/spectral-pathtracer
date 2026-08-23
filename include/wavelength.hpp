#include <hip/hip_runtime.h>

#include <gpu_ds.hpp>
#include <constants.hpp>

// optics color stuff that is just pure constants read from somewhere

__device__ float linearToSRGB(float c) {
    c = fmaxf(0.0f, c); // Clamp negative values from out-of-gamut spectral colors
    if (c <= 0.0031308f) {
        return 12.92f * c;
    }
    return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

__device__ float3 convertXYZtoRGB(float X, float Y, float Z) {
    float R =  3.2404542*X - 1.5371385*Y - 0.4985314*Z;
    float G = -0.9692660*X + 1.8760108*Y + 0.0415560*Z;
    float B =  0.0556434*X - 0.2040259*Y + 1.0572252*Z;

    return make_float3(R, G, B);
}

__device__ float3 XYZ_to_sRGB(float3 xyz) {
    // cnvert to linear sRGB
    float3 lin = convertXYZtoRGB(xyz.x, xyz.y, xyz.z);

    // apply sRGB non-linear electro-optical transfer function (EOTF)
    return {
        linearToSRGB(lin.x),
        linearToSRGB(lin.y),
        linearToSRGB(lin.z)
    };
}

//

__device__ float fwrap(float x, float min, float max) {
    return min + fmodf(x, max-min);
}

__device__ float sampleWavelength(RNG& rng) {
    float lamda = (MAX_WAVELENGTH_NM - MIN_WAVELENGTH_NM) * rng.next() + MIN_WAVELENGTH_NM;
    return lamda;
}

/* 
Our "hero wavelength" will be the one responsible for the actual
path traveling and stuff. It brings along additional wavelengths
(wrapped to be in the range [lamda_min, lamda_max)) to get a less fuzzy
result quicker
*/
__device__ float4 sampleHeroWavelength(RNG& rng) {
    float hero_wavelength = sampleWavelength(rng);
    float wavelength_range = MAX_WAVELENGTH_NM - MIN_WAVELENGTH_NM;
    float support1 = fwrap(hero_wavelength + wavelength_range / 4.0f, MAX_WAVELENGTH_NM, MIN_WAVELENGTH_NM);
    float support2 = fwrap(hero_wavelength + 2 * wavelength_range / 4.0f, MAX_WAVELENGTH_NM, MIN_WAVELENGTH_NM);
    float support3 = fwrap(hero_wavelength + 3 * wavelength_range / 4.0f, MAX_WAVELENGTH_NM, MIN_WAVELENGTH_NM);
    return make_float4(hero_wavelength, support1, support2, support3); 
}