#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// gpu
const int BLOCK_SIZE = 32;
const int SPP = 1024;
const short MAX_PTR_COUNT = 64;

// rendering
const float MIN_WAVELENGTH_NM = 360;
const float MAX_WAVELENGTH_NM = 830;
constexpr float UNIFORM_DENSITY = 0.00212765957;
const float NORMALIZATION_WAVELENGTH_NM = 550;
const float NORMALIZATION_CONSTANT = 0.011;

// MNEE: most glass crossings a seed path may have (one sphere = enter + leave)
const int MAX_SPECULAR_BOUNCES = 2;
// 2 error numbers per glass vertex (sideways lean along s and t) = size of the Newton system
const int MNEE_DIM = 2 * MAX_SPECULAR_BOUNCES;
const int MNEE_MAX_ITERATIONS = 15; // paper Table 1
const float MNEE_TOLERANCE = 1e-4f; // all errors below this = Snell holds

// sun
// (5, 5, 2) normalized
__device__ const float3 SUN_DIR = {0.680414f, 0.680414f, 0.272166f};
__device__ const float SUN_TEMPERATURE = 5778; // K

// for our sun to be brighter than the sky
// number pulled out from thin air for now just based on vibes
// to make it look semi right
__device__ const float SUN_SCALE = 2e6;

// blackbody(550nm, 5778) precomputed
__device__ const float SUN_NORMALIZING = 2.586 * 10e13;

// explained in light sampling function where they are used
__device__ const float COS_SUN_RADIUS = 0.9999893;
__device__ const float ONE_OVER_OMEGA_SUN = 14'900;

// physics
constexpr double PLANCKS = 6.62607015e-34;  // Planck constant (J*s)
constexpr double SPEED_OF_LIGHT = 299792458.0;     // Speed of light (m/s)
constexpr double BOLTZMANN = 1.380649e-23;    // Boltzmann constant (J/K)

// since the constants are so small we risk losing precision with just floats
// and since I dont want to deal with doubles during shader calculation we precompute
// the coefficients needed where C1 is around 10^-16 which is fine
// C1 = 2 * h * c^2
constexpr float COEF_C1 = static_cast<float>(2.0 * PLANCKS * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
// C2 = (h * c) / k
constexpr float COEF_C2 = static_cast<float>((PLANCKS * SPEED_OF_LIGHT) / BOLTZMANN);


// math
const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;
const float EPSILON = 1e-5;

#endif
