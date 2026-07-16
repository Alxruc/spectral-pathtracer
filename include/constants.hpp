#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// gpu
const int BLOCK_SIZE = 32;
const int SPP = 2048;
const short MAX_PTR_COUNT = 64;

// rendering
const float MIN_WAVELENGTH_NM = 360;
const float MAX_WAVELENGTH_NM = 830;
const float UNIFORM_DENSITY = 0.00212765957;
const float NORMALIZATION_WAVELENGTH_NM = 550;
const float NORMALIZATION_CONSTANT = 0.011;

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
