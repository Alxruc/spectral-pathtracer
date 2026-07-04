#ifndef GPU_DS_HPP
#define GPU_DS_HPP
#include <hip/hip_runtime.h>
#include <iostream>

// Per-thread RNG. PCG step (O'Neill 2014)
struct RNG {
    uint32_t state;
    __device__ uint32_t nextU32() {
        state = state * 747796405u + 2891336453u;                        // LCG advance
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;  // permute
        return (word >> 22u) ^ word;
    }
    __device__ float next() {            // uniform float in [0, 1)
        return (nextU32() >> 8) * (1.0f / 16777216.0f);
    }
};


#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}
#endif
