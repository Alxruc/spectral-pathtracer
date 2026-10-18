#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <hip/hip_runtime.h>
#include <materials.hpp>
#include <constants.hpp>

__host__ __device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline float3 normalize(float3 a) { return a / sqrtf(dot(a, a)); }
__host__ __device__ inline float3 cross(float3 a, float3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }

// two unit vectors s, t perpendicular to the unit vector n and to each other
__host__ __device__ inline void tangentFrame(float3 n, float3& s, float3& t) {
    // we simply need a helper that is not parallel to our normal
    float3 a = fabsf(n.x) > 0.9f ? float3(0, 1, 0) : float3(1, 0, 0);
    s = normalize(cross(a, n));
    t = cross(n, s);
}

inline float3 min(const float3& a, const float3& b) {
    return float3{ fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z) };
}

inline float3 max(const float3& a, const float3& b) {
    return float3{ fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z) };
}

struct Ray {
    float3 origin, dir;
    bool prev_was_diffuse = false; // when our ray first leaves the camera (eye) no NEE competes for contribution yet
};

// naive index were our BVH is just -1, breaks as soon as we have more stuff but fine for prototype
struct Hit    { float t; float3 p; float3 n; bool front_face; const Material* mat; int obj_idx; };

struct Sphere { float3 center; float radius; Material mat; };

__host__ __device__ inline float3 sphereNormal(const Sphere& s, float3 p) { return (p - s.center) / s.radius; }

// closest point on the sphere: the manifold walk's "push back onto the surface" step
__host__ __device__ inline float3 projectOntoSphere(const Sphere& s, float3 p) { return s.center + s.radius * normalize(p - s.center); }


// gaussian solver for the manifold next event estimation
__device__ bool solve4(const float A[MNEE_DIM][MNEE_DIM], const float b[MNEE_DIM], int m, float x[MNEE_DIM])
{
    // U = A, on copies so the caller's J and right-hand side stay intact
    float U[MNEE_DIM][MNEE_DIM];
    float y[MNEE_DIM];
    for (int i = 0; i < m; i++) {
        y[i] = b[i];
        for (int j = 0; j < m; j++) U[i][j] = A[i][j];
    }

    for (int k = 0; k < m - 1; k++) {
        // select the row i >= k that maximizes |u_ik|
        int p = k;
        for (int i = k + 1; i < m; i++)
            if (fabsf(U[i][k]) > fabsf(U[p][k])) p = i;

        if (fabsf(U[p][k]) < 1e-12f) return false;

        // interchange rows k and p of U (columns k..m-1) and of y
        for (int j = k; j < m; j++) {
            float tmp = U[k][j]; U[k][j] = U[p][j]; U[p][j] = tmp;
        }
        float tmp = y[k]; y[k] = y[p]; y[p] = tmp;

        // eliminate below the pivot: row_j -= l_jk * row_k
        for (int j = k + 1; j < m; j++) {
            float l_jk = U[j][k] / U[k][k];
            for (int c = k; c < m; c++) U[j][c] -= l_jk * U[k][c];
            y[j] -= l_jk * y[k];
        }
    }
    if (fabsf(U[m - 1][m - 1]) < 1e-12f) return false;

    // back substitution: x_j = (y_j - sum_{k > j} u_jk x_k) / u_jj, from the last row up
    for (int j = m - 1; j >= 0; j--) {
        float sum = y[j];
        for (int k = j + 1; k < m; k++) sum -= U[j][k] * x[k];
        x[j] = sum / U[j][j];
    }
    return true;
}

struct Triangle {
    float3 a, b, c;
    float3 normal;
};


#endif
