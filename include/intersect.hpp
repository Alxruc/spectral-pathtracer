#ifndef INTERSECT_HPP
#define INTERSECT_HPP

#include <hip/hip_runtime.h>
#include <geometry.hpp>
#include <constants.hpp>

// Möller-Trumbore intersection algorithm
// P = A + u(B-A) + v(C-A) of triangle ABC
// P = o + td
// => o - A = -td + u(B-A) + v(C-A)
__device__ inline bool rayTriangleIntersect(const Triangle& tri, const Ray& r, float& tOut) {
    Vec3 v1 = tri.a;
    Vec3 v2 = tri.b;
    Vec3 v3 = tri.c;

    Vec3 e1 = v2 - v1;
    Vec3 e2 = v3- v1;

    Vec3 r_e2 = cross(r.dir, e2);
    float det = dot(e1, r_e2);

    if (fabsf(det) < EPSILON) return false; // ray is parallel to triangle

    float inv_det = 1.0 / det;
    Vec3 tri_to_ray = r.origin - tri.a;
    float u = inv_det * dot(tri_to_ray, r_e2);

    if (u < -EPSILON || u - 1 > EPSILON) return false; // ray passes outside e2

    Vec3 ttr_e1 = cross(tri_to_ray, e1);
    float v = inv_det * dot(r.dir, ttr_e1);

    if (v < -EPSILON || u + v - 1 > EPSILON) return false; // ray passes outside e1

    tOut = inv_det * dot(e2, ttr_e1);
    if (tOut < EPSILON) return false;
    return true;
}

#endif
