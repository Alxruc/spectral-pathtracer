#ifndef INTERSECT_HPP
#define INTERSECT_HPP

#include <hip/hip_runtime.h>
#include <geometry.hpp>
#include <constants.hpp>
#include <acceleration_ds.hpp>

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

// we assume only leafs get sent here
__device__ bool inline rayLeafIntersect(const Ray& r, const DeviceBVH& bvh, const BVHNode* node, float& tOut) {
    float closestT = 1e20;
    float hit = false;
    for(uint32_t i = 0; i < node->tri_count; i++) {
        float hitT;
        uint32_t global_tri_idx = bvh.tri_idx[node->left_first + i];
        const Triangle& tri = bvh.tris[global_tri_idx];
        if (rayTriangleIntersect(tri, r, hitT)) {
            if(hitT < closestT) {
                closestT = hitT;
            }
            hit = true;
        }
    }

    if(hit) {
        tOut = closestT;
    }
    return hit;
}

__device__ bool inline rayAABBIntersect(const Ray& r, const AABB& box, float& tOut) {
    const Vec3 tmin = {
        (box.aabb_min.x - r.origin.x) / r.dir.x,
        (box.aabb_min.y - r.origin.y) / r.dir.y,
        (box.aabb_min.z - r.origin.z) / r.dir.z,
    };
    const Vec3 tmax = {
        (box.aabb_max.x - r.origin.x) / r.dir.x,
        (box.aabb_max.y - r.origin.y) / r.dir.y,
        (box.aabb_max.z - r.origin.z) / r.dir.z,
    };

    const float tnear = fmaxf(
        fmaxf(fminf(tmin.x, tmax.x), fminf(tmin.y, tmax.y)),
        fminf(tmin.z, tmax.z)
    );

    const float tfar = fminf(
        fminf(fmaxf(tmin.x, tmax.x), fmaxf(tmin.y, tmax.y)),
        fmaxf(tmin.z, tmax.z)
    );

    if (tnear <= tfar) {
        tOut = tnear;
        return true;
    }
    return false;
}

// per sphere: accept the NEAR root if it's inside (tmin, tmax);
// otherwise try the FAR root
__device__ inline bool hitSphereT(const Sphere& s, const Ray& r,
                           float tmin, float tmax, float& tOut)
{
    Vec3 oc = r.origin - s.center;
    float a = dot(r.dir, r.dir);
    float b = 2.0f * dot(oc, r.dir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return false;
    float sq = sqrtf(disc);
    float t = (-b - sq) / (2.0f * a);
    if (t < tmin || t > tmax) {
        t = (-b + sq) / (2.0f * a);
        if (t < tmin || t > tmax) return false;
    }
    tOut = t; return true;
}


__device__ inline bool rayBVHIntersect(const Ray &r, const DeviceBVH &bvh, float &tOut) {
    const BVHNode* stack[MAX_PTR_COUNT];
    const BVHNode** stackPtr = stack;
    *stackPtr++ = NULL;

    const BVHNode* root = bvh.getRoot();
    if(!rayAABBIntersect(r, root->bounds, tOut)) {
        return false; // we dont even hit the outermost bounding box
    }

    bool hit = false;
    float closestT = 1e20;

    const BVHNode* node = root;

    while (node != NULL) {
        if(node->is_leaf()) {
            float hitT;
            if(rayLeafIntersect(r, bvh, node, hitT)) {
                if(hitT < closestT) {
                    closestT = hitT;
                    hit = true;
                }
            }
        }
        else {
            // it's an internal node, test children
            const BVHNode* leftChild = bvh.getLeftChild(node);
            const BVHNode* rightChild = bvh.getRightChild(node);

            float tLeft, tRight;
            bool hitLeft = rayAABBIntersect(r, leftChild->bounds, tLeft) && tLeft < closestT;
            bool hitRight = rayAABBIntersect(r, rightChild->bounds, tRight) && tRight < closestT;

            if (hitLeft && hitRight) {
                if (tLeft < tRight) {
                    // push the closer one so it gets processed first
                    // further ones then get thrown away in the lines above
                    *stackPtr++ = rightChild;
                    *stackPtr++ = leftChild;
                } else {
                    *stackPtr++ = leftChild;
                    *stackPtr++ = rightChild;
                }
            } else if (hitLeft) {
                *stackPtr++ = leftChild;
            } else if (hitRight) {
                *stackPtr++ = rightChild;
            }
        }
        node = *--stackPtr; // pop
    }

    if(hit) {
        tOut = closestT;
    }
    return hit;
}

#endif
