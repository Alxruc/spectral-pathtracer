#ifndef INTERSECT_HPP
#define INTERSECT_HPP

#include <hip/hip_runtime.h>
#include <geometry.hpp>
#include <constants.hpp>
#include <acceleration_ds.hpp>

// Determinant epsilon guards against division by zero/NaNs
const float EPSILON_DET  = 1e-6f;
// Barycentric epsilon guards against cracks/gaps at creases
const float EPSILON_BARY = 1e-5f;
// Distance epsilon guards against ray acne
const float EPSILON_DIST = 1e-4f;

struct StackNode {
    const BVHNode* node;
    float t;
};

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

    if (fabsf(det) < EPSILON_DET) return false; // ray is parallel to triangle

    float inv_det = 1.0f / det;
    Vec3 tri_to_ray = r.origin - tri.a;
    float u = inv_det * dot(tri_to_ray, r_e2);

    if (u < -EPSILON_BARY || u - 1 > EPSILON_BARY) return false; // ray passes outside e2

    Vec3 ttr_e1 = cross(tri_to_ray, e1);
    float v = inv_det * dot(r.dir, ttr_e1);

    if (v < -EPSILON_BARY || u + v - 1 > EPSILON_BARY) return false; // ray passes outside e1

    tOut = inv_det * dot(e2, ttr_e1);
    if (tOut < EPSILON_DIST) return false;
    return true;
}

// we assume only leafs get sent here
__device__ bool inline rayLeafIntersect(const Ray& r, const DeviceBVH& bvh, const BVHNode* node, float globalClosestT, float& tOut, Vec3& nOut) {
    float closestT = globalClosestT;
    float hit = false;
    Vec3 closestN;
    for(uint32_t i = 0; i < node->tri_count; i++) {
        float hitT;
        uint32_t global_tri_idx = bvh.tri_idx[node->left_first + i];
        const Triangle& tri = bvh.tris[global_tri_idx];
        if (rayTriangleIntersect(tri, r, hitT)) {
            if(hitT < closestT) {
                closestT = hitT;
                closestN = tri.normal;
            }
            hit = true;
        }
    }

    if(hit) {
        tOut = closestT;
        nOut = closestN;
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

    if (tnear <= tfar && tfar >= 0.0f) {
        tOut = tnear; // Clamp to 0 if ray is inside the box
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


__device__ inline bool rayBVHIntersect(const Ray &r, const DeviceBVH &bvh, float &tOut, Vec3& nOut) {
    StackNode stack[MAX_PTR_COUNT];
    StackNode* stackPtr = stack;

    const BVHNode* root = bvh.getRoot();
    float tRoot;
    if(!rayAABBIntersect(r, root->bounds, tRoot)) {
        return false; // we dont even hit the outermost bounding box
    }

    bool hit = false;
    float closestT = 1e20;
    Vec3 closestN;
    const BVHNode* node = root;

    while (node != NULL) {
        if(node->is_leaf()) {
            float hitT;
            Vec3 hitN;
            if(rayLeafIntersect(r, bvh, node, closestT, hitT, hitN)) {
                if(hitT < closestT) {
                    closestT = hitT;
                    closestN = hitN;
                    hit = true;
                }
            }
            node = nullptr; // finished this leaf force a pop
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
                    // push the further child onto the stack with its intersection distance
                    // immediately descend into the closer child
                    *stackPtr++ = StackNode{ rightChild, tRight };
                    node = leftChild;
                } else {
                    *stackPtr++ = StackNode{ leftChild, tLeft };
                    node = rightChild;
                }
            } else if (hitLeft) {
                node = leftChild;
            } else if (hitRight) {
                node = rightChild;
            } else {
                node = nullptr;
            }
        }
        // stack pruning
        while (node == nullptr && stackPtr > stack) {
            StackNode popped = *--stackPtr;
            if (popped.t < closestT) {
                node = popped.node; // Found a valid node to traverse next
            }
            // if popped.t >= closestT, node remains nullptr, and we loop to pop again
        }
    }

    if(hit) {
        tOut = closestT;
        nOut = closestN;
    }
    return hit;
}

#endif
