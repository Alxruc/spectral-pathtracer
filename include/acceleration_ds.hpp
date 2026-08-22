#ifndef ACCELERATION_DS_HPP
#define ACCELERATION_DS_HPP

#include <hip/hip_runtime.h>
#include <gpu_ds.hpp>
#include <vector>
#include <geometry.hpp>
#include "materials.hpp"

const uint32_t INTERVAL_NUMBER = 8;

struct AABB {
    Vec3 aabb_min{ 1e20f,  1e20f,  1e20f};
    Vec3 aabb_max{-1e20f, -1e20f, -1e20f};
    void reset() {
        aabb_min = Vec3{ 1e20f,  1e20f,  1e20f};
        aabb_max = Vec3{-1e20f, -1e20f, -1e20f};
    }
    void grow(const Vec3& p) { aabb_min = min(aabb_min, p), aabb_max = max(aabb_max, p); }
    void grow(const AABB& other) {
        aabb_min = min(aabb_min, other.aabb_min);
        aabb_max = max(aabb_max, other.aabb_max);
    }
    void grow(const Triangle& t) { grow(t.a); grow(t.b); grow(t.c); }
    float area() {
        Vec3 e = aabb_max - aabb_min;
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
};

struct Bin {
    AABB bounds;
    int tri_count = 0;
};

struct alignas(16) BVHNode {
    AABB bounds;
    uint32_t left_first;
    uint32_t tri_count;

    __host__ __device__ bool is_leaf() const { return tri_count > 0; }
};

class BVH {
public:
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> tri_idx;
    std::vector<Triangle> tris;

    void build();

    __host__ const BVHNode* getRoot() const { return &nodes[0]; }
private:
    uint32_t nodes_used = 0;
    std::vector<Vec3> centroids;

    float findBestSplitPos(const struct BVHNode& node, float& splitPos, int& bestAxis);
    float calculateNodeCost(const struct BVHNode& node);
    void update_node_bounds(uint32_t node_i);
    void subdivide(uint32_t node_i);
};

struct DeviceBVH {
    BVHNode* nodes;
    uint32_t* tri_idx;
    Triangle* tris;
    Material mat = make_lambertian(0.8);

    __device__ const BVHNode* getRoot() const {
        return &nodes[0];
    }

    __device__ const BVHNode* getLeftChild(const BVHNode* node) const {
        return node->is_leaf() ? nullptr : &nodes[node->left_first];
    }

    __device__ const BVHNode* getRightChild(const BVHNode* node) const {
        return node->is_leaf() ? nullptr : &nodes[node->left_first + 1];
    }

    __device__ const AABB& getAABB(const BVHNode* node) const {
        return node->bounds;
    }
};

// for automatically freeing the memory again because I'll probably forget it
class GPUBVHManager {
public:
    DeviceBVH bvh = {nullptr, nullptr, nullptr};
    uint32_t node_count = 0;
    uint32_t tri_count = 0;

    // uploads data from CPU BVH to GPU VRAM
    void upload(const BVH& cpu_bvh) {
        // free any existing allocations just in case
        free();

        node_count = cpu_bvh.nodes.size();
        tri_count = cpu_bvh.tris.size();

        size_t nodes_bytes = node_count * sizeof(BVHNode);
        HIP_CHECK(hipMalloc(&bvh.nodes, nodes_bytes));
        HIP_CHECK(hipMemcpy(bvh.nodes, cpu_bvh.nodes.data(), nodes_bytes, hipMemcpyHostToDevice));

        size_t tri_idx_bytes = cpu_bvh.tri_idx.size() * sizeof(uint32_t);
        HIP_CHECK(hipMalloc(&bvh.tri_idx, tri_idx_bytes));
        HIP_CHECK(hipMemcpy(bvh.tri_idx, cpu_bvh.tri_idx.data(), tri_idx_bytes, hipMemcpyHostToDevice));

        size_t tris_bytes = tri_count * sizeof(Triangle);
        HIP_CHECK(hipMalloc(&bvh.tris, tris_bytes));
        HIP_CHECK(hipMemcpy(bvh.tris, cpu_bvh.tris.data(), tris_bytes, hipMemcpyHostToDevice));
    }

    // clean up GPU memory
    void free() {
        if (bvh.nodes)   { HIP_CHECK(hipFree(bvh.nodes));   bvh.nodes = nullptr; }
        if (bvh.tri_idx) { HIP_CHECK(hipFree(bvh.tri_idx)); bvh.tri_idx = nullptr; }
        if (bvh.tris)    { HIP_CHECK(hipFree(bvh.tris));    bvh.tris = nullptr; }
    }

    ~GPUBVHManager() {
        free();
    }
};

#endif
