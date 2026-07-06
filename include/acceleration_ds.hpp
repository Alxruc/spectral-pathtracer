#ifndef ACCELERATION_DS_HPP
#define ACCELERATION_DS_HPP

#include <hip/hip_runtime.h>
#include <gpu_ds.hpp>
#include <vector>
#include <geometry.hpp>
#include "materials.hpp"

struct AABB {
    Vec3 aabb_min;
    Vec3 aabb_max;
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
    bool isPermutation(); //will remove later just for testing build

    __host__ const BVHNode* getRoot() const { return &nodes[0]; }
private:
    uint32_t nodes_used = 0;
    std::vector<Vec3> centroids;

    void update_node_bounds(uint32_t node_i);
    void subdivide(uint32_t node_i);
};

struct DeviceBVH {
    BVHNode* nodes;
    uint32_t* tri_idx;
    Triangle* tris;
    Material mat = make_dielectric();

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
