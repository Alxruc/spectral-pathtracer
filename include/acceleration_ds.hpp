#ifndef ACCELERATION_DS_HPP
#define ACCELERATION_DS_HPP

#include <hip/hip_runtime.h>
#include <gpu_ds.hpp>
#include <vector>
#include <geometry.hpp>
#include "materials.hpp"

const uint32_t INTERVAL_NUMBER = 8;

struct AABB {
    float3 aabb_min{ 1e20f,  1e20f,  1e20f};
    float3 aabb_max{-1e20f, -1e20f, -1e20f};
    void reset() {
        aabb_min = float3{ 1e20f,  1e20f,  1e20f};
        aabb_max = float3{-1e20f, -1e20f, -1e20f};
    }
    void grow(const float3& p) { aabb_min = min(aabb_min, p), aabb_max = max(aabb_max, p); }
    void grow(const AABB& other) {
        aabb_min = min(aabb_min, other.aabb_min);
        aabb_max = max(aabb_max, other.aabb_max);
    }
    void grow(const Triangle& t) { grow(t.a); grow(t.b); grow(t.c); }
    float area() {
        float3 e = aabb_max - aabb_min;
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
    std::vector<float3> centroids;

    float findBestSplitPos(const struct BVHNode& node, float& splitPos, int& bestAxis);
    float calculateNodeCost(const struct BVHNode& node);
    void update_node_bounds(uint32_t node_i);
    void subdivide(uint32_t node_i);
};

struct DeviceBVH {
    BVHNode* nodes;
    uint32_t* tri_idx;
    Triangle* tris;
    Material lambertian = make_lambertian(0.7);
    Material glass = make_dielectric();
    Material metal = make_metal(MetalType::Chrome, 0.1);

    // for mapping the triangles to the objects they belong to
    const uint32_t* object_boundaries = nullptr;
    uint32_t num_objects = 0;

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

    __device__ uint32_t getObjectIdx(uint32_t triangle_index) const {
        if (num_objects <= 1) return 0;

        // Binary search to find which range triangle_index belongs to
        int low = 0;
        int high = num_objects - 2; // Last valid start index
        uint32_t obj_id = 0;

        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (object_boundaries[mid + 1] > triangle_index) {
                obj_id = mid;
                high = mid - 1; // Check lower ranges
            } else {
                low = mid + 1;  // Check upper ranges
            }
        }
        return obj_id;
    }
};

// for automatically freeing the memory again because I'll probably forget it
class GPUBVHManager {
public:
    DeviceBVH bvh = {nullptr, nullptr, nullptr};
    uint32_t node_count = 0;
    uint32_t tri_count = 0;

    // Device allocation for the boundaries
    uint32_t* d_object_boundaries = nullptr;

    // uploads data from CPU BVH to GPU VRAM
    void upload(const BVH& cpu_bvh, const std::vector<uint32_t>& host_boundaries) {
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

        bvh.num_objects = static_cast<uint32_t>(host_boundaries.size());
        if(bvh.num_objects > 0) {
            HIP_CHECK(hipMalloc(&d_object_boundaries, bvh.num_objects * sizeof(uint32_t)));
            HIP_CHECK(hipMemcpy(d_object_boundaries, host_boundaries.data(), 
                       bvh.num_objects * sizeof(uint32_t), hipMemcpyHostToDevice));
            
            bvh.object_boundaries = d_object_boundaries;
        }
    }

    // clean up GPU memory
    void free() {
        if (bvh.nodes)   { HIP_CHECK(hipFree(bvh.nodes));   bvh.nodes = nullptr; }
        if (bvh.tri_idx) { HIP_CHECK(hipFree(bvh.tri_idx)); bvh.tri_idx = nullptr; }
        if (bvh.tris)    { HIP_CHECK(hipFree(bvh.tris));    bvh.tris = nullptr; }
        if (d_object_boundaries) {  HIP_CHECK(hipFree(d_object_boundaries)) ; }
    }

    ~GPUBVHManager() {
        free();
    }
};

#endif
