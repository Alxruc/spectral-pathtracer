#include <acceleration_ds.hpp>
#include <intersect.hpp>
#include <cstdint>
#include <vector>
#include <utility>
#include <unordered_map>

void BVH::build() {
    const uint32_t N = (uint32_t)tris.size();
    tri_idx.resize(N);
    centroids.resize(N);
    for (uint32_t i = 0; i < N; i++) {
        tri_idx[i] = i;
        centroids[i] = (1.0f / 3.0f) * (tris[i].a + tris[i].b + tris[i].c);
    }
    nodes.clear();
    // worst case
    nodes.resize(2 * N);
    nodes_used = 2;

    nodes[0].left_first = 0;
    nodes[0].tri_count  = N;
    update_node_bounds(0);
    subdivide(0);

    centroids.clear();
    centroids.shrink_to_fit(); // release the scratch memory
    nodes.resize(nodes_used);
}

void BVH::update_node_bounds(uint32_t node_id) {
    BVHNode& node = nodes[node_id];
    node.bounds.aabb_min = Vec3{1e20f, 1e20f, 1e20f};
    node.bounds.aabb_max = Vec3{-1e20f, -1e20f, -1e20f};
    for (uint32_t i = 0; i < node.tri_count; i++) {
        const Triangle& t = tris[tri_idx[node.left_first + i]];
        node.bounds.aabb_min = min(node.bounds.aabb_min, min(t.a, min(t.b, t.c)));
        node.bounds.aabb_max = max(node.bounds.aabb_max, max(t.a, max(t.b, t.c)));
    }
}

void BVH::subdivide(uint32_t node_id) {
    BVHNode& node = nodes[node_id];
    if (node.tri_count <= 2) return;

    Vec3 extent = node.bounds.aabb_max - node.bounds.aabb_min;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float split_pos = node.bounds.aabb_min[axis] + extent[axis] * 0.5f;

    uint32_t i = node.left_first;
    uint32_t j = node.left_first + node.tri_count - 1;

    while (i <= j) {
        Vec3 c = centroids[tri_idx[i]];
        if (c[axis] < split_pos) {
            i++;
        }
        else {
            std::swap(tri_idx[i], tri_idx[j]);
            j--;
        }
    }

    // check if we accidentally put every triangle on one side
    uint32_t left_count = i - node.left_first;
    if (left_count == 0 || left_count == node.tri_count) return; // stays a leaf

    uint32_t left_i = nodes_used;
    nodes_used += 2; // two children created
    nodes[left_i].left_first = node.left_first;
    nodes[left_i].tri_count = left_count;
    nodes[left_i + 1].left_first = i;
    nodes[left_i + 1].tri_count = node.tri_count - left_count;

    // this node is no longer a leaf
    node.left_first = left_i;
    node.tri_count = 0;
    update_node_bounds(left_i);
    update_node_bounds(left_i + 1);
    subdivide(left_i);
    subdivide(left_i + 1);
}


bool isPermutation(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() != b.size()) return false;
    std::unordered_map<uint32_t, uint32_t> counts;
    for(auto i : a) {
        counts[i]++;
    }
    for(auto i : b) {
        if (counts[i] == 0) return false;
        counts[i]--;
    }

    return true;
}

// returns true if valid BVH (for testing)
bool BVH::isPermutation() {
    // A permutation of 0 to N-1 must have exactly N unique elements,
    // and every element must be strictly less than N.
    std::vector<bool> seen(tri_idx.size(), false);

    for (uint32_t idx : tri_idx) {
        // If an index is out of bounds, or we've seen it before, it's invalid
        if (idx >= tri_idx.size() || seen[idx]) {
            return false;
        }
        seen[idx] = true;
    }

    return true;
}
