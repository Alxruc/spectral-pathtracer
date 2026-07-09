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
    node.bounds.reset();
    for (uint32_t i = 0; i < node.tri_count; i++) {
        const Triangle& t = tris[tri_idx[node.left_first + i]];
        node.bounds.grow(t);
    }
}

float BVH::calculateNodeCost(const BVHNode& node) {
    Vec3 e = node.bounds.aabb_max - node.bounds.aabb_min;
    float surfaceArea = e.x * e.y + e.y * e.z + e.z * e.x;
    return node.tri_count * surfaceArea;
}

// with help from jbikker article
float BVH::findBestSplitPos(const BVHNode& node, float& splitPos, int& bestAxis) {
    float bestCost = 1e30f;
    for(int axis = 0; axis < 3; axis++) {
        float boundsMin = 1e30f;
        float boundsMax = -1e30f;
        for(uint32_t i = 0; i < node.tri_count; i++) {
            uint32_t currTriIdx = tri_idx[node.left_first + i];
            boundsMin = fminf(boundsMin, centroids[currTriIdx][axis]);
            boundsMax = fmaxf(boundsMax, centroids[currTriIdx][axis]);
        }
        if(boundsMin == boundsMax) continue;

        Bin bin[INTERVAL_NUMBER];
        float scale = INTERVAL_NUMBER / (boundsMax - boundsMin);
        for(uint32_t i = 0; i < node.tri_count; i++) {
            uint32_t currTriIdx = tri_idx[node.left_first + i];
            uint32_t binIdx = (uint32_t)fminf(INTERVAL_NUMBER - 1,
                (int)((centroids[currTriIdx][axis] - boundsMin) * scale));

            bin[binIdx].tri_count++;
            bin[binIdx].bounds.grow(tris[currTriIdx]);
        }

        // gather data for the n-1 planes between the n bins
        float leftArea[INTERVAL_NUMBER - 1], rightArea[INTERVAL_NUMBER - 1];
        int leftCount[INTERVAL_NUMBER - 1], rightCount[INTERVAL_NUMBER - 1];
        AABB leftBox, rightBox;
        int leftSum = 0, rightSum = 0;
        for (uint32_t i = 0; i < INTERVAL_NUMBER - 1; i++)
        {
            leftSum += bin[i].tri_count;
            leftCount[i] = leftSum;
            leftBox.grow( bin[i].bounds );
            leftArea[i] = leftBox.area();
            rightSum += bin[INTERVAL_NUMBER - 1 - i].tri_count;
            rightCount[INTERVAL_NUMBER - 2 - i] = rightSum;
            rightBox.grow( bin[INTERVAL_NUMBER - 1 - i].bounds );
            rightArea[INTERVAL_NUMBER - 2 - i] = rightBox.area();
        }

        // calculate SAH cost for the n-1 planes
        scale = (boundsMax - boundsMin) / INTERVAL_NUMBER;
        for (uint32_t i = 0; i < INTERVAL_NUMBER - 1; i++) {
            float planeCost = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
            if (planeCost < bestCost) {
                bestAxis = axis;
                splitPos = boundsMin + scale * (i + 1);
                bestCost = planeCost;
            }
        }
    }

    return bestCost;
}

void BVH::subdivide(uint32_t node_id) {
    BVHNode& node = nodes[node_id];
    if (node.tri_count <= 2) return;

    int bestAxis = 0;
    float splitPos = 0;
    float bestCost = findBestSplitPos(node, splitPos, bestAxis);

    float noSplitCost = calculateNodeCost(node);
    if (bestCost >= noSplitCost) return;

    uint32_t i = (int32_t)node.left_first;
    uint32_t j = (int32_t)(node.left_first + node.tri_count - 1);

    while (i <= j) {
        Vec3 c = centroids[tri_idx[i]];
        if (c[bestAxis] < splitPos) {
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
