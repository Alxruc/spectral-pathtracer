#include <vector>
#include <iostream>
#include <chrono>

#include "stb_image_write.h"

#include <renderer.hpp>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

int main() {
    int W = 1000;
    int H = 600;
    size_t bytes = size_t(W) * H * 3;

    Renderer* r = renderer_create(W, H);
    std::cout << "Building BVH and creating normals\n";

    auto t1 = high_resolution_clock::now();
    build_bvh(r);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "Took " << ms_int.count() << " ms\n";

    std::vector<unsigned char> pixels(bytes);
    std::cout << "Rendering \n";
    t1 = high_resolution_clock::now();
    render(r, pixels);
    t2 = high_resolution_clock::now();
    ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "Took " << ms_int.count() << " ms\n";
    renderer_destroy(r);

    if (!stbi_write_png("out.png", W, H, 3, pixels.data(), W * 3)) {
        return 1;
    }

    return 0;
}
