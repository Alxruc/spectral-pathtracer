#include <vector>

#include "stb_image_write.h"

#include <renderer.hpp>


int main() {
    int W = 1000;
    int H = 600;
    size_t bytes = size_t(W) * H * 3;

    
    Renderer* r = renderer_create(W, H);
    
    std::vector<unsigned char> pixels(bytes);
    render(r, pixels);
    renderer_destroy(r);

    if (!stbi_write_png("out.png", W, H, 3, pixels.data(), W * 3)) {
        return 1;
    }

    return 0;
}