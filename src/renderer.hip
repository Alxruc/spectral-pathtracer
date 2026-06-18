#include <hip/hip_runtime.h>

#include <gpu_ds.hpp>
#include <renderer.hpp>

const int BLOCK_SIZE = 16;

struct Renderer {
    unsigned char* d_img;
    dim3 block, grid;
    int width, height;
};


__device__ float hitSphere(Vec3 center, float radius, Ray &ray) {
    Vec3 oc = ray.origin - center;
    float a = dot(ray.dir, ray.dir);
    float b = 2 * dot(oc, ray.dir);
    float c = dot(oc, oc) - radius * radius;
    
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return -1;

    // return closer root
    return (-b - sqrtf(disc)) / (2 * a);
}

__global__ void scene(unsigned char* img, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float aspect = float(width) / float(height);
    float u = (px + 0.5f) / float(width);    // 0 to 1, left edge to right edge
    float v = (py + 0.5f) / float(height);   // 0 to 1, top edge to bottom edge


    Ray ray;
    ray.origin = {0.0f, 0.0f, 0.0f};
    ray.dir = {
        (u - 0.5f) * 2.0f * aspect,
        (0.5f - v) * 2.0f,
        -1.0f
    };
    
    Vec3 center = {0.5f, 0.8f, -2.0f};
    float radius = 0.7f;
    float t = hitSphere(center, radius, ray);
    Vec3 color;
    if (t > 0.0f) {
        Vec3 p = ray.origin + t * ray.dir;
        Vec3 n = normalize(p-center);
        color = 0.5 * (n + Vec3{1.0f, 1.0f, 1.0f});
    }
    else {
        Vec3 d = normalize(ray.dir);
        float s = 0.5f * (d.y + 1.0f);
        color = (1.0f - s) * Vec3{1.0f, 1.0f, 1.0f} + s * Vec3{0.5f, 0.7f, 1.0f};
    }

    int idx = 3 * (py * width + px);
    img[idx + 0] = (unsigned char)(255.999f * color.x);
    img[idx + 1] = (unsigned char)(255.999f * color.y);
    img[idx + 2] = (unsigned char)(255.999f * color.z);
}

Renderer* renderer_create(int w, int h) {
    Renderer* r = new Renderer{};
    r->width = w; r->height = h;
    r->block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    r->grid  = dim3((w + BLOCK_SIZE-1) / BLOCK_SIZE, (h + BLOCK_SIZE-1) / BLOCK_SIZE);
    HIP_CHECK(hipMalloc(&r->d_img, size_t(w) * h * 3));
    return r;
}

void renderer_destroy(Renderer* r) { 
    HIP_CHECK(hipFree(r->d_img));
    delete r;
}

void render(Renderer* r, std::vector<unsigned char> &pixels) {
    scene<<<r->grid, r->block>>>(r->d_img, r->width, r->height);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(pixels.data(), r->d_img, pixels.size(), hipMemcpyDeviceToHost));
}
