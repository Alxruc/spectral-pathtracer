#include <hip/hip_runtime.h>

#include <gpu_ds.hpp>
#include <renderer.hpp>
#include <constants.hpp>

struct Renderer {
    unsigned char* d_img;
    dim3 block, grid;
    int width, height;
};


__device__ Vec3 sampleSphere(RNG& rng)
{
    float z   = 1.0f - 2.0f * rng.next();          // uniform in [-1, 1]
    float r   = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = TWO_PI * rng.next();       // 2pi times u
    return {r * cosf(phi), r * sinf(phi), z};      // already unit length
}

// per sphere: accept the NEAR root if it's inside (tmin, tmax);
// otherwise try the FAR root
__device__ bool hitSphereT(const Sphere& s, const Ray& r,
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

__device__ bool hitScene(const Sphere* scn, int n, const Ray& r, Hit& h)
{
    const float TMIN = 1e-3f; // shadow-acne guard
    float closest = 1e30f;
    bool found = false;
    for (int i = 0; i < n; ++i) {
        float t;
        if (hitSphereT(scn[i], r, TMIN, closest, t)) {
            closest = t; found = true;
            h.t = t;
            h.p = r.origin + t * r.dir;
            h.n = (1.0f / scn[i].radius) * (h.p - scn[i].center);
            h.albedo = scn[i].albedo;
        }
    }
    return found;
}

__device__ float skyRadiance(Vec3 dir)
{
    Vec3 d = normalize(dir);
    float s = 0.5f * (d.y + 1.0f);
    return (1.0f - s) * 1.0f + s * 0.7f;
}

__device__ float traceRadiance(Ray ray, RNG& rng, const Sphere* scn, int n)
{
    float throughput = 1.0f;
    for (int depth = 0; depth < 64; ++depth) {
        Hit h;
        if (!hitScene(scn, n, ray, h))
            return throughput * skyRadiance(ray.dir);
        Vec3 d = sampleSphere(rng);
        if (dot(d, h.n) < 0.0f) d = -1.0f * d;
        throughput *= 2.0f * h.albedo * dot(d, h.n);
        ray.origin = h.p;
        ray.dir    = d;
    }
    return 0.0f;
}

__global__ void scene(unsigned char* img, int width, int height, int spp) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    RNG rng{ uint32_t(py) * uint32_t(width) + uint32_t(px) };

    float aspect = float(width) / float(height);
    float sum = 0.0;

    Sphere scn[2] = {{{0,-100.7,-2}, 100, 0.8}, {{0,0,-2}, 0.7, 0.5}};

    for (int s = 0; s < spp; ++s) {
        float u = (px + rng.next()) / float(width);   // + jitter
        float v = (py + rng.next()) / float(height);  // gives free anti-aliasing
        Ray ray;
        ray.origin = {0.0f, 0.0f, 0.0f};
        ray.dir    = {(u - 0.5f) * 2.0f * aspect, (0.5f - v) * 2.0f, -1.0f};
        sum += traceRadiance(ray, rng, scn, 2);
    }
    sum /= float(spp);

    int idx = 3 * (py * width + px);
    img[idx + 0] = (unsigned char)(255.999f * sum);
    img[idx + 1] = (unsigned char)(255.999f * sum);
    img[idx + 2] = (unsigned char)(255.999f * sum);
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
    scene<<<r->grid, r->block>>>(r->d_img, r->width, r->height, SPP);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(pixels.data(), r->d_img, pixels.size(), hipMemcpyDeviceToHost));
}
