#include <hip/hip_runtime.h>
#include <algorithm>

#include <gpu_ds.hpp>
#include <renderer.hpp>
#include <constants.hpp>
#include <materials.hpp>

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
            Vec3 outward = (1.0f / scn[i].radius) * (h.p - scn[i].center);
            h.front_face = dot(r.dir, outward) < 0.0f;
            h.n   = h.front_face ? outward : -outward;
            h.mat = &scn[i].mat;
        }
    }
    return found;
}

__device__ float skyRadiance(Vec3 dir, float lamda)
{
    Vec3 d = normalize(dir);
    float s = 0.5f * (d.y + 1.0f);
    float radiance = (1.0f - s) * 1.0f + s * 0.7f;
    return radiance * powf(NORMALIZATION_WAVELENGTH_NM / lamda, 4.0); // 550nm as baseline as human eye
}


// this is uniform for now which means calculating the density every time is wasteful
__device__ float sampleWavelength(RNG& rng, float& density) {
    density =  UNIFORM_DENSITY; // 1 / (MAX_WAVELENGTH_NM - MIN_WAVELENGTH_NM);
    float lamda = (MAX_WAVELENGTH_NM - MIN_WAVELENGTH_NM) * rng.next() + MIN_WAVELENGTH_NM;
    return lamda;
}

// Simple Analytic Approximations to the CIE XYZ Color Matching Functions
// Wyman et al
__device__ Vec3 evaluateCMF(float lamda) {
    float ln_lamda = logf(lamda);
    float x = 1.065 * expf(-0.5 * powf(((lamda - 595.8) / 33.33), 2)) + 0.366 * expf(-0.5 * powf(((lamda-446.8) / 19.44), 2));
    float y = 1.014 * expf(-0.5 * powf(((ln_lamda - logf(556.3)) / 0.075), 2));
    float z = 1.839 * expf(-0.5 * powf(((ln_lamda - logf(449.8)) / 0.051), 2));

    return {x, y, z};
}

__device__ Vec3 convertXYZtoRGB(float X, float Y, float Z) {
    float R =  3.2404542*X - 1.5371385*Y - 0.4985314*Z;
    float G = -0.9692660*X + 1.8760108*Y + 0.0415560*Z;
    float B =  0.0556434*X - 0.2040259*Y + 1.0572252*Z;

    return {R, G, B};
}

__device__ float cauchy(float lamda) {
    // n(lamda) = A + B / lamda ^ 2 (B and lamda in micrometer)
    // we use magic numbers here since we assume that every glass
    // is just BK7 glass, who cares for now
    return 1.5046 + 0.0042 / powf((lamda / 1000), 2);
}

__device__ float schlick(float cos_t, float eta) {
    float r = powf((1.0-eta) / (1.0+eta), 2);
    return r + (1-r) * powf((1-cos_t), 5);
}

__device__ Vec3 reflect(Vec3 dir, Vec3 normal) {
    return dir - 2.0f * (dot(dir, normal)) * normal;
}

__device__ Vec3 refract(Vec3 dir, Vec3 normal, float eta) {
    // standard opengl refract formula
    float dot_N_I = dot(normal, dir);
    float k = 1.0f - eta * eta * (1.0f - dot_N_I * dot_N_I);
    return eta * dir - (eta * dot_N_I + sqrtf(k)) * normal;
}


__device__ Vec3 traceRadiance(Ray ray, RNG& rng, const Sphere* scn, int n)
{
    float density = 0;
    float lamda = sampleWavelength(rng, density);
    float throughput = 1.0f / density;
    for (int depth = 0; depth < 64; ++depth) {
        Hit h;
        if (!hitScene(scn, n, ray, h))
            return throughput * skyRadiance(ray.dir, lamda) * evaluateCMF(lamda);

        Vec3 d;
        // we hit glass
        if (h.mat->type == MatType::Dielectric) {
            float ior = cauchy(lamda);
            // if we hit the front face, we are entering the glass, if not we are leaving the glass
            // eta = n_from / n_to
            float eta = h.front_face ? 1.0 / ior : ior;

            Vec3  unit = normalize(ray.dir);
            float cos_t = fminf(dot(-1.0f * unit, h.n), 1.0f);
            float sin_t = sqrtf(1.0f - cos_t * cos_t);

            bool tir = eta * sin_t > 1;

            if(tir || schlick(cos_t, eta) > rng.next()) {
                d = reflect(unit, h.n);
            }
            else {
                d = refract(unit, h.n, eta);
            }
            throughput *= h.mat->albedo;
        }
        // we hit metal
        else if (h.mat->type == MatType::Metal) {
            float n = getRefractiveIndex(MetalType::Chrome, lamda);  // chrome hardcoded for now
            float k = getExtinctionIndex(MetalType::Chrome, lamda);
            float r_0 = (powf(n-1, 2) + powf(k, 2)) / (powf(n+1, 2) + powf(k,2));

            Vec3  unit = normalize(ray.dir);
            float cos_t = fminf(dot(-1.0f * unit, h.n), 1.0f);
            float reflectance = r_0 + (1-r_0) * powf((1-cos_t), 5);

            d = reflect(unit, h.n);

            throughput *= reflectance;
        }
        // we hit other materials
        else {
            d = sampleSphere(rng);
            if (dot(d, h.n) < 0.0f) d = -1.0f * d;
            throughput *= 2.0f * h.mat->albedo * dot(d, h.n);
        }

        ray.origin = h.p;
        ray.dir    = d;
    }
    return {0.0f, 0.0f, 0.0f};
}

__device__ float clamp(float d, float min, float max) {
  const float t = d < min ? min : d;
  return t > max ? max : t;
}

__global__ void scene(unsigned char* img, int width, int height, int spp) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    RNG rng{ uint32_t(py) * uint32_t(width) + uint32_t(px) };

    float aspect = float(width) / float(height);
    Vec3 sum = {0.0f, 0.0f, 0.0f};

    Sphere floor = {
       {0,-100.7,-2},
       100,
       make_lambertian(0.8)
    };
    Sphere glass = {
        {0,0,-3},
        0.7,
        make_dielectric()
    };
    Sphere reg = {
        {1.2,0.5,-6},
        0.9,
        make_lambertian(0.4)
    };

    Sphere metal = {
        {-1, 0.5, -2.5},
        0.5,
        make_metal(1.0, 0.0)
    };

    Sphere scn[4] = {floor, glass, reg, metal};

    for (int s = 0; s < spp; ++s) {
        float u = (px + rng.next()) / float(width);   // + jitter
        float v = (py + rng.next()) / float(height);
        Ray ray;
        ray.origin = {0.0f, 0.0f, 0.0f};
        ray.dir    = {(u - 0.5f) * 2.0f * aspect, (0.5f - v) * 2.0f, -1.0f};
        sum = sum + traceRadiance(ray, rng, scn, std::size(scn));
    }

    // picture too bright otherwise
    sum = (NORMALIZATION_CONSTANT / float(spp)) * sum ;

    Vec3 rgb = convertXYZtoRGB(sum.x, sum.y, sum.z);

    int idx = 3 * (py * width + px);
    img[idx + 0] = (unsigned char)(255.999f * clamp(rgb.x, 0.0, 1.0));
    img[idx + 1] = (unsigned char)(255.999f * clamp(rgb.y, 0.0, 1.0));
    img[idx + 2] = (unsigned char)(255.999f * clamp(rgb.z, 0.0, 1.0));
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
