#include <hip/hip_runtime.h>

#include <complex>
#include <gpu_ds.hpp>
#include <geometry.hpp>
#include <iterator>
#include <renderer.hpp>
#include <constants.hpp>
#include <materials.hpp>
#include <intersect.hpp>
#include <acceleration_ds.hpp>
#include <io.hpp>

// dont know if this is bad
struct Scene {
    GPUBVHManager manager;
};

struct Renderer {
    unsigned char* d_img;
    dim3 block, grid;
    int width, height;
    Scene scene;
};

// (5, 5, 2) normalized
__device__ const Vec3 SUN_DIR = {0.680414f, 0.680414f, 0.272166f};
__device__ const float SUN_TEMPERATURE = 5778; // K

// for our sun to be brighter than the sky
// number pulled out from thin air for now just based on vibes
// to make it look semi right
__device__ const float SUN_SCALE = 10e4;

// blackbody(550nm, 5778) precomputed
__device__ const float SUN_NORMALIZING = 2.586 * 10e13;

// explained in light sampling function where they are used
__device__ const float COS_SUN_RADIUS = 0.9999893;
__device__ const float ONE_OVER_OMEGA_SUN = 14'900;


__device__ Vec3 sampleSphere(RNG& rng)
{
    float z   = 1.0f - 2.0f * rng.next();          // uniform in [-1, 1]
    float r   = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = TWO_PI * rng.next();       // 2pi times u
    return {r * cosf(phi), r * sinf(phi), z};      // already unit length
}

__device__ Vec3 sampleSunDisk(RNG& rng)
{
    // Two uniform random numbers
    float u1 = rng.next();
    float u2 = rng.next();

    // Uniformly sample cos(theta) between cos(sun radius) and 1.
    // Uniform in cos_theta <=> uniform over solid angle of the cone.
    float cos_theta = 1.0f - u1 * (1.0f - COS_SUN_RADIUS);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = TWO_PI * u2;

    // Direction in local frame where +z = SUN_DIR
    float x = sin_theta * cosf(phi);
    float y = sin_theta * sinf(phi);
    float z = cos_theta;

    // Build an orthonormal basis around SUN_DIR
    // SUN_DIR is assumed to be normalized here
    Vec3 a = fabsf(SUN_DIR.x) > 0.9f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 u = normalize(cross(a, SUN_DIR));
    Vec3 v = cross(SUN_DIR, u);

    return normalize(x * u + y * v + z * SUN_DIR);
}

__device__ bool hitScene(const Sphere* scn, int n, const DeviceBVH& bvh, const Ray& r, Hit& h)
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

    float t;
    Vec3 tri_n;
    if (rayBVHIntersect(r, bvh, t, tri_n) && t < closest) {
        found = true;
        h.t = t;
        h.front_face = dot(r.dir, tri_n) < 0.0f;
        h.n = h.front_face ? tri_n : -tri_n;
        h.p = r.origin + t * r.dir + 1e-4f * h.n; //small offset for shadow acne
        h.mat = &bvh.mat;
    }

    return found;
}

__device__ float blackbody(float lamda, float temperature) {
    float lamda_meters = lamda * 1e-9;

    float l5 = powf(lamda_meters, 5.0);
    float exponent = COEF_C2 / (lamda_meters * temperature);

    // protect against exp() overflow directly in float
    if (exponent > 88.0f) { // e^88 is near the upper limit of float (~3.4e38)
        return 0.0f;
    }

    float radiance = COEF_C1 / (l5 * (expf(exponent) - 1.0f));

    return radiance;
}

__device__ float skyRadiance(Vec3 dir, float lamda) {
    Vec3 d = normalize(dir);
    float s = 0.5f * (d.y + 1.0f);
    float radiance = (1.0f - s) * 1.0f + s * 0.7f;
    return radiance * powf(NORMALIZATION_WAVELENGTH_NM / lamda, 4.0); // 550nm as baseline as human eye
}

__device__ float sunRadiance(float lamda) {
    return SUN_SCALE * blackbody(lamda, SUN_TEMPERATURE) / SUN_NORMALIZING;
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

__device__ float light_pdf(Vec3 omega) {
    // If we land inside the sun's disk, meaning a cone spanning about 0.53 degrees
    // of angular diameter (we halve it for the radius and then compute cos(angle)),
    // then we return the density function of the light, which
    // is 1 / (size of sun as seen from Earth measured as area on the unit sphere)
    // often denoted as 1 / OMEGA_sun
    // Where OMEGA_sun = 6.7 * 10^-5 steredian
    if (dot(omega, SUN_DIR) > COS_SUN_RADIUS)
        return ONE_OVER_OMEGA_SUN;
    return 0.0;
}


__device__ Vec3 traceRadiance(Ray ray, RNG& rng, const Sphere* scn, int n, const DeviceBVH& bvh)
{
    float density = 0;
    float lamda = sampleWavelength(rng, density);

    float radiance = 0;
    float throughput = 1.0f / density;
    const float prev_brdf_density = 1 / TWO_PI; // because we sample a constant hemisphere for bounces this is constant for now
    for (int depth = 0; depth < 64; depth++) {
        Hit h;
        if (!hitScene(scn, n, bvh, ray, h)) {
            float w = 1;
            if (ray.prev_was_diffuse) {
                float light_density = light_pdf(ray.dir);
                // if light_density = 0, then w = 1 so our radiance doesnt get affected
                w = prev_brdf_density / (prev_brdf_density + light_density);
            }
            radiance += throughput * skyRadiance(ray.dir, lamda) * w;
            break;
        }


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
            throughput *= h.mat->dielectric.albedo;
            ray.prev_was_diffuse = false;
        }
        // we hit metal
        else if (h.mat->type == MatType::Metal) {
            float n = getRefractiveIndex(h.mat->metal.metal, lamda);
            float k = getExtinctionIndex(h.mat->metal.metal, lamda);
            float r_0 = (powf(n-1, 2) + powf(k, 2)) / (powf(n+1, 2) + powf(k,2));

            Vec3  unit = normalize(ray.dir);
            float cos_t = fminf(dot(-1.0f * unit, h.n), 1.0f);
            float reflectance = r_0 + (1-r_0) * powf((1-cos_t), 5);

            d = reflect(unit, h.n) + (h.mat->metal.roughness * sampleSphere(rng));
            throughput *= reflectance;
            ray.prev_was_diffuse = false;
        }
        // we hit other materials
        else {
            Vec3 l = sampleSunDisk(rng);
            Ray shadow_ray = {h.p + EPSILON * h.n, l, false};
            Hit throwaway;
            if(dot(l, h.n) > 0 && !hitScene(scn, n, bvh, shadow_ray, throwaway)) {
                radiance += throughput
                    * (h.mat->lambertian.albedo) / PI
                    * sunRadiance(lamda)
                    * dot(l, h.n)
                    / (ONE_OVER_OMEGA_SUN + prev_brdf_density);
            }

            d = sampleSphere(rng);
            if (dot(d, h.n) < 0.0f) d = -1.0f * d;
            throughput *= 2.0f * h.mat->lambertian.albedo * dot(d, h.n);
            ray.prev_was_diffuse = true;
        }

        ray.origin = h.p;
        ray.dir    = d;
    }
    return radiance * evaluateCMF(lamda);
}

__device__ float clamp(float d, float min, float max) {
  const float t = d < min ? min : d;
  return t > max ? max : t;
}

__global__ void scene(unsigned char* img, int width, int height, DeviceBVH bvh, int spp) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    RNG rng{ uint32_t(py) * uint32_t(width) + uint32_t(px) };

    float aspect = float(width) / float(height);
    Vec3 sum = {0.0f, 0.0f, 0.0f};
    const Vec3 worldUp = {0.0f, 1.0f, 0.0f};

    Sphere floor = {
       {0,-100.7,-2},
       100,
       make_lambertian(0.8)
    };

    Sphere scn[1] = {floor};

    const Vec3 targetPos = {0.0f, 0.0f, 0.0f};
    const Vec3 cameraPos = {0.0f, 0.0f, -2.0f};

    const Vec3 forward = normalize(targetPos - cameraPos);
    const Vec3 right = normalize(cross(forward, worldUp));
    const Vec3 up = cross(right, forward);

    for (int s = 0; s < spp; ++s) {
        float u = (px + rng.next()) / float(width);   // + jitter
        float v = (py + rng.next()) / float(height);
        Ray ray;
        ray.origin = cameraPos;
        ray.dir    = (u - 0.5f) * 2.0f * aspect * right + (0.5f - v) * 2.0f * up + forward;
        sum = sum + traceRadiance(ray, rng, scn, std::size(scn), bvh);
    }

    // picture too bright otherwise
    sum = (NORMALIZATION_CONSTANT / float(spp)) * sum ;

    Vec3 rgb = convertXYZtoRGB(sum.x, sum.y, sum.z);

    int idx = 3 * (py * width + px);
    img[idx + 0] = (unsigned char)(255.999f * clamp(rgb.x, 0.0, 1.0));
    img[idx + 1] = (unsigned char)(255.999f * clamp(rgb.y, 0.0, 1.0));
    img[idx + 2] = (unsigned char)(255.999f * clamp(rgb.z, 0.0, 1.0));
}

void build_bvh(Renderer* r) {
    std::vector<Triangle> mesh = loadObj("data/spot.obj");
    BVH bvh;
    bvh.tris = mesh;
    bvh.build();
    r->scene.manager.upload(bvh);
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
    hipLaunchKernelGGL(scene, r->grid, r->block, 0, 0, r->d_img, r->width, r->height, r->scene.manager.bvh, SPP);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(pixels.data(), r->d_img, pixels.size(), hipMemcpyDeviceToHost));
}
