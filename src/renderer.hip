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
#include <wavelength.hpp>
#include <light.hpp>

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

__device__ float3 sampleSphere(RNG& rng)
{
    float z   = 1.0f - 2.0f * rng.next();          // uniform in [-1, 1]
    float r   = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = TWO_PI * rng.next();       // 2pi times u
    return {r * cosf(phi), r * sinf(phi), z};      // already unit length
}

__device__ float3 sampleSunDisk(RNG& rng)
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
    float3 u, v;
    tangentFrame(SUN_DIR, u, v);

    return normalize(x * u + y * v + z * SUN_DIR);
}

__device__ bool hitScene(const DeviceBVH& bvh, const Ray& r, Hit& h)
{
    bool found = false;

    float t;
    float3 tri_n;
    uint32_t global_triangle_idx;
    if (rayBVHIntersect(r, bvh, t, tri_n, global_triangle_idx)) {
        found = true;
        h.t = t;
        h.front_face = dot(r.dir, tri_n) < 0.0f;
        h.n = h.front_face ? tri_n : -tri_n;
        h.p = r.origin + t * r.dir; // exact surface point, offsetRayOrigin handles acne when spawning
        h.obj_idx = bvh.getObjectIdx(global_triangle_idx);
        switch (h.obj_idx) {
            case 0:  h.mat = &bvh.metal;      break; // spot
            case 1:  h.mat = &bvh.lambertian; break; // floor
            case 2:  h.mat = &bvh.glass;      break; // prism
            default: h.mat = &bvh.emissive;   break; // light
        }
        h.tri = bvh.tris[global_triangle_idx];
    }

    return found;
}

// nudges a ray start off the surface toward the side dir is heading, so the surface
// it just left lies behind the ray and can't be hit again (works for either sign of n)
__device__ float3 offsetRayOrigin(float3 p, float3 n, float3 dir)
{
    const float RAY_OFFSET = 1e-4f;
    return dot(dir, n) > 0.0f ? p + RAY_OFFSET * n : p - RAY_OFFSET * n;
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

__device__ float skyRadiance(float3 dir, float lamda) {
    float3 d = normalize(dir);
    float s = 0.5f * (d.y + 1.0f);
    float radiance = (1.0f - s) * 1.0f + s * 0.7f;
    return radiance * powf(NORMALIZATION_WAVELENGTH_NM / lamda, 4.0); // 550nm as baseline as human eye
}

__device__ float sunRadiance(float lamda) {
    return SUN_SCALE * blackbody(lamda, SUN_TEMPERATURE) / SUN_NORMALIZING;
}

__device__ float4 skyRadiance(float3 dir, float4 lambdas) {
    float3 d = normalize(dir);
    float s = 0.5f * (d.y + 1.0f);
    float base_radiance = (1.0f - s) * 1.0f + s * 0.7f;

    // Rayleigh factor for each wavelength: (550nm / lambda)^4
    // 550nm as baseline for human eye
    float4 rayleigh;
    rayleigh.x = powf(NORMALIZATION_WAVELENGTH_NM / lambdas.x, 4.0f);
    rayleigh.y = powf(NORMALIZATION_WAVELENGTH_NM / lambdas.y, 4.0f);
    rayleigh.z = powf(NORMALIZATION_WAVELENGTH_NM / lambdas.z, 4.0f);
    rayleigh.w = powf(NORMALIZATION_WAVELENGTH_NM / lambdas.w, 4.0f);

    // Component-wise multiplication: base_radiance * rayleigh
    return make_float4(
        base_radiance * rayleigh.x,
        base_radiance * rayleigh.y,
        base_radiance * rayleigh.z,
        base_radiance * rayleigh.w
    );
}

__device__ float4 sunRadiance(float4 lamdas) {
    return make_float4(
        SUN_SCALE * blackbody(lamdas.x, SUN_TEMPERATURE) / SUN_NORMALIZING,
        SUN_SCALE * blackbody(lamdas.y, SUN_TEMPERATURE) / SUN_NORMALIZING,
        SUN_SCALE * blackbody(lamdas.z, SUN_TEMPERATURE) / SUN_NORMALIZING,
        SUN_SCALE * blackbody(lamdas.w, SUN_TEMPERATURE) / SUN_NORMALIZING
    );
}

__device__ float4 emittedRadiance(const EmissiveProps& e, float4 lamdas) {
    float norm = e.brightness / blackbody(NORMALIZATION_WAVELENGTH_NM, e.temperature);
    return make_float4(
        norm * blackbody(lamdas.x, e.temperature),
        norm * blackbody(lamdas.y, e.temperature),
        norm * blackbody(lamdas.z, e.temperature),
        norm * blackbody(lamdas.w, e.temperature)
    );
}


// Simple Analytic Approximations to the CIE XYZ Color Matching Functions
// Wyman et al
__device__ float3 evaluateCMF(float lamda) {
    float ln_lamda = logf(lamda);
    float x = 1.065 * expf(-0.5 * powf(((lamda - 595.8) / 33.33), 2)) + 0.366 * expf(-0.5 * powf(((lamda-446.8) / 19.44), 2));
    float y = 1.014 * expf(-0.5 * powf(((ln_lamda - logf(556.3)) / 0.075), 2));
    float z = 1.839 * expf(-0.5 * powf(((ln_lamda - logf(449.8)) / 0.051), 2));

    return {x, y, z};
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

__device__ float3 reflect(float3 dir, float3 normal) {
    return dir - 2.0f * (dot(dir, normal)) * normal;
}

__device__ float3 refract(float3 dir, float3 normal, float eta) {
    // standard opengl refract formula
    float dot_N_I = dot(normal, dir);
    float k = 1.0f - eta * eta * (1.0f - dot_N_I * dot_N_I);
    return eta * dir - (eta * dot_N_I + sqrtf(k)) * normal;
}

__device__ float light_pdf(float3 omega) {
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

// one place where the straight shadow ray crosses a glass surface
struct SpecularVertex { float3 p; int obj_idx; bool entering; Triangle tri; };

enum class SeedOutcome { Escaped, Blocked };

// seed path Y from the MNEE paper (Sec. 3.1): the glass crossings between x_b and the light
struct SeedPath {
    SpecularVertex v[MAX_SPECULAR_BOUNCES];
    int count = 0;
    SeedOutcome outcome = SeedOutcome::Blocked;
};

// Walks the shadow ray straight toward the light, passing through glass without bending
// the seed only tells the manifold walk which surfaces to solve on
__device__ SeedPath traceSeedPath(Ray shadow_ray, const DeviceBVH& bvh)
{
    SeedPath seed;
    for (int i = 0; i < MAX_SPECULAR_BOUNCES; i++) {
        Hit shadow_hit;
        if (!hitScene(bvh, shadow_ray, shadow_hit)) {
            seed.outcome = SeedOutcome::Escaped;
            return seed;
        }

        //  Hit: opaque = give up; Glass = save it and keep walking
        if(shadow_hit.mat->type != MatType::Dielectric) {
            seed.outcome = SeedOutcome::Blocked;
            return seed;
        }
        else {
            SpecularVertex glassVertex = {shadow_hit.p, shadow_hit.obj_idx, shadow_hit.front_face, shadow_hit.tri };
            seed.v[seed.count] = glassVertex;
            seed.count++;
            shadow_ray.origin = offsetRayOrigin(shadow_hit.p, shadow_hit.n, shadow_ray.dir);
        }
    }

    // cap reached: anything still along the ray (more glass or something opaque) means we give up
    Hit shadow_hit;
    seed.outcome = hitScene(bvh, shadow_ray, shadow_hit) ? SeedOutcome::Blocked : SeedOutcome::Escaped;
    return seed;
}

// Snell's law error at one glass vertex: how much the generalized half vector leans sideways
// wi and wo point away from the vertex, eta_i/eta_o are of the medium they travel through
__device__ float2 vertexError(float3 normal, float3 wi, float eta_i, float3 wo, float eta_o) {
    float3 half = normalize(eta_i * wi + eta_o * wo);
    float3 s, t;
    tangentFrame(normal, s, t);
    return make_float2(dot(half, s), dot(half, t));
}

// Fills errors[2i] and errors[2i + 1] for every glass vertex i of path
__device__ void getGlassErrors(float ior, float3 x_b, float3 l,
                               const SeedPath& path, float errors[MNEE_DIM]) {
    for (int i = 0; i < path.count; i++) {
        const SpecularVertex& v = path.v[i];

        float3 prev = i == 0 ? x_b : path.v[i - 1].p;
        float3 wi = normalize(prev - v.p);
        // the last vertex has no next point, its outgoing arrow is the sun direction itself
        float3 wo = i == path.count - 1 ? l : normalize(path.v[i + 1].p - v.p);

        // entering: wi travels through air and wo through glass, leaving: the other way round
        float eta_i = v.entering ? 1.0f : ior;
        float eta_o = v.entering ? ior : 1.0f;

        Triangle triangle = v.tri;
        float2 e = vertexError(triangle.normal, wi, eta_i, wo, eta_o);
        errors[2 * i]     = e.x;
        errors[2 * i + 1] = e.y;
    }
}

// J[row][col] = how much errors[row] changes per nudge along col
// Column 2i nudges vertex i along its s, column 2i + 1 along its t 
// what tangentFrame gives at  the vertex's current position, so the Newton step later moves points in the same directions.
__device__ void glassErrorJacobian(float ior, float3 x_b, float3 l,
                                   const SeedPath& path, float J[MNEE_DIM][MNEE_DIM], float base_errors[MNEE_DIM]) {

    // nudge the path point, project it back onto its sphere and check the change in errors
    for (int col = 0; col < 2 * path.count; col++) {
        int idx = col / 2;
        SeedPath nudged = path; 
        float3 x = nudged.v[idx].p;

        Triangle triangle = path.v[idx].tri;
        float3 s,t;
        tangentFrame(triangle.normal, s, t);

        float delta = 1e-3f; // just the same constant for everything for now for simplicity
        if(col % 2 == 0) x = x + delta * s;
        else             x = x + delta * t;

        // i dont really care that we could land outside the mesh here
        // trying stuff out
        x = projectOntoPlane(triangle.normal, triangle.a, x); 
        nudged.v[idx].p = x;
        float nudged_errors[MNEE_DIM];

        getGlassErrors(ior, x_b, l, nudged, nudged_errors);

        for(int row = 0; row < 2 * path.count; row++) {
            J[row][col] = (nudged_errors[row] - base_errors[row]) / delta;
        }
    }

}


// does a light reach from to to with nothing in between
__device__ bool segmentReaches(const DeviceBVH& bvh, float3 from, float3 to, int obj_idx)
{
    Ray r = {from, normalize(to - from), false};
    Hit hit;
    if (!hitScene(bvh, r, hit) || hit.obj_idx != obj_idx) return false;
    float3 diff = hit.p - to;
    return dot(diff, diff) < 1e-6f;
}

__device__ float3 traceRadiance(Ray ray, RNG& rng, const DeviceBVH& bvh)
{
    float4 lamdas = sampleHeroWavelength(rng);
    float4 radiance = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 throughput = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

    const float prev_brdf_density = 1 / TWO_PI; // because we sample a constant hemisphere for bounces this is constant for now
    for (int depth = 0; depth < 64; depth++) {
        Hit h;
        if (!hitScene(bvh, ray, h)) {
            float w = 1;
            if (ray.prev_was_diffuse) {
                float light_density = light_pdf(ray.dir);
                // if light_density = 0, then w = 1 so our radiance doesnt get affected
                w = prev_brdf_density / (prev_brdf_density + light_density);
            }
            float4 sky_rad = skyRadiance(ray.dir, lamdas);
            radiance += throughput * sky_rad * w;
            break;
        }
        float3 d;
        // we hit glass
        if (h.mat->type == MatType::Dielectric) {
            // when dispersing we let go of the hero wavelength idea
            // since all different wavelengths take different paths here
            // we just focus on the hero
            float ior = cauchy(lamdas.x);
            // if we hit the front face, we are entering the glass, if not we are leaving the glass
            // eta = n_from / n_to
            float eta = h.front_face ? 1.0 / ior : ior;

            float3  unit = normalize(ray.dir);
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
            float n = getRefractiveIndex(h.mat->metal.metal, lamdas.x);
            float k = getExtinctionIndex(h.mat->metal.metal, lamdas.x);
            float r_0 = (powf(n-1, 2) + powf(k, 2)) / (powf(n+1, 2) + powf(k,2));

            float3  unit = normalize(ray.dir);
            float cos_t = fminf(dot(-1.0f * unit, h.n), 1.0f);
            float reflectance = r_0 + (1-r_0) * powf((1-cos_t), 5);

            d = reflect(unit, h.n) + (h.mat->metal.roughness * sampleSphere(rng));
            throughput *= reflectance;
            ray.prev_was_diffuse = false;
        }
        else if (h.mat->type == MatType::Emissive) {
            float cos_l = dot(h.n, -ray.dir);
            if (h.front_face) { // only the emitting side
                // Convert the mesh's area PDF to solid angle measure for this bounce hit
                float pdf_light_sa = h.mat->emissive.pdf_area * (h.t * h.t) / cos_l;
                
                // Balance heuristic against the BSDF PDF that sampled this bounce ray
                float mis_weight = 1;
                if (ray.prev_was_diffuse) {
                    mis_weight = prev_brdf_density / (prev_brdf_density + pdf_light_sa);
                }
                
                radiance += throughput * emittedRadiance(h.mat->emissive, lamdas) * mis_weight;  
            }
            break;
        }
        // we hit other materials
        else {
            float3 l = sampleSunDisk(rng);
            Ray sun_block_ray = {offsetRayOrigin(h.p, h.n, l), l, false};
            float n_dot_l = dot(l, h.n);
            SeedPath seed = traceSeedPath(sun_block_ray, bvh);
            bool escaped = seed.outcome == SeedOutcome::Escaped;

            // nothing in the way: regular NEE
            if (escaped && seed.count == 0 && n_dot_l > 0) {
                float4 sun_rad = sunRadiance(lamdas);
                float mis_weight = 1.0f / (ONE_OVER_OMEGA_SUN + prev_brdf_density);

                radiance += throughput 
                          * (h.mat->lambertian.albedo / PI) 
                          * sun_rad 
                          * n_dot_l
                          * mis_weight;
            }
            // only glass, we do MNEE seed.v[0 .. seed.count) is the straight line seed
            else if (escaped && seed.count > 1) {
                float ior = cauchy(lamdas.x);
                SeedPath path = seed; // the walk along points, seed is starting guess
                bool converged = false;
                float errors[MNEE_DIM] = {};

                // Newton loop
                for(int i = 0; i < MNEE_MAX_ITERATIONS; i++) {
                    getGlassErrors(ior, h.p, l, path, errors);
                    bool done = true;
                    for(int j = 0; j < 2 * path.count; j++) {
                        if(fabsf(errors[j]) > MNEE_TOLERANCE) {
                            done = false;
                            break;
                        }
                    }
                    if(done) {
                        converged = true;
                        break;
                    }
                
                    float J[MNEE_DIM][MNEE_DIM] = {};
                    glassErrorJacobian(ior, h.p, l, path, J, errors);
                    float step[MNEE_DIM] = {};

                    float flipped[MNEE_DIM] = {};
                    for (int e = 0; e < MNEE_DIM; e++) {
                        flipped[e] = -errors[e];
                    }
                    if(!solve4(J, flipped, 2 * path.count, step)) break;

                    for(int p = 0; p < path.count; p++) {
                        Triangle triangle = path.v[p].tri;
                        float3 s,t;
                        tangentFrame(triangle.normal, s, t);
                        float3 adjusted = path.v[p].p + step[2*p] * s + step[2*p + 1] * t;
                        path.v[p].p = projectOntoPlane(triangle.normal, triangle.a, adjusted);
                    }

                }

                if(converged) {
                    const float3 x1 = path.v[0].p, x2 = path.v[1].p;
                    const float3 n1 = path.v[0].tri.normal;
                    const float3 n2 = path.v[1].tri.normal;

                    // light really arrives along x_b to x1, not along l
                    float3 to_x1 = normalize(x1 - h.p);
                    float cos_b = dot(to_x1, h.n);

                    // the bent path must exist in the scene so ground hit to x1, x1 to x2, and x2 to the light
                    Hit escape_hit;
                    bool valid = cos_b > 0
                        && segmentReaches(bvh, offsetRayOrigin(h.p, h.n, to_x1), x1, path.v[0].obj_idx)
                        && segmentReaches(bvh, offsetRayOrigin(x1, n1, x2 - x1), x2, path.v[1].obj_idx)
                        && !hitScene(bvh, Ray{offsetRayOrigin(x2, n2, l), l, false}, escape_hit);

                    if (valid) {
                        // Fresnel transmission at each glass point
                        // schlick's r0 is the same for ior and 1/ior, so the ratio direction doesn't matter.
                        float T1 = 1.0f - schlick(fabsf(dot(-1.0f * to_x1, n1)), ior);
                        float T2 = 1.0f - schlick(fabsf(dot(l, n2)), ior);

                        float delta = 1e-3f;
                        float3 u,v;
                        tangentFrame(l,u,v);
                        float3 l_u = normalize(l + delta * u);
                        float3 l_v = normalize(l + delta * v);
                        // same points, tilted sun: only the last vertex's two errors become non-zero
                        float errors_l_u[MNEE_DIM], errors_l_v[MNEE_DIM] = {};
                        getGlassErrors(ior, h.p, l_u, path, errors_l_u);
                        getGlassErrors(ior, h.p, l_v, path, errors_l_v);

                        float J[MNEE_DIM][MNEE_DIM] = {};
                        glassErrorJacobian(ior, h.p, l, path, J, errors);

                        float rhs[MNEE_DIM] = {};
                        for(int i = 0; i < MNEE_DIM; i++)
                            rhs[i] = -(errors_l_u[i] - errors[i]) / delta;

                        float dX_u[MNEE_DIM];
                        solve4(J, rhs, 2*path.count, dX_u);

                        for(int i = 0; i < MNEE_DIM; i++)
                            rhs[i] = -(errors_l_v[i] - errors[i]) / delta;

                        float dX_v[MNEE_DIM];
                        solve4(J, rhs, 2*path.count, dX_v);

                        float M[2][2] = {{dX_u[0], dX_v[0]}, {dX_u[1], dX_v[1]}};

                        float G = fabsf(M[0][0] * M[1][1] - M[0][1]*M[1][0]) * 
                                    fabsf(dot(n1, -to_x1)) 
                                    / dot(x1 - h.p, x1 - h.p);

                        // the walk used the hero wavelength's IOR, so only that lane can use this path.
                        // (no x4 for the missing lanes: fwrap() in sampleHeroWavelength currently puts the
                        // support lanes outside the visible range, so lane x is the whole signal anyway)
                        float sun_rad = sunRadiance(lamdas.x);
                        radiance.x += throughput.x
                                    * (h.mat->lambertian.albedo / PI)
                                    * sun_rad
                                    * cos_b * T1 * T2 * G
                                    / ONE_OVER_OMEGA_SUN;
                    }
                }
            }
            // otherwise x_b is in shadow

            // NEE towards the emissive square
            // really ugly and (hopefully) temporary
            AreaLight light_sample = sampleLight(bvh, rng);
            float3 light_dir = light_sample.p - h.p;
            float light_dist_sq = length_sq(light_dir);

            light_dir = light_dir / sqrtf(light_dist_sq);

            float n_dot_light = dot(light_dir, h.n);

            float cos_light = dot(light_sample.n, -light_dir);

            if(n_dot_light > 0 && cos_light > 0 && 
                segmentReaches(bvh, offsetRayOrigin(h.p, h.n, light_dir), light_sample.p, LIGHT_OBJ_IDX)) {
                float4 light_rad = emittedRadiance(light_sample.material->emissive, lamdas);
                float pdf_light_sa = light_sample.material->emissive.pdf_area * (light_dist_sq) / cos_light;
                float mis_weight =  1.0f / (pdf_light_sa + prev_brdf_density);

                radiance += throughput 
                          * (h.mat->lambertian.albedo / PI) 
                          * light_rad 
                          * n_dot_light
                          * mis_weight;
            }

            d = sampleSphere(rng);
            if (dot(d, h.n) < 0.0f) d = -1.0f * d;
            throughput *= 2.0f * h.mat->lambertian.albedo * dot(d, h.n);
            ray.prev_was_diffuse = true;
        }

        ray.origin = offsetRayOrigin(h.p, h.n, d);
        ray.dir    = d;
    }
    // convert accumulated 4-lane radiance to CIE XYZ at the end
    constexpr float CIE_Y_INTEGRAL = 106.856895f;
    const float spectral_norm = 1.0f / (4.0f * UNIFORM_DENSITY * CIE_Y_INTEGRAL);
    
    float3 cmf0 = evaluateCMF(lamdas.x);
    float3 cmf1 = evaluateCMF(lamdas.y);
    float3 cmf2 = evaluateCMF(lamdas.z);
    float3 cmf3 = evaluateCMF(lamdas.w);

    // Sum all 4 spectral contributions into the XYZ accumulator
    float3 xyz = spectral_norm * 
                (radiance.x * cmf0 + 
                radiance.y * cmf1 + 
                radiance.z * cmf2 + 
                radiance.w * cmf3);

    return xyz; // returns linear CIE XYZ
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
    float3 sum = {0.0f, 0.0f, 0.0f};
    const float3 worldUp = {0.0f, 1.0f, 0.0f};

    const float3 targetPos = {0.0f, 0.0f, 0.0f};
    const float3 cameraPos = {0.0f, 0.0f, -2.0f};

    const float3 forward = normalize(targetPos - cameraPos);
    const float3 right = normalize(cross(forward, worldUp));
    const float3 up = cross(right, forward);

    for (int s = 0; s < spp; ++s) {
        float u = (px + rng.next()) / float(width);   // + jitter
        float v = (py + rng.next()) / float(height);
        Ray ray;
        ray.origin = cameraPos;
        ray.dir    = (u - 0.5f) * 2.0f * aspect * right + (0.5f - v) * 2.0f * up + forward;
        sum = sum + traceRadiance(ray, rng, bvh);
    }

    float3 avg_xyz = sum / float(spp);

    float3 rgb = XYZ_to_sRGB(avg_xyz);

    int idx = 3 * (py * width + px);
    img[idx + 0] = (unsigned char)(255.999f * clamp(rgb.x, 0.0, 1.0));
    img[idx + 1] = (unsigned char)(255.999f * clamp(rgb.y, 0.0, 1.0));
    img[idx + 2] = (unsigned char)(255.999f * clamp(rgb.z, 0.0, 1.0));
}

void build_bvh(Renderer* r) {
    std::vector<Triangle> mesh = loadObj("data/spot.obj");
    std::vector<uint32_t> indexBoundaries = {0};
    indexBoundaries.push_back(mesh.size());

    std::vector<Triangle> floor = loadObj("data/floor.obj");
    std::vector<Triangle> combined = mesh;
    combined.insert(combined.end(), floor.begin(), floor.end());
    indexBoundaries.push_back(combined.size());

    std::vector<Triangle> prism = loadObj("data/prism.obj");
    combined.insert(combined.end(), prism.begin(), prism.end());
    indexBoundaries.push_back(combined.size());

    std::vector<Triangle> light = loadObj("data/light.obj");
    combined.insert(combined.end(), light.begin(), light.end());
    indexBoundaries.push_back(combined.size());

    BVH bvh;
    bvh.tris = combined;
    bvh.build();
    r->scene.manager.upload(bvh, indexBoundaries);
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
