/**
  * \file	RayGenKernels.cu
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	RayGen kernels source file.
  */

#include "rt/HipUtil.h"
#include "rt/ray/RayGenKernels.h"

DEVICE_INLINE Vec3f getTexel(unsigned long long diffuseTex, int index, const Vec2f& texCoord) {
    int4 texItem = diffuseTextureItems[index];
    float4 colorByte = tex2D<float4>((TEXTURE_OBJECT)diffuseTex, texItem.z + texItem.x * texCoord.x, texItem.w + texItem.y * texCoord.y);
    return Vec3f(colorByte.x, colorByte.y, colorByte.z);
}

DEVICE_INLINE float hammersley(int i, int num) {
    return (i + 0.5f) / num;
}

DEVICE_INLINE float halton(int i, int base) {
    float r = 0;
    float f = 1.0f / base;
    int j = i;
    while (j > 0) {
        r += f * (j % base);
        j = floor(float(j) / base);
        f /= base;
    }
    return r;
}

DEVICE_INLINE Vec2f hammersley2D(int i, int n) {
    return Vec2f(halton(i, 2), hammersley(i, n));
}

DEVICE_INLINE Vec2f halton2D(int i, int base0 = 2, int base1 = 3) {
    return Vec2f(halton(i, base0), halton(i, base1));
}

DEVICE_INLINE Vec2f sobol2D(int i) {
    Vec2f result;
    // Remaining components by matrix multiplication.
    unsigned int r1 = 0, r2 = 0;
    for (unsigned int v1 = 1U << 31, v2 = 3U << 30; i; i >>= 1) {
        if (i & 1) {
            // Vector addition of matrix column by XOR.
            r1 ^= v1;
            r2 ^= v2 << 1;
        }
        // Update matrix columns.
        v1 |= v1 >> 1;
        v2 ^= v2 >> 1;
    }
    // Map to unit cube [0,1)^2.
    result.x = r1 * (1.0f / (1ULL << 32));
    result.y = r2 * (1.0f / (1ULL << 32));
    return result;
}

DEVICE_INLINE float luminance(const Vec3f & color) {
    return 0.2125f * color.x + 0.7154f * color.y + 0.0721f * color.z;
}

DEVICE_INLINE Vec3f arbitraryNormal(const Vec3f & normal) {
    float dist2 = normal.x * normal.x + normal.y * normal.y;
    if (dist2 > 0.0001f) {
        float invSize = 1.0f / sqrtf(dist2);
        return Vec3f(normal.y * invSize, -normal.x * invSize, 0);
    }
    float invSize = 1.0f / sqrtf(normal.z * normal.z + normal.x * normal.x);
    return Vec3f(-normal.z * invSize, 0, normal.x * invSize);
}

DEVICE_INLINE void rightHandedBase(Vec3f & u, Vec3f & v, const Vec3f & w) {
    v = arbitraryNormal(w);
    u = cross(v, w);
}

DEVICE_INLINE Vec3f hemisphericCos(float & pdf, float h, float r0, float r1) {
    const float sinTheta = sqrtf(1.0f - powf(r1, 2.0f / float(h + 1)));
    const float cosTheta = powf(r1, 1.0f / float(h + 1));
    const float phi = 2.0f * M_PIf * r0;
    pdf = (h + 1) / (2.0f * M_PIf) * powf(cosTheta, h);
    return Vec3f(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
}

DEVICE_INLINE Vec3f glossyRandomVector(float & pdf, const Vec3f & r, float h, float r0, float r1) {
    Vec3f u, v;
    rightHandedBase(u, v, r);
    Vec3f hem = hemisphericCos(pdf, h, r0, r1);
    return normalize(hem.x * u + hem.y * v + hem.z * r);
}

DEVICE_INLINE Vec3f cosineRandomVector(float r0, float r1, const Vec3f & normal) {
    float theta = 2.0f * M_PIf * r1;
    float radius = sqrtf(r0);
    float x = radius * sinf(theta);
    float z = radius * cosf(theta);
    float y = sqrtf(1.0f - x * x - z * z);
    Vec3f u, v;
    rightHandedBase(u, v, normal);
    return x * u + z * v + y * normal;
}

extern "C" GLOBAL void generatePrimaryRays(
    const int sampleIndex,
    Vec3f origin,
    Mat4f screenToWorld,
    Vec2i size,
    float maxDist,
    int * indexToPixel,
    Ray * rays
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Batch size.
    const int numberOfPixels = size.x * size.y;

    // Only valid threads.
    if (taskIndex < numberOfPixels) {

        // Pixel.
        int pixel = indexToPixel[taskIndex];

        // Sample rays using Halton sequence.
        Vec2i sample(pixel % size.x, pixel / size.x);
        Vec2f haltonSample = halton2D(sampleIndex);
        float x = float(sample.x) + haltonSample.x;
        float y = float(sample.y) + haltonSample.y;

        // Compute ray.
        Vec4f nscreenPos;
        nscreenPos.x = 2.0f * x / float(size.x) - 1.0f;
        nscreenPos.y = 2.0f * y / float(size.y) - 1.0f;
        nscreenPos.z = 0.0f;
        nscreenPos.w = 1.0f;
        Vec4f worldPos4D = screenToWorld * nscreenPos;
        Vec3f worldPos = Vec3f(worldPos4D) / worldPos4D.w;

        // Write result.
        Ray ray;
        ray.origin = origin;
        ray.direction = normalize(worldPos - origin);
        ray.tmin = 0.0f;
        ray.tmax = maxDist;
        rays[taskIndex] = ray;

    }

}

extern "C" GLOBAL void generateShadowRays(
    const int batchBegin,
    const int batchSize,
    const int numberOfSamples,
    unsigned int * seeds,
    float lightRadius,
    Vec3f light,
    Ray * inputRays,
    Ray * outputRays,
    RayResult * inputResults,
    int * outputSlotToIndex,
    int * outputIndexToSlot
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Only valid threads.
    if (taskIndex < batchSize) {

        // Seed.
        unsigned int seed = seeds[taskIndex];

        // Input ray.
        Ray inputRay = inputRays[batchBegin + taskIndex];
        RayResult inputResult = inputResults[batchBegin + taskIndex];

        // Pick random offset.
        unsigned int hashA = seed + taskIndex;
        unsigned int hashB = 0x9e3779b9u;
        unsigned int hashC = 0x9e3779b9u;
        jenkinsMix(hashA, hashB, hashC);
        jenkinsMix(hashA, hashB, hashC);
        Vec3f offset((float)hashA * exp2(-32.0f), (float)hashB * exp2(-32.0f), (float)hashC * exp2(-32.0f));

        // Generate each sample.
        int tri = inputResult.id;

        // Ray index.
        int rayIndex = numberOfSamples * taskIndex;

        // Compute origin, backtracking a little bit.
        const float EPSILON = 1.0e-5f;
        Vec3f origin = inputRay.origin + inputRay.direction * fmaxf(inputResult.t - EPSILON, 0.0f);
        if (tri == -1) origin = Vec3f(0.0f);

        for (int i = 0; i < numberOfSamples; ++i) {

            // QMC.
            Vec3f position(sobol2D(i), hammersley(i, numberOfSamples)); // [0,1]
            position += offset; // Cranley-Patterson
            if (position.x >= 1.0f) position.x -= 1.0f;
            if (position.y >= 1.0f) position.y -= 1.0f;
            if (position.z >= 1.0f) position.z -= 1.0f;
            position = 2.0f * position - Vec3f(1.0f); // [-1,1]

            // Target position.
            const Vec3f target = light + lightRadius * position;
            const Vec3f direction = target - origin;

            // Output ray.
            Ray outputRay;
            outputRay.origin = origin;
            outputRay.direction = tri == -1 ? Vec3f(1.0f, 0.0f, 0.0f) : normalize(direction);
            outputRay.tmin = 0.0f;
            outputRay.tmax = tri == -1 ? -1.0f : length(direction);
            outputRays[rayIndex] = outputRay;
            outputSlotToIndex[rayIndex] = rayIndex;
            outputIndexToSlot[rayIndex] = rayIndex;
            ++rayIndex;

        }

    }

}

extern "C" GLOBAL void generateAORays(
    const int batchBegin,
    const int batchSize,
    const int numberOfSamples,
    unsigned int * seeds,
    float maxDist,
    Ray * inputRays,
    Ray * outputRays,
    RayResult * inputResults,
    int * outputSlotToIndex,
    int * outputIndexToSlot,
    Vec3i * vertIndices,
    Vec3f * vertices
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Only valid threads.
    if (taskIndex < batchSize) {

        // Seed.
        unsigned int seed = seeds[taskIndex];

        // Input ray.
        Ray inputRay = inputRays[batchBegin + taskIndex];
        RayResult inputResult = inputResults[batchBegin + taskIndex];

        // Compute origin, backtracking a little bit.
        const float EPSILON = 1.0e-4f;
        Vec3f origin = inputRay.origin + inputRay.direction * fmaxf(inputResult.t - EPSILON, 0.0f);

        // Lookup normal, flipping back-facing directions.
        Vec3f normal(1.0f, 0.0f, 0.0f);
        if (inputResult.id != -1) {
            Vec3i triangle = vertIndices[inputResult.id];
            Vec3f v0 = vertices[triangle.x];
            Vec3f v1 = vertices[triangle.y];
            Vec3f v2 = vertices[triangle.z];
            normal = normalize(Vec3f(cross(v1 - v0, v2 - v0)));
        }

        // Flip normal.
        float dotProd = dot(normal, inputRay.direction);
        if (dotProd > 0.0f) normal = -normal;

        // Construct perpendicular vectors.
        Vec3f na = abs(normal);
        float nm = fmaxf(fmaxf(na.x, na.y), na.z);
        Vec3f perp(normal.y, -normal.x, 0.0f); // assume y is largest
        if (nm == na.z) perp = Vec3f(0.0f, normal.z, -normal.y);
        else if (nm == na.x) perp = Vec3f(-normal.z, 0.0f, normal.x);

        perp = normalize(perp);
        Vec3f biperp = cross(normal, perp);

        // Pick random rotation angle.
        unsigned int hashA = seed + taskIndex;
        unsigned int hashB = 0x9e3779b9u;
        unsigned int hashC = 0x9e3779b9u;
        jenkinsMix(hashA, hashB, hashC);
        jenkinsMix(hashA, hashB, hashC);
        float angle = 2.0f * M_PIf * (float)hashC * exp2f(-32.0f);

        // Construct rotated tangent vectors.
        Vec3f t0 = perp * cosf(angle) + biperp * sinf(angle);
        Vec3f t1 = perp * -sinf(angle) + biperp * cosf(angle);

        // Ray index.
        int rayIndex = numberOfSamples * taskIndex;

        // Generate each sample.
        for (int i = 0; i < numberOfSamples; ++i) {

            // Base-2 Halton sequence for X.
            float x = 0.0f;
            float xadd = 1.0f;
            unsigned int hc2 = i + 1;
            while (hc2 != 0) {
                xadd *= 0.5f;
                if ((hc2 & 1) != 0)
                    x += xadd;
                hc2 >>= 1;
            }

            // Base-3 Halton sequence for Y.
            float y = 0.0f;
            float yadd = 1.0f;
            int hc3 = i + 1;
            while (hc3 != 0) {
                yadd *= 1.0f / 3.0f;
                y += (float)(hc3 % 3) * yadd;
                hc3 /= 3;
            }

            // Warp to a point on the unit hemisphere.
            float angle = 2.0f * M_PIf * y;
            float r = sqrtf(x);
            x = r * cosf(angle);
            y = r * sinf(angle);
            float z = sqrtf(1.0f - x * x - y * y);

            // Output ray.
            Ray outputRay;
            outputRay.origin = origin;
            outputRay.direction = normalize(x * t0 + y * t1 + z * normal);
            outputRay.tmin = 0.0f;
            outputRay.tmax = (inputResult.id == -1) ? -1.0f : maxDist;
            outputRays[rayIndex] = outputRay;
            outputSlotToIndex[rayIndex] = rayIndex;
            outputIndexToSlot[rayIndex] = rayIndex;
            ++rayIndex;

        }

    }

}

extern "C" GLOBAL void generatePathRays(
    const bool russianRoulette,
    const int numberOfInputRays,
    unsigned long long diffuseTex,
    unsigned int * seeds,
    int * numberOfOutputRaysLoc,
    int * inputIndexToPixel,
    int * outputIndexToPixel,
    int * matIndices,
    Vec3i * triangles,
    Vec3f * normals,
    Vec2f * texCoords,
    Ray * inputRays,
    Ray * outputRays,
    RayResult * inputResults,
    Vec3f * decreases
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Thread inde in the warp.
    const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

    // Only valid threads.
    if (taskIndex < numberOfInputRays) {

        // Pixel.
        const int pixel = inputIndexToPixel[taskIndex];

        // Seed.
        unsigned int seed = seeds[taskIndex];

        // Output ray.
        Ray outputRay;
        Vec3f weight;
        bool terminated = true;

        // Input ray.
        const Ray inputRay = inputRays[taskIndex];
        const RayResult inputResult = inputResults[taskIndex];

        // Triangle index.
        int triangleIndex = inputResult.id;

        // Process only active paths.
        if (triangleIndex >= 0) {

            // Normal.
            Vec3i triangle = triangles[triangleIndex];
            Vec3f normal = getSmoothNormal(normals, triangle, inputResult.u, inputResult.v);

            // Check normal.
            float dotProd = dot(normal, inputRay.direction);
            if (dotProd > 0.0f) {
                normal = -normal;
                dotProd = -dotProd;
            }

            // Material.
            int matIndex = matIndices[inputResult.id];
            Material mat = ((Material*)materials)[matIndex];

            // Diffuse texture.
            Vec3f diffuseColor = mat.diffuse;
            if (mat.texIndex >= 0) {
                Vec2f texCoord = getSmoothTexCoord(texCoords, triangle, inputResult.u, inputResult.v);
                diffuseColor = getTexel(diffuseTex, mat.texIndex, texCoord);
            }

            // Materials components.
            Vec3f kd = diffuseColor;
            Vec3f ks = mat.specular;
            float pd = max(max(kd.x, kd.y), kd.z);
            float ps = max(max(ks.x, ks.y), ks.z);
            float h = mat.shininess;

            // Material scale.
            if (pd + ps > 1.0f) {
                float s = 1.0f / (pd + ps);
                kd *= s;
                ks *= s;
                pd *= s;
                ps *= s;
            }

            // Generate random numbers.
            float r0 = randf(seed);
            float r1 = randf(seed);
            float r2 = randf(seed);

            // Russian roulette off => normalize probabilities.
            if (!russianRoulette) {
                float s = 1.0f / (pd + ps);
                pd *= s;
                ps *= s;
            }

            if (r2 <= pd + ps) {
                float pdf;
                float cosTheta;
                float cosAlpha;

                // Reflected ray.
                Vec3f reflectedDirection = normalize(inputRay.direction - (2 * dotProd) * normal);

                // Intersection point.
                Vec3f origin = inputRay.origin + inputRay.direction * inputResult.t;

                // Ray offset.
                const float EPSILON = 1.0e-4f;

                // Glossy distributed ray.
                if (r2 >= pd) {
                    outputRay.direction = glossyRandomVector(pdf, reflectedDirection, mat.shininess, r0, r1);
                    outputRay.origin = origin + outputRay.direction * EPSILON;
                    outputRay.tmax = MAX_FLOAT;
                    cosTheta = dot(normal, outputRay.direction);
                    cosAlpha = dot(reflectedDirection, outputRay.direction);
                    Vec3f brdf = (h + 2.0f) / (2.0f * M_PIf) * ks * powf(cosAlpha, h);
                    weight = brdf * cosTheta / pdf / (pd + ps);
                }

                // Cosine distributed ray.
                if (r2 < pd || (r2 >= pd && (cosTheta < 0.0f || pdf < 0.01f || cosAlpha < 0.0f))) {
                    outputRay.direction = cosineRandomVector(r0, r1, normal);
                    outputRay.origin = origin + outputRay.direction * EPSILON;
                    outputRay.tmax = MAX_FLOAT;
                    cosTheta = dot(normal, outputRay.direction);
                    cosAlpha = dot(reflectedDirection, outputRay.direction);
                    pdf = cosTheta / M_PIf;
                    Vec3f brdf = kd / M_PIf;
                    weight = brdf * cosTheta / pdf / (pd + ps);
                }

                // Not terminated.
                terminated = false;

            }

            // Save seed.
            seeds[taskIndex] = seed;

        }

        // Warp wide prefix scan of active paths.
        const unsigned int prefixScanMask = __ballot(!terminated);
        const int activePathsInWarp = __popc(prefixScanMask);
        const int rayOffset = __popc(prefixScanMask & ((1u << warpThreadIndex) - 1));

        // Add count of new tasks to the global counter.
        int warpOffset = 0;
        if (warpThreadIndex == 0)
            warpOffset = atomicAdd(numberOfOutputRaysLoc, activePathsInWarp);

        // Exchange offset between threads.
        const int rayIndex = __shfl(warpOffset, 0) + rayOffset;

        // Write output ray.
        if (!terminated) {
            outputRays[rayIndex] = outputRay;
            outputIndexToPixel[rayIndex] = pixel;
            decreases[pixel] *= weight;
        }

    }

}

extern "C" GLOBAL void initSeeds(
    const int numberOfPixels,
    const int frameIndex,
    unsigned int * seeds
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (taskIndex < numberOfPixels) {

        // Init seed.
        seeds[taskIndex] = tea<16>(taskIndex, frameIndex);

    }

}
