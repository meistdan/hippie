/**
 * \file	RendererKenrels.cu
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Renderer kernels source file.
 */

#include "rt/HipUtil.h"
#include "rt/renderer/RendererKernels.h"

DEVICE_INLINE Vec3f getTexel(unsigned long long diffuseTex, int index, const Vec2f & texCoord) {
    int4 texItem = diffuseTextureItems[index];
    float4 colorByte = tex2D<float4>((TEXTURE_OBJECT)diffuseTex, texItem.z + texItem.x * texCoord.x, texItem.w + texItem.y * texCoord.y);
    return Vec3f(colorByte.x, colorByte.y, colorByte.z);
}

DEVICE_INLINE Vec3f getBackground(unsigned long long environmentTex, const Vec3f & direction) {
    float d = sqrtf(direction.x * direction.x + direction.y * direction.y);
    float r = d > 0 ? 0.159154943f * acosf(direction.z) / d : 0.0f;
    float4 colorByte = tex2D<float4>((TEXTURE_OBJECT)environmentTex, 0.5f + direction.x * r, -(0.5f + direction.y * r));
    return Vec3f(colorByte.x, colorByte.y, colorByte.z);
}

DEVICE_INLINE Vec3f rainbowColorMapping(float _value) {
    Vec3f color;
    float value = 4.0f*(1.0f - _value);
    if (value < 0.0f)
        value = 0.0f;
    else
        if (value > 4.0f)
            value = 4.0f;
    int band = (int)(value);
    value -= band;
    switch (band) {
    case 0:
        color.x = 1.0f;
        color.y = value;
        color.z = 0.0f;
        break;
    case 1:
        color.x = 1.0f - value;
        color.y = 1.0f;
        color.z = 0.0f;
        break;
    case 2:
        color.x = 0.0f;
        color.y = 1.0f;
        color.z = value;
        break;
    case 3:
        color.x = 0.0f;
        color.y = 1.0f - value;
        color.z = 1.0f;
        break;
    default:
        color.x = value;
        color.y = 0.0f;
        color.z = 1.0f;
        break;
    }
    return color;
}

// Reinhard's tone mapping.
// https://www.cs.utah.edu/~reinhard/cdrom/tonemap.pdf
// https://www.yamedev.net/topic/407348-reinhards-tone-mapping-operator/
DEVICE_INLINE Vec3f toneMappingReinhard(Vec3f color, float keyValue, float whitePoint) {

    // Avg. luminance.
    float avgLuminance = 0.5f;

    // RGB -> XYZ conversion
    //const mat3 RGB2XYZ = mat3(0.5141364, 0.3238786,  0.16036376,
    //						  0.265068,  0.67023428, 0.06409157,
    //						  0.0241188, 0.1228178,  0.84442666); 
    const Mat3f RGB2XYZ = Mat3f(
        0.5141364f, 0.265068f, 0.0241188f,
        0.3238786f, 0.67023428f, 0.1228178f,
        0.16036376f, 0.06409157f, 0.84442666f
    );
    Vec3f XYZ = RGB2XYZ * color;

    // XYZ -> Yxy conversion
    Vec3f Yxy;
    Yxy.x = XYZ.y; // copy luminance Y
    Yxy.y = XYZ.x / (XYZ.x + XYZ.y + XYZ.z); // x = X / (X + Y + Z)
    Yxy.z = XYZ.y / (XYZ.x + XYZ.y + XYZ.z); // y = Y / (X + Y + Z)

    // (Lp) Map average luminance to the middlegrey zone by scaling pixel luminance
    float Lp = Yxy.x * keyValue / avgLuminance;
    // (Ld) Scale all luminance within a displayable range of 0 to 1
    Yxy.x = (Lp * (1.0f + Lp / (whitePoint * whitePoint))) / (1.0f + Lp);

    // Yxy -> XYZ conversion
    XYZ.x = Yxy.x * Yxy.y / Yxy.z;               // X = Y * x / y
    XYZ.y = Yxy.x;                                // copy luminance Y
    XYZ.z = Yxy.x * (1 - Yxy.y - Yxy.z) / Yxy.z;  // Z = Y * (1-x-y) / y

    // XYZ -> RGB conversion
    //const mat3 XYZ2RGB  = mat3(2.5651,-1.1665,-0.3986,
    //						    -1.0217, 1.9777, 0.0439, 
    //						     0.0753, -0.2543, 1.1892);
    const Mat3f XYZ2RGB = Mat3f(
        2.5651f, -1.0217f, 0.0753f,
        -1.1665f, 1.9777f, -0.2543f,
        -0.3986f, 0.0439f, 1.1892f
    );
    color = XYZ2RGB * XYZ;
    return color;
}

DEVICE_INLINE Vec3f toneMapping(const Vec3f & color, float keyValue, float whitePoint) {
    // sigmoid mapping - seems to have not very intuitive parameters
    // simple linear clamp with keyValue / whitePoint (currently data independent)
    float m = keyValue;
    float f = whitePoint - 1.0;
    return color * m + Vec3f(1.0f) * f;
}

extern "C" GLOBAL void reconstructSmooth(
    const int numberOfRays,
    const int numberOfSamples,
    unsigned long long diffuseTex,
    unsigned long long environmentTex,
    int * matIndices,
    Vec3i * triangles,
    Vec3f * normals,
    Vec2f * texCoords,
    Ray * rays,
    RayResult *  results,
    Vec3f light,
    int * indexToPixel,
    Vec4f * pixels,
    Vec3f * decreases
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (taskIndex < numberOfRays) {

        // Pixel.
        int pixel = indexToPixel[taskIndex];

        // Color.
        Vec3f color = Vec3f(pixels[pixel]);

        // Decrease.
        Vec3f decrease = decreases[pixel];

        // Ray result.

        // Ray.
        Ray ray = rays[taskIndex];
        RayResult result = results[taskIndex];

        // Triangle index.
        int triangleIndex = result.id;

        // Miss.
        if (triangleIndex == -1) {
            color += decrease * getBackground(environmentTex, ray.direction) / float(numberOfSamples);
        }

        // Hit.
        else {

            // Normal.
            Vec3i triangle = triangles[triangleIndex];
            Vec3f N = getSmoothNormal(normals, triangle, result.u, result.v);

            // Material.
            int matIndex = matIndices[triangleIndex];
            Material mat = ((Material*)materials)[matIndex];

            // Diffuse texture.
            Vec3f diffuseColor = mat.diffuse;
            if (mat.texIndex >= 0) {
                Vec2f texCoord = getSmoothTexCoord(texCoords, triangle, result.u, result.v);
                diffuseColor = getTexel(diffuseTex, mat.texIndex, texCoord);
            }

            // Colors.
            Vec3f L = normalize(light - (ray.origin + ray.direction * result.t));

            // N * L
            float NdotL = dot(N, L);

            // Diffuse.
            float alpha = fmaxf(NdotL, 0.1f);

            // Specular.
            Vec3f R = normalize(-(2.0f * NdotL * N - L));
            float beta = powf(fmaxf(0.0f, dot(R, ray.direction)), mat.shininess);
            if (mat.shininess < 1.0f)
                beta = 0.0f;

            // Add color.
            color += decrease * (alpha * diffuseColor + beta * mat.specular) / float(numberOfSamples);

        }

        // Update color.
        pixels[pixel] = Vec4f(color, 1.0f);

    }

}

extern "C" GLOBAL void reconstructPseudocolor(
    const int numberOfPixels,
    const int numberOfSamples,
    int * matIndices,
    Vec3i * triangles,
    Vec3f * normals,
    Vec3f * pseudocolors,
    Ray * rays,
    RayResult *  results,
    Vec3f light,
    int * indexToPixel,
    Vec4f * pixels
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (taskIndex < numberOfPixels) {

        // Pixel.
        int pixel = indexToPixel[taskIndex];

        // Color.
        Vec3f color = Vec3f(pixels[pixel]);

        // Ray.
        Ray ray = rays[taskIndex];
        RayResult result = results[taskIndex];

        // Triangle index.
        int triangleIndex = result.id;

        // Miss.
        if (triangleIndex == -1) {
            color += BACKGROUND_COLOR / float(numberOfSamples);
        }

        // Hit.
        else {

            // Normal.
            Vec3i triangle = triangles[triangleIndex];
            Vec3f N = getSmoothNormal(normals, triangle, result.u, result.v);

            // Pseudocolor.
            Vec3f pseudocolor = pseudocolors[triangleIndex];

            // L.
            Vec3f L = normalize(light - (ray.origin + ray.direction * result.t));

            // Diffuse.
            float alpha = fmaxf(dot(N, L), 0.1f);
            color += (alpha * pseudocolor) / float(numberOfSamples);

        }

        // Update color.
        pixels[pixel] = Vec4f(color, 1.0f);

    }

}

extern "C" GLOBAL void reconstructThermal(
    const int numberOfPixels,
    const int numberOfSamples,
    const int threshold,
    int * indexToPixel,
    Vec2i * stats,
    Vec4f * pixels
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (taskIndex < numberOfPixels) {

        // Pixel.
        int pixel = indexToPixel[taskIndex];

        // Color.
        Vec3f color = Vec3f(pixels[pixel]);

        // Stats.
        Vec2i stat = stats[taskIndex];
        float value = fminf(stat.x + stat.y, threshold) / float(threshold);
        //float value = fminf(stat.x, threshold) / float(threshold);
        //float value = fminf(stat.y, threshold) / float(threshold);

        // Rainbow color mapping.
        color += rainbowColorMapping(value) / float(numberOfSamples);

        // Update color.
        pixels[pixel] = Vec4f(color, 1.0f);

    }

}

extern "C" GLOBAL void reconstructShadow(
    const int batchBegin,
    const int batchSize,
    const int numberOfSamples,
    const bool replace,
    RayResult * outputResults,
    int * indexToPixel,
    int * indexToSlot,
    Vec4f * inPixels,
    Vec4f * outPixels
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (taskIndex < batchSize) {

        // Pixel.
        int pixel = indexToPixel[batchBegin + taskIndex];

        // Color.
        Vec3f color = Vec3f(inPixels[pixel]);

        // Shadow factor.
        float shadow = 0.0f;
        for (int i = 0; i < numberOfSamples; ++i) {
            RayResult outputResult = outputResults[indexToSlot[taskIndex * numberOfSamples + i]];
            if (!outputResult.hit())
                shadow += 1.0f;
        }
        shadow /= float(numberOfSamples);

        // Update color.
        if (replace) outPixels[pixel] = Vec4f(color * shadow, 1.0f);
        else outPixels[pixel] += Vec4f(color * shadow, 1.0f);

    }

}

extern "C" GLOBAL void reconstructAO(
    const int batchBegin,
    const int batchSize,
    const int numberOfSamples,
    const bool replace,
    RayResult * outputResults,
    int * indexToPixel,
    int * indexToSlot,
    Vec4f * inPixels,
    Vec4f * outPixels
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (taskIndex < batchSize) {

        // Pixel.
        int pixel = indexToPixel[batchBegin + taskIndex];

        // Color.
        Vec3f color = Vec3f(inPixels[pixel]);

        // AO factor.
        float ao = 0.0f;
        for (int i = 0; i < numberOfSamples; ++i) {
            RayResult outputResult = outputResults[indexToSlot[taskIndex * numberOfSamples + i]];
            if (outputResult.id == -1)
                ao += 1.0f;
        }
        ao /= numberOfSamples;

        // Update color.
        if (replace) outPixels[pixel] = Vec4f(color * ao, 1.0f);
        else outPixels[pixel] += Vec4f(color * ao, 1.0f);

    }

}

extern "C" GLOBAL void interpolateColors(
    const int numberOfPixels,
    const int frameIndex,
    const float keyValue,
    const float whitePoint,
    Vec4f * framePixels,
    Vec4f * pixels
)
{

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (taskIndex < numberOfPixels) {

        // Interpolate color according to the frame index.
        if (frameIndex > 1) {
            Vec3f color0 = Vec3f(pixels[taskIndex]);
            Vec3f color1 = Vec3f(framePixels[taskIndex]);
            float a = 1.0f / float(frameIndex);
            float b = a * (float(frameIndex) - 1.0f);
            Vec3f color = a * color1 + b * color0;
            pixels[taskIndex] = Vec4f(color, 1.0f);
            framePixels[taskIndex] = Vec4f(toneMappingReinhard(Vec3f(color), keyValue, whitePoint), 1.0f);
        }

        // First frame => Just assign the color.
        else {
            Vec4f color = framePixels[taskIndex];
            pixels[taskIndex] = color;
            framePixels[taskIndex] = Vec4f(toneMappingReinhard(Vec3f(color), keyValue, whitePoint), 1.0f);
        }

    }
}

extern "C" GLOBAL void initDecreases(
    const int numberOfPixels,
    Vec3f * decreases
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (taskIndex < numberOfPixels) {

        // Init decrease.
        decreases[taskIndex] = Vec3f(1.0f);

    }

}

extern "C" GLOBAL void countRayHits(
    const int numberOfRays,
    RayResult * rayResults
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Count hits by each thread.
    int _rayHits = 0;
    if (taskIndex < numberOfRays && rayResults[taskIndex].id >= 0)
        ++_rayHits;

    // Perform reduction within the warp.
    __shared__ volatile int cache[HITS_BLOCK_THREADS];
    cache[threadIdx.x] = _rayHits;
    cache[threadIdx.x] += cache[threadIdx.x ^ 1];
    cache[threadIdx.x] += cache[threadIdx.x ^ 2];
    cache[threadIdx.x] += cache[threadIdx.x ^ 4];
    cache[threadIdx.x] += cache[threadIdx.x ^ 8];
    cache[threadIdx.x] += cache[threadIdx.x ^ 16];

    // Perform cacheuction within the block.
    __syncthreads();
    if ((threadIdx.x & 32) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 32];

    __syncthreads();
    if ((threadIdx.x & 64) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 64];

    __syncthreads();
    if ((threadIdx.x & 128) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 128];

    // Accumulate globally.
    if (threadIdx.x == 0) atomicAdd(&rayHits, cache[threadIdx.x]);

}
