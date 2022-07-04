/**
  * \file	RayBufferKernels.cu
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	RayGen kernels source file.
  */

#include "rt/ray/RayBufferKernels.h"
#include "rt/HipUtil.h"
#include "util/AABB.h"

DEVICE float intersect(const Ray & ray, const AABB & box) {

    float txmin, txmax, tymin, tymax, tzmin, tzmax;

    // X coordinates.
    if (ray.direction.x >= 0.0f) {
        txmin = (box.mn.x - ray.origin.x) / ray.direction.x;
        txmax = (box.mx.x - ray.origin.x) / ray.direction.x;
    }
    else {
        txmin = (box.mx.x - ray.origin.x) / ray.direction.x;
        txmax = (box.mn.x - ray.origin.x) / ray.direction.x;
    }

    // Y coordinates.
    if (ray.direction.y >= 0.0f) {
        tymin = (box.mn.y - ray.origin.y) / ray.direction.y;
        tymax = (box.mx.y - ray.origin.y) / ray.direction.y;
    }
    else {
        tymin = (box.mx.y - ray.origin.y) / ray.direction.y;
        tymax = (box.mn.y - ray.origin.y) / ray.direction.y;
    }

    // Z coordinates.
    if (ray.direction.z >= 0.0f) {
        tzmin = (box.mn.z - ray.origin.z) / ray.direction.z;
        tzmax = (box.mx.z - ray.origin.z) / ray.direction.z;
    }
    else {
        tzmin = (box.mx.z - ray.origin.z) / ray.direction.z;
        tzmax = (box.mn.z - ray.origin.z) / ray.direction.z;
    }

    float tmin = fmaxf(tzmin, fmaxf(txmin, tymin));
    float tmax = fminf(tzmax, fminf(txmax, tymax));

    if (tmin > tmax)
        return -1.0f;

    // Shorten length a bit to prevent numerical errors.
    const float EPS = 1.0e-3f;
    return (1.0f - EPS) * (tmin >= 0.0f ? tmin : tmax);

}

extern "C" GLOBAL void computeMortonCodesTwoPoint32(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned int * mortonCodes,
    int * rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Ray length.
        float rayLength = globalRayLength;
#if REAL_RAY_LENGTH
        rayLength = fmaxf(results[rayIndex].t, 0.0f);
#endif
        rayLength = fminf(rayLength, fmaxf(ray.tmax, 0.0f));
        rayLength = fminf(rayLength, intersect(ray, _raysBoundingBox));

        // Point for Morton codes.
        Vec3f a = ray.origin;
        Vec3f b = (ray.origin + rayLength * ray.direction);

        // Morton codes.
        Vec3f scale = 1.0f / _raysBoundingBox.diagonal();
        a = (a - _raysBoundingBox.mn) * scale;
        b = (b - _raysBoundingBox.mn) * scale;

        Vec3i ia = a * 32767.0f; //15b/dim
        Vec3i ib = b * 32767.0f;  //15b/dim

        unsigned long long mortonCode = 0;

        for (int i = 14; i >= 4; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (6 * i - 21); // max 63
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (6 * i - 22); // max 62
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (6 * i - 23); // max 61
        }

        for (int i = 14; i >= 5; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (6 * i - 24); // max 60
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (6 * i - 25); // max 59
            mortonCode |= (unsigned long long)((ib.z >> i) & 1) << (6 * i - 26); // max 58
        }
        mortonCode |= (unsigned long long)(ib.x & 1); // max 0

        // Output key.
        mortonCodes[rayIndex] = (unsigned int)(mortonCode >> (64 - mortonCodeBits));
        rayIndices[rayIndex] = rayIndex;

    }

}

extern "C" GLOBAL void computeMortonCodesAila32(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned int * mortonCodes,
    int * rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);
        Vec3f b = (normalize(ray.direction) + 1.0f) * 0.5f;

        Vec3i ia = a * 8191.0f; //13b/dim
        Vec3i ib = b * 8191.0f;  //13b/dim

        unsigned long long mortonCode = 0;

        mortonCode |= (unsigned long long)((ia.x >> 12) & 1) << (63); // max 63
        mortonCode |= (unsigned long long)((ia.y >> 12) & 1) << (62); // max 62
        mortonCode |= (unsigned long long)((ia.z >> 12) & 1) << (61); // max 61

        mortonCode |= (unsigned long long)((ia.x >> 11) & 1) << (60); // max 60
        mortonCode |= (unsigned long long)((ia.y >> 11) & 1) << (59); // max 59
        mortonCode |= (unsigned long long)((ia.z >> 11) & 1) << (58); // max 58

        mortonCode |= (unsigned long long)((ia.x >> 10) & 1) << (57); // max 57
        mortonCode |= (unsigned long long)((ia.y >> 10) & 1) << (56); // max 56
        mortonCode |= (unsigned long long)((ia.z >> 10) & 1) << (55); // max 55

        for (int i = 9; i >= 1; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (6 * i + 0); // max 54
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (6 * i - 1); // max 53
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (6 * i - 2); // max 52
        }

        for (int i = 12; i >= 4; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (6 * i - 21); // max 51
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (6 * i - 22); // max 50
            mortonCode |= (unsigned long long)((ib.z >> i) & 1) << (6 * i - 23); // max 49
        }

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;

    }
}

extern "C" GLOBAL void computeMortonCodesParaboloid32(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned int * mortonCodes,
    int * rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);

        Vec3f nd = normalize(ray.direction);
        Vec3f b;
#if 1
        if (nd.z < 0) {
            float d = (-nd.z + 1.0f);
            b.x = 0.25f * nd.x / d + 0.25f; // b in -1 .. 1 -> 0 .. 0.5f
            b.y = 0.5f  * nd.y / d + 0.5f; // b in -1 .. 1 -> 0 .. 0.5f
        }
        else {
            float d = (nd.z + 1.0f);
            b.x = 0.25f * nd.x / d + 0.75f; // b in -1 .. 1 -> 0.5f .. 1.0f
            b.y = 0.5f * nd.y / d + 0.5f; // b in -1 .. 1 -> 0.5f .. 1.0f
        }
#else
        b.x = atan2(nd.y, nd.x) / (2.0f * M_PIf) + 0.5f;
        b.y = acos(nd.z) / M_PIf;
#endif
        Vec3i ia = a * 8191.0f; //13b/dim
        Vec3i ib = b * 8191.0f;  //13b/dim

        unsigned long long mortonCode = 0;

        mortonCode |= (unsigned long long)((ia.x >> 12) & 1) << (63); // max 63
        mortonCode |= (unsigned long long)((ia.y >> 12) & 1) << (62); // max 62
        mortonCode |= (unsigned long long)((ia.z >> 12) & 1) << (61); // max 61

        mortonCode |= (unsigned long long)((ia.x >> 11) & 1) << (60); // max 60
        mortonCode |= (unsigned long long)((ia.y >> 11) & 1) << (59); // max 59
        mortonCode |= (unsigned long long)((ia.z >> 11) & 1) << (58); // max 58

        mortonCode |= (unsigned long long)((ia.x >> 10) & 1) << (57); // max 57
        mortonCode |= (unsigned long long)((ia.y >> 10) & 1) << (56); // max 56
        mortonCode |= (unsigned long long)((ia.z >> 10) & 1) << (55); // max 55

        mortonCode |= (unsigned long long)((ia.x >> 9) & 1) << (54); // max 57
        mortonCode |= (unsigned long long)((ia.y >> 9) & 1) << (53); // max 56
        mortonCode |= (unsigned long long)((ia.z >> 9) & 1) << (52); // max 55

        for (int i = 8; i >= 5; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (5 * i + 11); // max 51
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (5 * i + 10); // max 50
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (5 * i + 9); // max 49
        }

        for (int i = 12; i >= 9; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (5 * i - 12); // max 48
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (5 * i - 13); // max 47
        }

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;

    }

}

extern "C" GLOBAL void computeMortonCodesOctahedron32(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned int * mortonCodes,
    int * rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);

        Vec3f nd = normalize(ray.direction);
        Vec3f b;

        Vec3f absD = abs(nd);
        float fact = 1.0f / (absD.x + absD.y + absD.z);

        if (nd.z >= 0.0f) {
            b.x = nd.x * fact;
            b.y = nd.y * fact;
        }
        else {
            b.x = (1.0f - absD.y * fact) * ((nd.x <= 0.0f) ? -1.0f : 1.0f);
            b.y = (1.0f - absD.x * fact) * ((nd.y <= 0.0f) ? -1.0f : 1.0f);
        }
        b.x = (b.x + 1.0f) * 0.5f;
        b.y = (b.y + 1.0f) * 0.5f;

        Vec3i ia = a * 8191.0f; //13b/dim
        Vec3i ib = b * 8191.0f;  //13b/dim

        unsigned long long mortonCode = 0;

        mortonCode |= (unsigned long long)((ia.x >> 12) & 1) << (63); // max 63
        mortonCode |= (unsigned long long)((ia.y >> 12) & 1) << (62); // max 62
        mortonCode |= (unsigned long long)((ia.z >> 12) & 1) << (61); // max 61

        mortonCode |= (unsigned long long)((ia.x >> 11) & 1) << (60); // max 60
        mortonCode |= (unsigned long long)((ia.y >> 11) & 1) << (59); // max 59
        mortonCode |= (unsigned long long)((ia.z >> 11) & 1) << (58); // max 58

        mortonCode |= (unsigned long long)((ia.x >> 10) & 1) << (57); // max 57
        mortonCode |= (unsigned long long)((ia.y >> 10) & 1) << (56); // max 56
        mortonCode |= (unsigned long long)((ia.z >> 10) & 1) << (55); // max 55

        mortonCode |= (unsigned long long)((ia.x >> 9) & 1) << (54); // max 57
        mortonCode |= (unsigned long long)((ia.y >> 9) & 1) << (53); // max 56
        mortonCode |= (unsigned long long)((ia.z >> 9) & 1) << (52); // max 55

        for (int i = 8; i >= 5; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (5 * i + 11); // max 51
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (5 * i + 10); // max 50
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (5 * i + 9); // max 49
        }

        for (int i = 12; i >= 9; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (5 * i - 12); // max 48
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (5 * i - 13); // max 47
        }

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;

    }

}

extern "C" GLOBAL void computeMortonCodesOrigin32(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned int* mortonCodes,
    int* rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);
        Vec3i ia = a * 8388607.0f; //23b/dim

        unsigned long long mortonCode = 0;
        for (int i = 22; i >= 2; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (3 * i - 3); // max 63
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (3 * i - 4); // max 62
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (3 * i - 5); // max 61
        }
        mortonCode |= (unsigned long long)(ia.x & 1); // max 0

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;

    }

}

extern "C" GLOBAL void computeMortonCodesCosta32(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned int * mortonCodes,
    int * rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);

        Vec3f nd = normalize(ray.direction);
        Vec3f b;
        b.x = atan2(nd.y, nd.x) / (2.0f * M_PIf) + 0.5f;
        b.y = acos(nd.z) / M_PIf;

        Vec3i ia = a * 8191.0f; //13b/dim
        Vec3i ib = b * 8191.0f;  //13b/dim

        unsigned long long mortonCode = 0;

        for (int i = 12; i >= 9; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (2 * i + 39); // max 63
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (2 * i + 38); // max 62
        }

        for (int i = 12; i >= 5; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (3 * i + 19); // max 55
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (3 * i + 18); // max 54
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (3 * i + 17); // max 53
        }

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;

    }

}

extern "C" GLOBAL void computeMortonCodesReis32(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned int* mortonCodes,
    int* rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);
        Vec3f nd = normalize(ray.direction);
        Vec3f b;
        b.x = atan2(nd.y, nd.x) / (2.0f * M_PIf) + 0.5f;
        b.y = acos(nd.z) / M_PIf;

        Vec3i ia = a * 255.0f; //8b/dim
        Vec3i ib = b * 255.0f;  //8b/dim

        unsigned int mortonCode = 0;

        for (int i = 7; i >= 1; --i) {
            mortonCode |= ((ia.x >> i) & 1) << (3 * i + 10); // max 31
            mortonCode |= ((ia.y >> i) & 1) << (3 * i + 9); // max 30
            mortonCode |= ((ia.z >> i) & 1) << (3 * i + 8); // max 29
        }
        mortonCode |= (ia.x & 1) << (10); // max 10

        for (int i = 7; i >= 3; --i) {
            mortonCode |= ((ib.x >> i) & 1) << (2 * i - 5); // max 9
            mortonCode |= ((ib.y >> i) & 1) << (2 * i - 6); // max 8
        }

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (32 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;

    }

}

extern "C" GLOBAL void computeMortonCodesTwoPoint64(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned long long * mortonCodes,
    int * rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Ray length.
        float rayLength = globalRayLength;
#if REAL_RAY_LENGTH
        rayLength = fmaxf(results[rayIndex].t, 0.0f);
#endif
        rayLength = fminf(rayLength, fmaxf(ray.tmax, 0.0f));
        rayLength = fminf(rayLength, intersect(ray, _raysBoundingBox));

        // Point for Morton codes.
        Vec3f a = ray.origin;
        Vec3f b = (ray.origin + rayLength * ray.direction);

        // Morton codes.
        Vec3f scale = 1.0f / _raysBoundingBox.diagonal();
        a = (a - _raysBoundingBox.mn) * scale;
        b = (b - _raysBoundingBox.mn) * scale;

        Vec3i ia = a * 8191.0f; //13b/dim
        Vec3i ib = b * 8191.0f;  //13b/dim

        unsigned long long mortonCode = 0;

        for (int i = 12; i >= 2; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (6 * i - 9); // max 63
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (6 * i - 10); // max 62
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (6 * i - 11); // max 61
        }

        for (int i = 12; i >= 3; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (6 * i - 12); // max 60
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (6 * i - 13); // max 59
            mortonCode |= (unsigned long long)((ib.z >> i) & 1) << (6 * i - 14); // max 58
        }
        mortonCode |= (unsigned long long)(ib.x & 1); // max 0

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;

    }

}

extern "C" GLOBAL void computeMortonCodesAila64(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned long long * mortonCodes,
    int * rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);
        Vec3f b = (normalize(ray.direction) + 1.0f) * 0.5f;

        Vec3i ia = a * 8191.0f; //13b/dim
        Vec3i ib = b * 8191.0f;  //13b/dim

        unsigned long long mortonCode = 0;

        mortonCode |= (unsigned long long)((ia.x >> 12) & 1) << (63); // max 63
        mortonCode |= (unsigned long long)((ia.y >> 12) & 1) << (62); // max 62
        mortonCode |= (unsigned long long)((ia.z >> 12) & 1) << (61); // max 61

        mortonCode |= (unsigned long long)((ia.x >> 11) & 1) << (60); // max 60
        mortonCode |= (unsigned long long)((ia.y >> 11) & 1) << (59); // max 59
        mortonCode |= (unsigned long long)((ia.z >> 11) & 1) << (58); // max 58

        mortonCode |= (unsigned long long)((ia.x >> 10) & 1) << (57); // max 57
        mortonCode |= (unsigned long long)((ia.y >> 10) & 1) << (56); // max 56
        mortonCode |= (unsigned long long)((ia.z >> 10) & 1) << (55); // max 55

        for (int i = 9; i >= 1; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (6 * i + 0); // max 54
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (6 * i - 1); // max 53
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (6 * i - 2); // max 52
        }

        for (int i = 12; i >= 4; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (6 * i - 21); // max 51
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (6 * i - 22); // max 50
            mortonCode |= (unsigned long long)((ib.z >> i) & 1) << (6 * i - 23); // max 49
        }

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;
    }
}

extern "C" GLOBAL void computeMortonCodesParaboloid64(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned long long * mortonCodes,
    int * rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);

        Vec3f nd = normalize(ray.direction);
        Vec3f b;
        if (nd.z < 0) {
            float d = (-nd.z + 1.0f);
            b.x = 0.25f * nd.x / d + 0.25f; // b in -1 .. 1 -> 0 .. 0.5f
            b.y = 0.5f  * nd.y / d + 0.5f; // b in -1 .. 1 -> 0 .. 0.5f
        }
        else {
            float d = (nd.z + 1.0f);
            b.x = 0.25f * nd.x / d + 0.75f; // b in -1 .. 1 -> 0.5f .. 1.0f
            b.y = 0.5f * nd.y / d + 0.5f; // b in -1 .. 1 -> 0.5f .. 1.0f
        }

        Vec3i ia = a * 32767.0f; //15b/dim
        Vec3i ib = b * 32767.0f;  //15b/dim

        unsigned long long mortonCode = 0;


        mortonCode |= (unsigned long long)((ia.x >> 14) & 1) << (63); // max 63
        mortonCode |= (unsigned long long)((ia.y >> 14) & 1) << (62); // max 62
        mortonCode |= (unsigned long long)((ia.z >> 14) & 1) << (61); // max 61

        mortonCode |= (unsigned long long)((ia.x >> 13) & 1) << (60); // max 60
        mortonCode |= (unsigned long long)((ia.y >> 13) & 1) << (59); // max 59
        mortonCode |= (unsigned long long)((ia.z >> 13) & 1) << (58); // max 58

        mortonCode |= (unsigned long long)((ia.x >> 12) & 1) << (57); // max 57
        mortonCode |= (unsigned long long)((ia.y >> 12) & 1) << (56); // max 56
        mortonCode |= (unsigned long long)((ia.z >> 12) & 1) << (55); // max 55

        mortonCode |= (unsigned long long)((ia.x >> 11) & 1) << (54); // max 54
        mortonCode |= (unsigned long long)((ia.y >> 11) & 1) << (53); // max 53
        mortonCode |= (unsigned long long)((ia.z >> 11) & 1) << (52); // max 52

        for (int i = 10; i >= 0; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (5 * i + 1); // max 51
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (5 * i + 0); // max 50
            if (i > 0)
                mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (5 * i - 1); // max 49
        }

        for (int i = 14; i >= 5; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (5 * i - 22); // max 48
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (5 * i - 23); // max 47
        }

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;
    }

}

extern "C" GLOBAL void computeMortonCodesOctahedron64(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned long long* mortonCodes,
    int* rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);

        Vec3f nd = normalize(ray.direction);
        Vec3f b;

        Vec3f absD = abs(nd);
        float fact = 1.0f / (absD.x + absD.y + absD.z);

        if (nd.z >= 0.0f) {
            b.x = nd.x * fact;
            b.y = nd.y * fact;
        }
        else {
            b.x = (1.0f - absD.y * fact) * ((nd.x <= 0.0f) ? -1.0f : 1.0f);
            b.y = (1.0f - absD.x * fact) * ((nd.y <= 0.0f) ? -1.0f : 1.0f);
        }
        b.x = (b.x + 1.0f) * 0.5f;
        b.y = (b.y + 1.0f) * 0.5f;

        Vec3i ia = a * 32767.0f; //15b/dim
        Vec3i ib = b * 32767.0f;  //15b/dim

        unsigned long long mortonCode = 0;

        mortonCode |= (unsigned long long)((ia.x >> 14) & 1) << (63); // max 63
        mortonCode |= (unsigned long long)((ia.y >> 14) & 1) << (62); // max 62
        mortonCode |= (unsigned long long)((ia.z >> 14) & 1) << (61); // max 61

        mortonCode |= (unsigned long long)((ia.x >> 13) & 1) << (60); // max 60
        mortonCode |= (unsigned long long)((ia.y >> 13) & 1) << (59); // max 59
        mortonCode |= (unsigned long long)((ia.z >> 13) & 1) << (58); // max 58

        mortonCode |= (unsigned long long)((ia.x >> 12) & 1) << (57); // max 57
        mortonCode |= (unsigned long long)((ia.y >> 12) & 1) << (56); // max 56
        mortonCode |= (unsigned long long)((ia.z >> 12) & 1) << (55); // max 55

        mortonCode |= (unsigned long long)((ia.x >> 11) & 1) << (54); // max 54
        mortonCode |= (unsigned long long)((ia.y >> 11) & 1) << (53); // max 53
        mortonCode |= (unsigned long long)((ia.z >> 11) & 1) << (52); // max 52

        for (int i = 10; i >= 0; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (5 * i + 1); // max 51
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (5 * i + 0); // max 50
            if (i > 0)
                mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (5 * i - 1); // max 49
        }

        for (int i = 14; i >= 5; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (5 * i - 22); // max 48
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (5 * i - 23); // max 47
        }

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;
    }

}

extern "C" GLOBAL void computeMortonCodesOrigin64(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned long long* mortonCodes,
    int* rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);
        Vec3i ia = a * 8388607.0f; //23b/dim

        unsigned long long mortonCode = 0;
        for (int i = 22; i >= 2; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (3 * i - 3); // max 63
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (3 * i - 4); // max 62
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (3 * i - 5); // max 61
        }
        mortonCode |= (unsigned long long)(ia.x & 1); // max 0

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;

    }

}

extern "C" GLOBAL void computeMortonCodesCosta64(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned long long * mortonCodes,
    int * rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);

        Vec3f nd = normalize(ray.direction);
        Vec3f b;
        b.x = atan2(nd.y, nd.x) / (2.0f * M_PIf) + 0.5f;
        b.y = acos(nd.z) / M_PIf;

        Vec3i ia = a * 32767.0f; //15b/dim
        Vec3i ib = b * 32767.0f;  //15b/dim

        unsigned long long mortonCode = 0;

        for (int i = 14; i >= 11; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (2 * i + 35); // max 63
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (2 * i + 34); // max 62
        }

        for (int i = 14; i >= 0; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (3 * i + 13); // max 55
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (3 * i + 12); // max 54
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (3 * i + 11); // max 53
        }

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;

    }

}

extern "C" GLOBAL void computeMortonCodesReis64(
    const int numberOfRays,
    const int mortonCodeBits,
    const float globalRayLength,
    Ray * rays,
    RayResult * results,
    unsigned long long* mortonCodes,
    int* rayIndices
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Rays bounding box.
    AABB _raysBoundingBox = *((AABB*)&raysBoundingBox);

    if (rayIndex < numberOfRays) {

        // Ray.
        Ray ray = rays[rayIndex];

        // Point for Morton codes.
        Vec3f a = (ray.origin - _raysBoundingBox.mn) / (_raysBoundingBox.mx - _raysBoundingBox.mn);

        Vec3f nd = normalize(ray.direction);
        Vec3f b;
        b.x = atan2(nd.y, nd.x) / (2.0f * M_PIf) + 0.5f;
        b.y = acos(nd.z) / M_PIf;

        Vec3i ia = a * 2097151.0f; //21b/dim
        Vec3i ib = b * 2097151.0f; //21b/dim

        unsigned long long mortonCode = 0;

        for (int i = 20; i >= 14; --i) {
            mortonCode |= (unsigned long long)((ia.x >> i) & 1) << (3 * i + 3); // max 63
            mortonCode |= (unsigned long long)((ia.y >> i) & 1) << (3 * i + 2); // max 62
            mortonCode |= (unsigned long long)((ia.z >> i) & 1) << (3 * i + 1); // max 61
        }
        mortonCode |= (unsigned long long)((ia.x >> 13) & 1) << (42); // max 42

        for (int i = 20; i >= 0; --i) {
            mortonCode |= (unsigned long long)((ib.x >> i) & 1) << (2 * i + 1); // max 41
            mortonCode |= (unsigned long long)((ib.y >> i) & 1) << (2 * i + 0); // max 40
        }

        // Output key.
        mortonCodes[rayIndex] = mortonCode >> (64 - mortonCodeBits);
        rayIndices[rayIndex] = rayIndex;

    }

}

extern "C" GLOBAL void reorderRays(
    const int numberOfRays,
    int * rayIndices,
    Ray * inRays,
    Ray * outRays,
    int * inSlotToIndex,
    int * outSlotToIndex,
    int * outIndexToSlot
) {

    // Ray index.
    const int rayIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (rayIndex < numberOfRays) {
        int oldRayIndex = rayIndices[rayIndex];
        int index = inSlotToIndex[oldRayIndex];
        outRays[rayIndex] = inRays[oldRayIndex];
        outSlotToIndex[rayIndex] = index;
        outIndexToSlot[index] = rayIndex;
    }

}
