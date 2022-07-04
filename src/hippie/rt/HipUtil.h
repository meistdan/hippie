#ifndef _HIP_UTIL_H_
#define _HIP_UTIL_H_

#include "Globals.h"
#include "util/Math.h"

//---------------------------------------------------------------------------
// FLOAT ATOMIC MIN/MAX
//---------------------------------------------------------------------------

#ifdef __CUDACC__
DEVICE_INLINE void atomicMin(float * ptr, float value) {
    unsigned int curr = atomicAdd((unsigned int*)ptr, 0);
    while (value < __int_as_float(curr)) {
        unsigned int prev = curr;
        curr = atomicCAS((unsigned int*)ptr, curr, __float_as_int(value));
        if (curr == prev) break;
    }
}

DEVICE_INLINE void atomicMax(float * ptr, float value) {
    unsigned int curr = atomicAdd((unsigned int*)ptr, 0);
    while (value > __int_as_float(curr)) {
        unsigned int prev = curr;
        curr = atomicCAS((unsigned int*)ptr, curr, __float_as_int(value));
        if (curr == prev) break;
    }
}
#endif

//---------------------------------------------------------------------------
// MORTON CODE
//---------------------------------------------------------------------------

DEVICE_INLINE unsigned int mortonCode(unsigned int x, unsigned int y, unsigned int z) {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y << 8)) & 0x0300F00F;
    y = (y | (y << 4)) & 0x030C30C3;
    y = (y | (y << 2)) & 0x09249249;
    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z << 8)) & 0x0300F00F;
    z = (z | (z << 4)) & 0x030C30C3;
    z = (z | (z << 2)) & 0x09249249;
    return x | (y << 1) | (z << 2);
}

DEVICE_INLINE unsigned int mortonCode(const float3 & centroid, const float3 & sceneExtentInv) {
    unsigned int x = (centroid.x * sceneExtentInv.x) * 1023u;
    unsigned int y = (centroid.y * sceneExtentInv.y) * 1023u;
    unsigned int z = (centroid.z * sceneExtentInv.z) * 1023u;
    return mortonCode(x, y, z);
}

DEVICE_INLINE unsigned int mortonCode(const Vec3f & centroid, const Vec3f & sceneExtentInv) {
    unsigned int x = (centroid.x * sceneExtentInv.x) * 1023u;
    unsigned int y = (centroid.y * sceneExtentInv.y) * 1023u;
    unsigned int z = (centroid.z * sceneExtentInv.z) * 1023u;
    return mortonCode(x, y, z);
}

DEVICE_INLINE unsigned int mortonCode(const Vec3f & centroid) {
    unsigned int x = centroid.x * 1023u;
    unsigned int y = centroid.y * 1023u;
    unsigned int z = centroid.z * 1023u;
    return mortonCode(x, y, z);
}

DEVICE_INLINE unsigned long long mortonCode64(unsigned int x, unsigned int y, unsigned int z) {
    unsigned int loX = x & 1023u;
    unsigned int loY = y & 1023u;
    unsigned int loZ = z & 1023u;
    unsigned int hiX = x >> 10u;
    unsigned int hiY = y >> 10u;
    unsigned int hiZ = z >> 10u;
    unsigned long long lo = mortonCode(loX, loY, loZ);
    unsigned long long hi = mortonCode(hiX, hiY, hiZ);
    return (hi << 30) | lo;
}

DEVICE_INLINE unsigned long long mortonCode64(const float3 & centroid, const float3 & sceneExtentInv) {
    unsigned int scale = (1u << 20) - 1;
    unsigned int x = (centroid.x * sceneExtentInv.x) * scale;
    unsigned int y = (centroid.y * sceneExtentInv.y) * scale;
    unsigned int z = (centroid.z * sceneExtentInv.z) * scale;
    return mortonCode64(x, y, z);
}

DEVICE_INLINE unsigned long long mortonCode64(const Vec3f & centroid, const Vec3f & sceneExtentInv) {
    unsigned int scale = (1u << 20) - 1;
    unsigned int x = (centroid.x * sceneExtentInv.x) * scale;
    unsigned int y = (centroid.y * sceneExtentInv.y) * scale;
    unsigned int z = (centroid.z * sceneExtentInv.z) * scale;
    return mortonCode64(x, y, z);
}

DEVICE_INLINE unsigned long long mortonCode64(const Vec3f & centroid) {
    unsigned int scale = (1u << 20) - 1;
    unsigned int x = centroid.x * scale;
    unsigned int y = centroid.y * scale;
    unsigned int z = centroid.z * scale;
    return mortonCode64(x, y, z);
}

//---------------------------------------------------------------------------
// SWAP
//---------------------------------------------------------------------------

template <class T>
DEVICE_INLINE void swap(T & a, T & b) {
    T t = a;
    a = b;
    b = t;
}

template <class T>
DEVICE_INLINE void swapAndAdd(T & a, T & b) {
    T t = a;
    a = b;
    b += t;
}

//---------------------------------------------------------------------------
// SIGN
//---------------------------------------------------------------------------

template <typename T>
DEVICE_INLINE int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

//---------------------------------------------------------------------------
// PREFIX SCAN
//---------------------------------------------------------------------------

// Hillis-Steele warp scan.
template <typename T>
DEVICE_INLINE int warpScan(int warpThreadIndex, T warpSum) {
    T warpValue = warpSum;
    warpValue = __shfl_up(warpSum, 1); if (warpThreadIndex >= 1) warpSum += warpValue;
    warpValue = __shfl_up(warpSum, 2); if (warpThreadIndex >= 2) warpSum += warpValue;
    warpValue = __shfl_up(warpSum, 4); if (warpThreadIndex >= 4) warpSum += warpValue;
    warpValue = __shfl_up(warpSum, 8); if (warpThreadIndex >= 8) warpSum += warpValue;
    warpValue = __shfl_up(warpSum, 16); if (warpThreadIndex >= 16) warpSum += warpValue;
    return warpSum;
}

// Hillis-Steele block scan.
template <int SCAN_BLOCK_THREADS>
DEVICE_INLINE void blockScan(int & blockSum, volatile int * blockCache) {
    volatile int * blockHalfCache = blockCache + SCAN_BLOCK_THREADS;
    const int threadIndex = (int)threadIdx.x;
    blockCache[threadIndex] = 0;
    blockHalfCache[threadIndex] = blockSum;
    for (int i = 1; i < SCAN_BLOCK_THREADS; i <<= 1) {
        __syncthreads();
        blockSum += blockHalfCache[threadIndex - i];
        __syncthreads();
        blockHalfCache[threadIndex] = blockSum;
    }
    __syncthreads();
}

template <>
DEVICE_INLINE void blockScan<1024>(int & blockSum, volatile int * blockCache) {
    volatile int * blockHalfCache = blockCache + 1024;
    const int threadIndex = (int)threadIdx.x;
    blockCache[threadIdx.x] = 0;
    blockHalfCache[threadIdx.x] = blockSum;
    __syncthreads();
    blockSum += blockHalfCache[threadIndex - 1]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
    blockSum += blockHalfCache[threadIndex - 2]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
    blockSum += blockHalfCache[threadIndex - 4]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
    blockSum += blockHalfCache[threadIndex - 8]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
    blockSum += blockHalfCache[threadIndex - 16]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
    blockSum += blockHalfCache[threadIndex - 32]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
    blockSum += blockHalfCache[threadIndex - 64]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
    blockSum += blockHalfCache[threadIndex - 128]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
    blockSum += blockHalfCache[threadIndex - 256]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
    blockSum += blockHalfCache[threadIndex - 512]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
}

//---------------------------------------------------------------------------
// INTERPOLATION
//---------------------------------------------------------------------------

DEVICE_INLINE Vec3f getSmoothNormal(Vec3f * normals, const Vec3i & triangle, float u, float v) {
    Vec3f n0 = normals[triangle.x];
    Vec3f n1 = normals[triangle.y];
    Vec3f n2 = normals[triangle.z];
    return normalize(u * n0 + v * n1 + (1.0f - u - v) * n2);
}

DEVICE_INLINE Vec2f getSmoothTexCoord(Vec2f * texCoords, const Vec3i & triangle, float u, float v) {
    Vec2f texCoord0 = texCoords[triangle.x];
    Vec2f texCoord1 = texCoords[triangle.y];
    Vec2f texCoord2 = texCoords[triangle.z];
    Vec2f res = u * texCoord0 + v * texCoord1 + (1.0f - u - v) * texCoord2;
    res.x -= int(res.x); if (res.x < 0.0f) res.x += 1.0f;
    res.y -= int(res.y); if (res.y < 0.0f) res.y += 1.0f;
    return res;
}

//---------------------------------------------------------------------------
// RNG AND HASHING
//---------------------------------------------------------------------------

DEVICE_INLINE void jenkinsMix(unsigned int & a, unsigned int & b, unsigned int & c) {
    a -= b; a -= c; a ^= (c >> 13);
    b -= c; b -= a; b ^= (a << 8);
    c -= a; c -= b; c ^= (b >> 13);
    a -= b; a -= c; a ^= (c >> 12);
    b -= c; b -= a; b ^= (a << 16);
    c -= a; c -= b; c ^= (b >> 5);
    a -= b; a -= c; a ^= (c >> 3);
    b -= c; b -= a; b ^= (a << 10);
    c -= a; c -= b; c ^= (b >> 15); // ~36 instructions
}

DEVICE_INLINE unsigned int lcg(unsigned int & seed) {
    const unsigned int LCG_A = 1103515245u;
    const unsigned int LCG_C = 12345u;
    const unsigned int LCG_M = 0x00FFFFFFu;
    seed = (LCG_A * seed + LCG_C);
    return seed & LCG_M;
}

DEVICE_INLINE float randf(unsigned int & seed) {
    return ((float)lcg(seed) / (float)0x01000000);
}

template<unsigned int N>
DEVICE unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

//---------------------------------------------------------------------------
// BARRIER
//---------------------------------------------------------------------------

//DEVICE_INLINE void barrier(const int blocks, int & barrierThreshold) {
//	//__threadfence();
//	__syncthreads();
//	if (threadIdx.x == 0) {
//		atomicAdd(&barrierCount, 1);
//		while (atomicCAS(&barrierCount, barrierThreshold, barrierThreshold) != barrierThreshold);
//		barrierThreshold += blocks;
//	}
//	//__threadfence();
//	__syncthreads();
//}

#endif /* _HIP_UTIL_H_ */
