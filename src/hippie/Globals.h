/**
 * \file	Globals.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	A header file including HIP and GLEW libraries.
 *			The file contains some useful constant, macros and functions.
 */

#ifndef _GLOBALS_H_
#define _GLOBALS_H_

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define __KERNELCC__
#endif

#ifndef __KERNELCC__
#include <type_traits>
#include <hip/hip_runtime_api.h>
#include <GL/glew.h>
#include <QDebug>
#endif

#if _WIN32 || _WIN64
#if _WIN64
#define _64_BIT_
#else
#define _32_BIT_
#endif
#endif

#if __GNUC__
#if __x86_64__ || __ppc64__
#define _64_BIT_
#else
#define _32_BIT_
#endif
#endif

#ifdef __KERNELCC__
#define HOST_DEVICE __device__
#if __CUDACC__
#define HOST_DEVICE_INLINE __device__ __forceinline__
#else
#define HOST_DEVICE_INLINE __device__ inline
#endif
#else
#define HOST_DEVICE __host__
#define HOST_DEVICE_INLINE __host__ inline
#endif

#define GLOBAL __global__
#define CONSTANT __constant__
#define DEVICE __device__
#if __CUDACC__
#define TEXTURE_OBJECT unsigned long long
#define DEVICE_INLINE __device__ __forceinline__
#else
#define TEXTURE_OBJECT hipTextureObject_t
#define DEVICE_INLINE __device__ inline
#endif

#ifndef M_PI
#define M_PI 3.14159265358
#endif
#define M_PIf float(M_PI)

#define LOG_WARP_THREADS 5
#define WARP_THREADS (1 << LOG_WARP_THREADS)
#define MIN_FLOAT (1.175494351e-38f)
#define MAX_FLOAT (3.402823466e+38f)
#define MIN_INT (~0x7FFFFFFF)
#define MAX_INT (0x7FFFFFFF)
#define MIN(a, b) (((b) < (a)) ? (b) : (a))
#define MAX(a, b) (((b) > (a)) ? (b) : (a))
#define roundUpNearest(x, y) ((((x) + (y) - 1) / (y)) * y)
#define divCeil(a, b) (((a) + (b) - 1) / (b))
#define divCeilLog(a, b) (((a) + (1 << (b)) - 1) >> (b))
#define mod2k(a, b) ((a) & ((b) - 1))

#ifndef __KERNELCC__
void checkGLErrors(void);
#endif

HOST_DEVICE_INLINE float bitsToFloat(int val) {
#ifdef __KERNELCC__
    return __int_as_float(val);
#else
    return *(float*)&val;
#endif
}

HOST_DEVICE_INLINE int floatToBits(float val) {
#ifdef __KERNELCC__
    return __float_as_int(val);
#else
    return *(int*)&val;
#endif
}

HOST_DEVICE_INLINE int sumArithmeticSequence(int numberOfElements, int firstElement, int lastElement) {
    return numberOfElements * (firstElement + lastElement) / 2;
}

#endif /* _GLOBALS_H_ */
