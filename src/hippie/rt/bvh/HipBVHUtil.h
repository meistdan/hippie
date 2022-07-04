/**
 * \file	HipBVHUtil.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	A header file containing useful functions used for building the hierarchy.
 */

#ifndef _HIP_BVH_UTIL_H_
#define _HIP_BVH_UTIL_H_

#include "rt/HipUtil.h"
#include "util/AABB.h"

template <typename T>
DEVICE_INLINE int delta(int i, int j, int n, T * mortonCodes);

template <>
DEVICE_INLINE int delta<unsigned long long>(int i, int j, int n, unsigned long long * mortonCodes) {
    if (j < 0 || j >= n) return -1;
    unsigned long long a = mortonCodes[i];
    unsigned long long b = mortonCodes[j];
    if (a != b) return __clzll(a ^ b);
    else return __clzll(i ^ j) + sizeof(unsigned long long) * 8;
}

template <>
DEVICE_INLINE int delta<unsigned int>(int i, int j, int n, unsigned int * mortonCodes) {
    if (j < 0 || j >= n) return -1;
    unsigned int a = mortonCodes[i];
    unsigned int b = mortonCodes[j];
    if (a != b) return __clz(a ^ b);
    else return __clz(i ^ j) + sizeof(unsigned int) * 8;
}

#endif /* _HIP_BVH_UTIL_H_ */
