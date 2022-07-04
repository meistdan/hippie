/**
  * \file	RayBufferKernels.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	RayBuffer kernels header file.
  */

#ifndef _RAY_BUFFER_KERNELS_H_
#define _RAY_BUFFER_KERNELS_H_

#include "Globals.h"
#include "Ray.h"

#define REAL_RAY_LENGTH 0

#define REDUCTION_BLOCK_THREADS 256

#ifdef __KERNELCC__
extern "C" {

    CONSTANT float raysBoundingBox[6];

    GLOBAL void computeMortonCodesTwoPoint32(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray * rays,
        RayResult * results,
        unsigned int * mortonCodes,
        int * rayIndices
    );

    GLOBAL void computeMortonCodesAila32(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray * rays,
        RayResult * results,
        unsigned int * mortonCodes,
        int * rayIndices
    );

    GLOBAL void computeMortonCodesParaboloid32(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray * rays,
        RayResult * results,
        unsigned int * mortonCodes,
        int * rayIndices
    );

    GLOBAL void computeMortonCodesOctahedron32(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray* rays,
        RayResult* results,
        unsigned int* mortonCodes,
        int* rayIndices
    );

    GLOBAL void computeMortonCodesOrigin32(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray* rays,
        RayResult* results,
        unsigned int* mortonCodes,
        int* rayIndices
    );

    GLOBAL void computeMortonCodesCosta32(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray * rays,
        RayResult * results,
        unsigned int * mortonCodes,
        int * rayIndices
    );

    GLOBAL void computeMortonCodesReis32(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray* rays,
        RayResult* results,
        unsigned int* mortonCodes,
        int* rayIndices
    );

    GLOBAL void computeMortonCodesTwoPoint64(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray * rays,
        RayResult * results,
        unsigned long long * mortonCodes,
        int * rayIndices
    );

    GLOBAL void computeMortonCodesAila64(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray * rays,
        RayResult * results,
        unsigned long long * mortonCodes,
        int * rayIndices
    );

    GLOBAL void computeMortonCodesParaboloid64(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray * rays,
        RayResult * results,
        unsigned long long * mortonCodes,
        int * rayIndices
    );

    GLOBAL void computeMortonCodesOctahedron64(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray* rays,
        RayResult* results,
        unsigned long long* mortonCodes,
        int* rayIndices
    );

    GLOBAL void computeMortonCodesOrigin64(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray* rays,
        RayResult* results,
        unsigned long long* mortonCodes,
        int* rayIndices
    );

    GLOBAL void computeMortonCodesCosta64(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray * rays,
        RayResult * results,
        unsigned long long * mortonCodes,
        int * rayIndices
    );

    GLOBAL void computeMortonCodesReis64(
        const int numberOfRays,
        const int mortonCodeBits,
        const float globalRayLength,
        Ray* rays,
        RayResult* results,
        unsigned long long* mortonCodes,
        int* rayIndices
    );

    GLOBAL void reorderRays(
        const int numberOfRays,
        int * rayIndices,
        Ray * inRays,
        Ray * outRays,
        int * inSlotToIndex,
        int * outSlotToIndex,
        int * outIndexToSlot
    );

}
#endif

#endif /* _RAY_BUFFER_KERNELS_H_ */
