/**
 * \file	LBVHBuilderKernels.h
 * \author	Daniel Meister
 * \date	2015/11/27
 * \brief	LBVHBuilder kernels header file.
 */

#ifndef _LBVH_BUILDER_KERNELS_H_
#define _LBVH_BUILDER_KERNELS_H_

#include "Globals.h"
#include "util/Math.h"

#ifdef __KERNELCC__
extern "C" {

    DEVICE int prefixScanOffset;
    CONSTANT float sceneBox[6];

    GLOBAL void setupBoxes(
        const int numberOfTriangles,
        int * triangleIndices,
        Vec3i * triangles,
        Vec3f * vertices,
        Vec4f * referenceBoxesMin,
        Vec4f * referenceBoxesMax
    );

    GLOBAL void computeMortonCodes30(
        const int numberOfRefrences,
        const int mortonCodeBits,
        unsigned int * mortonCodes,
        int * referenceIndices,
        Vec4f * referenceBoxesMin,
        Vec4f * referenceBoxesMax
    );

    GLOBAL void computeMortonCodes60(
        const int numberOfRefrences,
        const int mortonCodeBits,
        unsigned long long * mortonCodes,
        int * referenceIndices,
        Vec4f * referenceBoxesMin,
        Vec4f * referenceBoxesMax
    );

    GLOBAL  void setupLeaves(
        const int numberOfReferences,
        int * referenceIndices0,
        int * referenceIndices1,
        int * triangleIndices0,
        int * triangleIndices1,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        Vec4f * nodeBoxesMin,
        Vec4f * nodeBoxesMax,
        Vec4f * referenceBoxesMin,
        Vec4f * referenceBoxesMax
    );

    GLOBAL void construct30(
        const int n,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        unsigned int * mortonCodes
    );

    GLOBAL void construct60(
        const int n,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        unsigned long long * mortonCodes
    );

    GLOBAL void refit(
        const int numberOfNodes,
        int * termCounters,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        Vec4f * nodeBoxesMin,
        Vec4f * nodeBoxesMax
    );

}
#endif

#endif /* _LBVH_BUILDER_KERNELS_H_ */
