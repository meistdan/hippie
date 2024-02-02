/**
 * \file	InsertionBuilderKernels.h
 * \author	Daniel Meister
 * \date	2017/02/07
 * \brief	InsertionBuilder kernels header file.
 */

#ifndef _INSERTION_BUILDER_KERNELS_H_
#define _INSERTION_BUILDER_KERNELS_H_

#include "Globals.h"
#include "util/Math.h"

#define REDUCTION_BLOCK_THREADS 256
#define INSERTION_LOG 1

#ifdef __KERNELCC__
extern "C" {
 
    DEVICE int foundNodes;
    DEVICE int insertedNodes;
    DEVICE float cost;

    GLOBAL void findBestNode(
        const int numberOfNodes,
        const int numberOfReferences,
        const int mod,
        const int remainder,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        int * outNodeIndices,
        float * areaReductions,
        Vec4f * nodeBoxesMin,
        Vec4f * nodeBoxesMax
    );

    GLOBAL void lockNodes(
        const int numberOfNodes,
        const int numberOfReferences,
        const int mod,
        const int remainder,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        int * outNodeIndices,
        float * areaReductions,
        unsigned long long * locks
    );

    GLOBAL void checkLocks(
        const int numberOfNodes,
        const int numberOfReferences,
        const int mod,
        const int remainder,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        int * outNodeIndices,
        float * areaReductions,
        unsigned long long * locks
    );

    GLOBAL void reinsert(
        const int numberOfNodes,
        const int mod,
        const int remainder,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        int * outNodeIndices,
        float * areaReductions
    );

    GLOBAL void computeCost(
        const int numberOfNodes,
        const int numberOfReferences,
        const float sceneBoxArea,
        const float ct,
        const float ci,
        Vec4f * nodeBoxesMin,
        Vec4f * nodeBoxesMax
    );

}
#endif

#endif /* _INSERTION_BUILDER_KERNELS_H_ */
