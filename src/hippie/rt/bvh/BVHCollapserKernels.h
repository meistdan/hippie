/**
 * \file	BVHCollapserKernels.h
 * \author	Daniel Meister
 * \date	2016/03/15
 * \brief	BVHCollapser kernels header file.
 */

#ifndef _BVH_COLLAPSER_KERNELS_H_
#define _BVH_COLLAPSER_KERNELS_H_

#include "HipBVHNode.h"

#ifdef __KERNELCC__
extern "C" {

    DEVICE int interiorPrefixScanOffset;
    DEVICE int leafPrefixScanOffset;
    DEVICE int prefixScanOffset;

    GLOBAL void computeSizes(
        const int numberOfNodes,
        int * termCounters,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        int * nodeSizes
    );

    GLOBAL void computeNodeStatesAdaptive(
        const int numberOfReferences,
        const float ci,
        const float ct,
        int * termCounters,
        float * nodeCosts,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        int * nodeSizes,
        int * nodeStates,
        Vec4f * nodeBoxesMin,
        Vec4f * nodeBoxesMax
    );

    GLOBAL void computeNodeStates(
        const int numberOfReferences,
        const int maxLeafSize,
        int * termCounters,
        int * nodeParentIndices,
        int * nodeSizes,
        int * nodeStates
    );

    GLOBAL void computeLeafIndices(
        const int numberOfReferences,
        int * leafIndices,
        int * nodeParentIndices,
        int * nodeStates
    );

    GLOBAL void invalidateCollapsedNodes(
        const int numberOfReferences,
        int * termCounters,
        int * leafIndices,
        int * nodeParentIndices,
        int * nodeStates
    );

    GLOBAL void computeNodeOffsets(
        const int taskOffset,
        const int numberOfTasks,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        int * nodeIndices,
        int * nodeOffsets,
        int * nodeStates
    );

    GLOBAL void computeReferenceOffsets(
        const int numberOfNodes,
        int * nodeStates,
        int * nodeSizes,
        int * referenceOffsets
    );

    GLOBAL void reorderTriangleIndices(
        const int numberOfReferences,
        int * nodeLeftIndices,
        int * referenceOffsets,
        int * inputTriangleIndices,
        int * outputTriangleIndices,
        int * leafIndices
    );
    
    #define DECLARE_COMPACT(HipBVHNode, SUFFIX)                             \
    GLOBAL void compact ## SUFFIX(                                      \
        const int numberOfNodes,                                            \
        const int newNumberOfInteriorNodes,                                 \
        int * nodeStates,                                                   \
        int * nodeOffsets,                                                  \
        int * nodeParentIndices,                                            \
        int * nodeLeftIndices,                                              \
        int * nodeRightIndices,                                             \
        int * nodeSizes,                                                    \
        int * triangleOffsets,                                              \
        Vec4f * nodeBoxesMin,                                               \
        Vec4f * nodeBoxesMax,                                               \
        HipBVHNode * nodes                                                  \
    );

    DECLARE_COMPACT(HipBVHNodeBin, Bin)

}
#endif

#endif /* _BVH_COLLAPSER_KERNELS_H_ */