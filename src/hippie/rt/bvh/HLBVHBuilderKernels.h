/**
 * \file	HLBVHBuilderKernels.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	HLBVHBuilder kernels header file.
 */

#ifndef _HLBVH_BUILDER_KERNELS_H_
#define _HLBVH_BUILDER_KERNELS_H_

#define MAX_BINS 32

struct HLBVHTask {
    int nodeIndex;
    int parentIndex;
    HOST_DEVICE_INLINE HLBVHTask(void) : nodeIndex(0), parentIndex(0) {}
    HOST_DEVICE_INLINE HLBVHTask(int nodeIndex, int parentIndex) : nodeIndex(nodeIndex), parentIndex(parentIndex) {}
};

#ifdef __KERNELCC__
extern "C" {

    DEVICE int prefixScanOffset;
    CONSTANT float sceneExtentInv[3];

    GLOBAL void computeNodeStates(
        const int numberOfReferences,
        const int mortonCodeBits,
        const int mortonCodeSAHBits,
        int * termCounters,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        int * nodeStates,
        unsigned long long * mortonCodes
    );

    GLOBAL void computeLeafClusterIndices(
        const int numberOfReferences,
        int * leafClusterIndices,
        int * nodeParentIndices,
        int * nodeStates
    );

    GLOBAL void invalidateIntermediateClusters(
        const int numberOfReferences,
        int * termCounters,
        int * leafClusterIndices,
        int * nodeParentIndices,
        int * nodeStates
    );

    GLOBAL void computeNodeOffsets(
        const int numberOfReferences,
        int * nodeOffsets,
        int * nodeStates
    );

    GLOBAL void compact(
        const int numberOfNodes,
        int * nodeOffsets,
        int * nodeStates,
        int * inputParentIndices,
        int * inputLeftIndices,
        int * inputRightIndices,
        int * outputParentIndices,
        int * outputLeftIndices,
        int * outputRightIndices
    );

    GLOBAL void computeClusters(
        const int numberOfNodes,
        const int mortonCodeBits,
        const int mortonCodeSAHBits,
        int * nodeStates,
        int * nodeOffsets,
        int * nodeLeftIndices,
        int * clusterNodeIndices,
        Vec4i * clusterBinIndices,
        unsigned long long * mortonCodes
    );

    GLOBAL void resetBins(
        const int numberOfAllBins,
        Vec4f * binBoxesMin,
        Vec4f * binBoxesMax
    );

    GLOBAL void binClusters(
        const int numberOfTasks,
        const int numberOfBins,
        const int numberOfClusters,
        int * clusterTaskIndices,
        int * clusterNodeIndices,
        Vec4i * clusterBinIndices,
        Vec4f * nodeBoxesMin,
        Vec4f * nodeBoxesMax,
        Vec4f * binBoxesMin,
        Vec4f * binBoxesMax
    );

    GLOBAL void split(
        const int numberOfBins,
        const int inputQueueSize,
        int * outputQueueSizeLoc,
        int * newTaskIndices,
        int * splitIndices,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        Vec4f * nodeBoxesMin,
        Vec4f * nodeBoxesMax,
        Vec4f * binBoxesMin,
        Vec4f * binBoxesMax,
        HLBVHTask * inputQueue,
        HLBVHTask * outputQueue
    );

    GLOBAL void distributeClusters(
        const int numberOfTasks,
        const int numberOfBins,
        const int numberOfClusters,
        int * newTaskIndices,
        int * splitIndices,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        int * clusterTaskIndices,
        int * clusterNodeIndices,
        Vec4i * clusterBinIndices,
        Vec4f* binBoxesMin,
        Vec4f* binBoxesMax,
        HLBVHTask * inputQueue
    );

}
#endif

#endif /* _HLBVH_BUILDER_KERNELS_H_ */
