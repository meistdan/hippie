/**
 * \file	PLOCBuilderKernels.h
 * \author	Daniel Meister
 * \date	2015/10/29
 * \brief	PLOCBuilder kernels header file.
 */

#ifndef _PLOC_BUILDER_KERNELS_H_
#define _PLOC_BUILDER_KERNELS_H_

#include "Globals.h"
#include "util/Math.h"

#define PLOC_SCAN_BLOCK_THREADS 1024
#define PLOC_REDUCTION_BLOCK_THREADS 256
#define PLOC_GEN_BLOCK_THREADS 256

#ifdef __KERNELCC__
extern "C" {

    DEVICE int prefixScanOffset;

    GLOBAL void generateNeighboursCached(
        const int numberOfClusters,
        const int radius,
        float * neighbourDistances,
        int * neighbourIndices,
        Vec4f * nodeBoxesMin,
        Vec4f * nodeBoxesMax,
        int * nodeIndices
    );

    GLOBAL void generateNeighbours(
        const int numberOfClusters,
        const int radius,
        float * neighbourDistances,
        int * neighbourIndices,
        Vec4f * nodeBoxesMin,
        Vec4f * nodeBoxesMax,
        int * nodeIndices
    );

    GLOBAL void merge(
        const int numberOfClusters,
        const int nodeOffset,
        int * neighbourIndices,
        int * nodeIndices0,
        int * nodeIndices1,
        int * nodeParentIndices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        Vec4f * nodeBoxesMin,
        Vec4f * nodeBoxesMax
    );

    GLOBAL void localPrefixScan(
        const int numberOfClusters,
        int * nodeIndices,
        int * threadOffsets,
        int * blockOffsets
    );

    GLOBAL void globalPrefixScan(
        const int numberOfBlocks,
        int * blockOffsets
    );

    GLOBAL void compact(
        const int numberOfClusters,
        int * nodeIndices0,
        int * nodeIndices1,
        int * blockOffsets,
        int * threadOffsets
    );

}
#endif

#endif /* _PLOC_BUILDER_KERNELS_H_ */
