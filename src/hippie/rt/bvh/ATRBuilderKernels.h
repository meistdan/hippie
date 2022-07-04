/**
 * \file	ATRBuilderKernels.h
 * \author	Daniel Meister
 * \date	2016/02/11
 * \brief	ATRBuilder kernels header file.
 */

#ifndef _ATR_BUILDER_KERNELS_H_
#define _ATR_BUILDER_KERNELS_H_

#include "rt/ray/Ray.h"

#ifdef __KERNELCC__
extern "C" {

    DEVICE int prefixScanOffset;

    GLOBAL void optimize(
        const int numberOfReferences,
        const int treeletSize,
        const int scheduleSize,
        const int distanceMatrixSize,
        const int gamma,
        const float ci,
        const float ct,
        float * costs,
        float * surfaceAreas,
        float * distanceMatrices,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        float4 * nodeBoxesMin,
        float4 * nodeBoxesMax,
        int * termCounters,
        int * parentIndices,
        int * subtreeReferences,
        int * schedule
    );

    __launch_bounds__(128, 12)
        GLOBAL void optimizeSmall(
            const int numberOfReferences,
            const int treeletSize,
            const int scheduleSize,
            const int distanceMatrixSize,
            const int gamma,
            const float ci,
            const float ct,
            float * costs,
            float * surfaceAreas,
            int * nodeLeftIndices,
            int * nodeRightIndices,
            float4 * nodeBoxesMin,
            float4 * nodeBoxesMax,
            int * termCounters,
            int * parentIndices,
            int * subtreeReferences,
            int * schedule
        );

    GLOBAL void computeSurfaceAreas(
        const int numberOfNodes,
        float * surfaceAreas,
        float4 * nodeBoxesMin,
        float4 * nodeBoxesMax
    );

}
#endif

#endif /* _ATR_BUILDER_KERNELS_H_ */
