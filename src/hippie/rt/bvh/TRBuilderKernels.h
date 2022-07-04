/**
 * \file	TRBuilderKernels.h
 * \author	Daniel Meister
 * \date	2016/03/14
 * \brief	TRBuilder kernels header file.
 */

#ifndef _TR_BUILDER_KERNELS_H_
#define _TR_BUILDER_KERNELS_H_

#define TR_WEIGHT 0.9f

#ifdef __KERNELCC__
extern "C" {

    __launch_bounds__(128, 12)
    GLOBAL void optimize(
        const int numberOfReferences,
        const int numberOfRounds,
        const int treeletSize,
        const int gamma,
        const float ci,
        const float ct,
        float * nodesSah,
        float * surfaceAreas,
        int * nodeLeftIndices,
        int * nodeRightIndices,
        float4 * nodeBoxesMin,
        float4 * nodeBoxesMax,
        int * termCounters,
        int * parentIndices,
        int * subtreeReferences,
        float * subsetAreas,
        float4 * subsetBoxesMin,
        float4 * subsetBoxesMax,
        int * schedule,
        int * currentInternalNode,
        int * stackNode,
        int * stackSize,
        char * stackMask
    );

    GLOBAL void computeSurfaceAreas(
        const int numberOfNodes,
        float * surfaceAreas,
        float4 * nodeBoxesMin,
        float4 * nodeBoxesMax
    );

}
#endif

#endif /* _TR_BUILDER_KERNELS_H_ */
