/**
 * \file	HipBVHKernels.h
 * \author	Daniel Meister
 * \date	2014/08/19
 * \brief	HipBVH kernels header file.
 */

#ifndef _HIP_BVH_KERNELS_H_
#define _HIP_BVH_KERNELS_H_

#include "HipBVHNode.h"

#define MAX_LEAF_SIZE 32
#define TRIANGLE_ALIGN 4096

#define REDUCTION_BLOCK_THREADS 256
#define REFIT_BLOCK_THREADS 128

#ifdef __KERNELCC__
extern "C" {

    DEVICE int nodesOffset;
    DEVICE int indicesOffset;

    DEVICE int leafSizeHistogram[MAX_LEAF_SIZE + 1];
    DEVICE int sumOfLeafSizes;
    DEVICE float cost;

    GLOBAL void woopifyTriangles(
        const int numberOfReferences,
        int * triangleIndices,
        Vec3i* triangles,
        Vec3f* vertices,
        Vec4f * triWoops
    );

    GLOBAL void generateColors(
        const int numberOfNodes,
        unsigned int * nodeColors
    );

    #define DECLARE_COLORIZE_TRIANGLES(HipBVHNode, SUFFIX)                  \
    GLOBAL void colorizeTriangles ## SUFFIX(                            \
        const int numberOfNodes,                                            \
        const int numberOfInteriorNodes,                                    \
        const int nodeSizeThreashold,                                       \
        int * triangleIndices,                                              \
        unsigned int * nodeColors,                                          \
        Vec3f * triangleColors,                                             \
        HipBVHNode * nodes                                                  \
    );

    #define DECLARE_COMPUTE_SUM_OF_LEAF_SIZES(HipBVHNode, SUFFIX)           \
    GLOBAL void computeSumOfLeafSizes ## SUFFIX(                        \
        const int numberOfNodes,                                            \
        const int numberOfInteriorNodes,                                    \
        HipBVHNode * nodes                                                  \
    );                                                                     

    #define DECLARE_COMPUTE_LEAF_SIZE_HISTOGRAM(HipBVHNode, SUFFIX)         \
    GLOBAL void computeLeafSizeHistogram ## SUFFIX(                     \
        const int numberOfNodes,                                            \
        const int numberOfInteriorNodes,                                    \
        HipBVHNode * nodes                                                  \
    );          

    #define DECLARE_COMPUTE_COST(HipBVHNode, SUFFIX)                        \
    GLOBAL void computeCost ## SUFFIX(                                  \
        const int numberOfNodes,                                            \
        const float sceneBoundingBoxArea,                                   \
        const float ct,                                                     \
        const float ci,                                                     \
        HipBVHNode * nodes                                                  \
    );

    #define DECLARE_REFIT_LEAVES(HipBVHNode, SUFFIX)                        \
    GLOBAL void refitLeaves ## SUFFIX(                                  \
        const int numberOfNodes,                                            \
        const int numberOfInteriorNodes,                                    \
        int * triangleIndices,                                              \
        Vec3i* triangles,                                                   \
        Vec3f* vertices,                                                    \
        HipBVHNode * nodes                                                  \
    );
    
    #define DECLARE_REFIT_INTERIORS(HipBVHNode, SUFFIX)                     \
    GLOBAL void refitInteriors ## SUFFIX(                               \
        const int numberOfNodes,                                            \
        const int numberOfInteriorNodes,                                    \
        int * termCounters,                                                 \
        HipBVHNode * nodes                                                  \
    );

    DECLARE_COLORIZE_TRIANGLES(HipBVHNodeBin, Bin)
    DECLARE_COLORIZE_TRIANGLES(HipBVHNodeQuad, Quad)
    DECLARE_COLORIZE_TRIANGLES(HipBVHNodeOct, Oct)

    DECLARE_COMPUTE_SUM_OF_LEAF_SIZES(HipBVHNodeBin, Bin)
    DECLARE_COMPUTE_SUM_OF_LEAF_SIZES(HipBVHNodeQuad, Quad)
    DECLARE_COMPUTE_SUM_OF_LEAF_SIZES(HipBVHNodeOct, Oct)

    DECLARE_COMPUTE_LEAF_SIZE_HISTOGRAM(HipBVHNodeBin, Bin)
    DECLARE_COMPUTE_LEAF_SIZE_HISTOGRAM(HipBVHNodeQuad, Quad)
    DECLARE_COMPUTE_LEAF_SIZE_HISTOGRAM(HipBVHNodeOct, Oct)

    DECLARE_COMPUTE_COST(HipBVHNodeBin, Bin)
    DECLARE_COMPUTE_COST(HipBVHNodeQuad, Quad)
    DECLARE_COMPUTE_COST(HipBVHNodeOct, Oct)

    DECLARE_REFIT_LEAVES(HipBVHNodeBin, Bin)
    DECLARE_REFIT_LEAVES(HipBVHNodeQuad,Quad)
    DECLARE_REFIT_LEAVES(HipBVHNodeOct, Oct)

    DECLARE_REFIT_INTERIORS(HipBVHNodeBin, Bin)
    DECLARE_REFIT_INTERIORS(HipBVHNodeQuad, Quad)
    DECLARE_REFIT_INTERIORS(HipBVHNodeOct, Oct)

}
#endif

#endif /* _HIP_BVH_KERNELS_H_ */
