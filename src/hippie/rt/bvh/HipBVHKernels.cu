/**
 * \file	HipBVHKernels.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	HipBVH kernels soruce file.
 */

#include "rt/bvh/HipBVHKernels.h"
#include "rt/bvh/HipBVHUtil.h"
#include "util/Math.h"

DEVICE_INLINE void woopifyTriangle(const Vec3f & v0, const Vec3f & v1, const Vec3f & v2, Mat4f & m) {
    Vec3f v02 = v0 - v2;
    Vec3f v12 = v1 - v2;
    Vec3f norm = cross(v02, v12);
    m[0][0] = v02.x;
    m[0][1] = v02.y;
    m[0][2] = v02.z;
    m[0][3] = 0.0f;
    m[1][0] = v12.x;
    m[1][1] = v12.y;
    m[1][2] = v12.z;
    m[1][3] = 0.0f;
    m[2][0] = norm.x;
    m[2][1] = norm.y;
    m[2][2] = norm.z;
    m[2][3] = 0.0f;
    m[3][0] = v2.x;
    m[3][1] = v2.y;
    m[3][2] = v2.z;
    m[3][3] = 1.0f;
    m = inverse(m);
}

extern "C" GLOBAL void woopifyTriangles(
    const int numberOfReferences,
    int * triangleIndices,
    Vec3i * triangles,
    Vec3f * vertices,
    Vec4f * triWoops
) {

    // Reference index.
    const int referenceIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Woop's matrix.
    Mat4f im;

    if (referenceIndex < numberOfReferences) {

        // Triangle.
        Vec3i triangle = triangles[triangleIndices[referenceIndex]];
        Vec3f v0 = vertices[triangle.x];
        Vec3f v1 = vertices[triangle.y];
        Vec3f v2 = vertices[triangle.z];

        // Woopify triangle.
        woopifyTriangle(v0, v1, v2, im);
        triWoops[3 * referenceIndex + 0] = Vec4f(im[0][2], im[1][2], im[2][2], -im[3][2]);
        triWoops[3 * referenceIndex + 1] = Vec4f(im[0][0], im[1][0], im[2][0], im[3][0]);
        triWoops[3 * referenceIndex + 2] = Vec4f(im[0][1], im[1][1], im[2][1], im[3][1]);

    }

}

extern "C" GLOBAL void generateColors(
    const int numberOfNodes,
    unsigned int * nodeColors
) {

    // Node index.
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (nodeIndex < numberOfNodes) {
        unsigned int seed = nodeIndex;
        nodeColors[nodeIndex] = floatToByte(Vec3f(randf(seed), randf(seed), randf(seed)));
    }

}

#define DEFINE_COLORIZE_TRIANGLES(HipBVHNode, SUFFIX)                                                           \
extern "C" GLOBAL void colorizeTriangles ## SUFFIX(                                                             \
    const int numberOfNodes,                                                                                    \
    const int numberOfInteriorNodes,                                                                            \
    const int nodeSizeThreashold,                                                                               \
    int * triangleIndices,                                                                                      \
    unsigned int * nodeColors,                                                                                  \
    Vec3f * triangleColors,                                                                                     \
    HipBVHNode * nodes                                                                                          \
) {                                                                                                             \
                                                                                                                \
    /* Leaf index. */                                                                                           \
    const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x + numberOfInteriorNodes;                        \
                                                                                                                \
    if (leafIndex < numberOfNodes) {                                                                            \
                                                                                                                \
        /* Node. */                                                                                             \
        int nodeIndex = leafIndex;                                                                              \
        HipBVHNode node = nodes[leafIndex];                                                                     \
        int parentIndex = node.getParentIndex();                                                                \
        HipBVHNode parent = nodes[parentIndex];                                                                 \
                                                                                                                \
        /* Find root index. */                                                                                  \
        while (parent.getSize() <= nodeSizeThreashold) {                                                        \
            nodeIndex = parentIndex;                                                                            \
            if (nodeIndex == 0)                                                                                 \
                break;                                                                                          \
            parentIndex = parent.getParentIndex();                                                              \
            parent = nodes[parentIndex];                                                                        \
        }                                                                                                       \
                                                                                                                \
        /* Node color. */                                                                                       \
        const Vec3f nodeColor = byteToFloat(nodeColors[nodeIndex]);                                             \
                                                                                                                \
        /* Colorize triangles in a leaf node. */                                                                \
        for (int referenceIndex = node.getBegin(); referenceIndex < node.getEnd(); ++referenceIndex)            \
            triangleColors[triangleIndices[referenceIndex]] = nodeColor;                                        \
                                                                                                                \
    }                                                                                                           \
                                                                                                                \
}

#define DEFINE_COMPUTE_SUM_OF_LEAF_SIZES(HipBVHNode, SUFFIX)                                                    \
extern "C" GLOBAL void computeSumOfLeafSizes ## SUFFIX(                                                         \
    const int numberOfNodes,                                                                                    \
    const int numberOfInteriorNodes,                                                                            \
    HipBVHNode * nodes                                                                                          \
) {                                                                                                             \
                                                                                                                \
    /* Leaf index. */                                                                                           \
    const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x + numberOfInteriorNodes;                        \
                                                                                                                \
    int leafSize = 0;                                                                                           \
    if (leafIndex < numberOfNodes) {                                                                            \
        HipBVHNode node = nodes[leafIndex];                                                                     \
        if (node.isLeaf()) leafSize += node.getSize();                                                          \
    }                                                                                                           \
                                                                                                                \
    /* Cache. */                                                                                                \
    __shared__ volatile int cache[REDUCTION_BLOCK_THREADS];                                                     \
                                                                                                                \
    /* Reduction. */                                                                                            \
    cache[threadIdx.x] = leafSize;                                                                              \
    cache[threadIdx.x] += cache[threadIdx.x ^ 1];                                                               \
    cache[threadIdx.x] += cache[threadIdx.x ^ 2];                                                               \
    cache[threadIdx.x] += cache[threadIdx.x ^ 4];                                                               \
    cache[threadIdx.x] += cache[threadIdx.x ^ 8];                                                               \
    cache[threadIdx.x] += cache[threadIdx.x ^ 16];                                                              \
                                                                                                                \
    __syncthreads();                                                                                            \
    if ((threadIdx.x & 32) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 32];                                 \
                                                                                                                \
    __syncthreads();                                                                                            \
    if ((threadIdx.x & 64) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 64];                                 \
                                                                                                                \
    __syncthreads();                                                                                            \
    if ((threadIdx.x & 128) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 128];                               \
                                                                                                                \
    /* Add to the global sum. */                                                                                \
    if (threadIdx.x == 0) atomicAdd(&sumOfLeafSizes, cache[threadIdx.x]);                                       \
                                                                                                                \
}

#define DEFINE_COMPUTE_LEAF_SIZE_HISTOGRAM(HipBVHNode, SUFFIX)                                                  \
extern "C" GLOBAL void computeLeafSizeHistogram ## SUFFIX(                                                      \
    const int numberOfNodes,                                                                                    \
    const int numberOfInteriorNodes,                                                                            \
    HipBVHNode * nodes                                                                                          \
) {                                                                                                             \
                                                                                                                \
    /* Leaf index. */                                                                                           \
    const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x + numberOfInteriorNodes;                        \
                                                                                                                \
    if (leafIndex < numberOfNodes) {                                                                            \
        HipBVHNode node = nodes[leafIndex];                                                                     \
        int leafSize = node.getSize();                                                                          \
        if (leafSize <= MAX_LEAF_SIZE)                                                                          \
            atomicAdd(&leafSizeHistogram[leafSize], 1);                                                         \
    }                                                                                                           \
                                                                                                                \
}

#define DEFINE_COMPUTE_COST(HipBVHNode, SUFFIX)                                                                 \
extern "C" GLOBAL void computeCost ## SUFFIX(                                                                   \
    const int numberOfNodes,                                                                                    \
    const float sceneBoundingBoxArea,                                                                           \
    const float ct,                                                                                             \
    const float ci,                                                                                             \
    HipBVHNode * nodes                                                                                          \
) {                                                                                                             \
                                                                                                                \
    /* Node index. */                                                                                           \
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;                                                \
                                                                                                                \
    /* Cost. */                                                                                                 \
    float _cost = 0.0f;                                                                                         \
                                                                                                                \
    if (nodeIndex < numberOfNodes) {                                                                            \
                                                                                                                \
        HipBVHNode node = nodes[nodeIndex];                                                                     \
        float P = node.getSurfaceArea() / sceneBoundingBoxArea;                                                 \
                                                                                                                \
        /* Leaf. */                                                                                             \
        if (node.isLeaf()) 	_cost += ci * P * (HipBVHNode::N > 2 ?                                              \
            divCeil(node.getSize(), HipBVHNode::N) * HipBVHNode::N : node.getSize());                           \
                                                                                                                \
        /* Interior node. */                                                                                    \
        else _cost += ct * P;                                                                                   \
                                                                                                                \
    }                                                                                                           \
                                                                                                                \
    /* Cache. */                                                                                                \
    __shared__ volatile float cache[REDUCTION_BLOCK_THREADS];                                                   \
                                                                                                                \
    /* Cost reduction. */                                                                                       \
    cache[threadIdx.x] = _cost;                                                                                 \
    cache[threadIdx.x] += cache[threadIdx.x ^ 1];                                                               \
    cache[threadIdx.x] += cache[threadIdx.x ^ 2];                                                               \
    cache[threadIdx.x] += cache[threadIdx.x ^ 4];                                                               \
    cache[threadIdx.x] += cache[threadIdx.x ^ 8];                                                               \
    cache[threadIdx.x] += cache[threadIdx.x ^ 16];                                                              \
                                                                                                                \
    __syncthreads();                                                                                            \
    if ((threadIdx.x & 32) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 32];                                 \
                                                                                                                \
    __syncthreads();                                                                                            \
    if ((threadIdx.x & 64) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 64];                                 \
                                                                                                                \
    __syncthreads();                                                                                            \
    if ((threadIdx.x & 128) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 128];                               \
                                                                                                                \
    /* Update total cost. */                                                                                    \
    if (threadIdx.x == 0) {                                                                                     \
        atomicAdd(&cost, cache[threadIdx.x]);                                                                   \
    }                                                                                                           \
                                                                                                                \
}

#define DEFINE_REFIT_LEAVES(HipBVHNode, SUFFIX)                                                                 \
extern "C" GLOBAL void refitLeaves ## SUFFIX(                                                                   \
    const int numberOfNodes,                                                                                    \
    const int numberOfInteriorNodes,                                                                            \
    int * triangleIndices,                                                                                      \
    Vec3i* triangles,                                                                                           \
    Vec3f* vertices,                                                                                            \
    HipBVHNode * nodes                                                                                          \
) {                                                                                                             \
                                                                                                                \
    /* Leaf index. */                                                                                           \
    const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x + numberOfInteriorNodes;                        \
                                                                                                                \
    if (leafIndex < numberOfNodes) {                                                                            \
                                                                                                                \
        /* Leaf node. */                                                                                        \
        HipBVHNode leaf = nodes[leafIndex];                                                                     \
                                                                                                                \
        /* Bounding box. */                                                                                     \
        AABB box;                                                                                               \
        for (int i = leaf.getBegin(); i < leaf.getEnd(); ++i) {                                                 \
            Vec3i triangle = triangles[triangleIndices[i]];                                                     \
            Vec3f v0 = vertices[triangle.x];                                                                    \
            Vec3f v1 = vertices[triangle.y];                                                                    \
            Vec3f v2 = vertices[triangle.z];                                                                    \
            box.grow(v0);                                                                                       \
            box.grow(v1);                                                                                       \
            box.grow(v2);                                                                                       \
        }                                                                                                       \
        leaf.setBoundingBox(box);                                                                               \
                                                                                                                \
        /* Update the leaf. */                                                                                  \
        nodes[leafIndex] = leaf;                                                                                \
                                                                                                                \
    }                                                                                                           \
                                                                                                                \
}

#define DEFINE_REFIT_INTERIORS(HipBVHNode, SUFFIX)                                                              \
extern "C" GLOBAL void refitInteriors ## SUFFIX(                                                                \
    const int numberOfNodes,                                                                                    \
    const int numberOfInteriorNodes,                                                                            \
    int * termCounters,                                                                                         \
    HipBVHNode * nodes                                                                                          \
) {                                                                                                             \
                                                                                                                \
    /* Leaf index. */                                                                                           \
    const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x + numberOfInteriorNodes;                        \
                                                                                                                \
    if (leafIndex < numberOfNodes) {                                                                            \
                                                                                                                \
        /* Node. */                                                                                             \
        int nodeIndex = nodes[leafIndex].getParentIndex();                                                      \
        HipBVHNode node = nodes[nodeIndex];                                                                     \
                                                                                                                \
        /* Go up to the root. */                                                                                \
        while (atomicAdd(&termCounters[nodeIndex], 1) >= node.getNumberOfChildren() - 1) {                      \
                                                                                                                \
            /* Sync. global memory writes. */                                                                   \
            __threadfence();                                                                                    \
                                                                                                                \
            /* Update each child's index and bounding box. */                                                   \
            for (int i = 0; i < node.getNumberOfChildren(); ++i) {                                              \
                                                                                                                \
                /* Child box. */                                                                                \
                HipBVHNode childNode = nodes[node.getChildIndex(i)];                                            \
                AABB childBox = childNode.getBoundingBox();                                                     \
                                                                                                                \
                /* Update the node. */                                                                          \
                node.setChildIndex(i, node.getChildIndex(i));                                                   \
                node.setChildBoundingBox(i, childBox);                                                          \
            }                                                                                                   \
                                                                                                                \
            /* update the node. */                                                                              \
            nodes[nodeIndex] = node;                                                                            \
                                                                                                                \
            /* Root. */                                                                                         \
            if (nodeIndex == 0) break;                                                                          \
                                                                                                                \
            /* Go to the parent. */                                                                             \
            nodeIndex = node.getParentIndex();                                                                  \
                                                                                                                \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
                                                                                                                \
}

DEFINE_COLORIZE_TRIANGLES(HipBVHNodeBin, Bin)
DEFINE_COLORIZE_TRIANGLES(HipBVHNodeQuad, Quad)
DEFINE_COLORIZE_TRIANGLES(HipBVHNodeOct, Oct)

DEFINE_COMPUTE_SUM_OF_LEAF_SIZES(HipBVHNodeBin, Bin)
DEFINE_COMPUTE_SUM_OF_LEAF_SIZES(HipBVHNodeQuad, Quad)
DEFINE_COMPUTE_SUM_OF_LEAF_SIZES(HipBVHNodeOct, Oct)

DEFINE_COMPUTE_LEAF_SIZE_HISTOGRAM(HipBVHNodeBin, Bin)
DEFINE_COMPUTE_LEAF_SIZE_HISTOGRAM(HipBVHNodeQuad, Quad)
DEFINE_COMPUTE_LEAF_SIZE_HISTOGRAM(HipBVHNodeOct, Oct)

DEFINE_COMPUTE_COST(HipBVHNodeBin, Bin)
DEFINE_COMPUTE_COST(HipBVHNodeQuad, Quad)
DEFINE_COMPUTE_COST(HipBVHNodeOct, Oct)

DEFINE_REFIT_LEAVES(HipBVHNodeBin, Bin)
DEFINE_REFIT_LEAVES(HipBVHNodeQuad, Quad)
DEFINE_REFIT_LEAVES(HipBVHNodeOct, Oct)

DEFINE_REFIT_INTERIORS(HipBVHNodeBin, Bin)
DEFINE_REFIT_INTERIORS(HipBVHNodeQuad, Quad)
DEFINE_REFIT_INTERIORS(HipBVHNodeOct, Oct)
