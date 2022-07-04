/**
 * \file	BVHCollapserKernels.cu
 * \author	Daniel Meister
 * \date	2016/03/15
 * \brief	BVHCollapser kernels soruce file.
 */

#include "rt/bvh/BVHCollapserKernels.h"
#include "rt/bvh/HipBVHUtil.h"

extern "C" GLOBAL void computeSizes(
    const int numberOfNodes,
    int * termCounters,
    int * nodeParentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    int * nodeSizes
) {

    // Leaf index.
    const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x + (numberOfNodes >> 1);

    if (leafIndex < numberOfNodes) {

        // Leaf of size one.
        nodeSizes[leafIndex] = 1;

        // Node index.
        int nodeIndex = nodeParentIndices[leafIndex];

        // Go up to the root.
        while (atomicAdd(&termCounters[nodeIndex], 1) > 0) {

            // Sync. global memory writes.
            __threadfence();
            
            // Node.
            int nodeLeftIndex = nodeLeftIndices[nodeIndex];
            int nodeRightIndex = nodeRightIndices[nodeIndex];

            // Size.
            int nodeLeftSize = nodeSizes[nodeLeftIndex];
            int nodeRightSize = nodeSizes[nodeRightIndex];
            nodeSizes[nodeIndex] = nodeLeftSize + nodeRightSize;

            // Root.
            if (nodeIndex == 0) break;

            // Go to the parent.
            nodeIndex = nodeParentIndices[nodeIndex];

        }

    }

}

extern "C" GLOBAL void computeNodeStatesAdaptive(
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
) {

    // Reference index.
    const int referenceIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (referenceIndex < numberOfReferences) {

        // Node index.
        int nodeIndex = referenceIndex + numberOfReferences - 1;

        // Box.
        AABB box;
        box.grow(Vec3f(nodeBoxesMin[nodeIndex]));
        box.grow(Vec3f(nodeBoxesMax[nodeIndex]));

        // Cost.
        nodeCosts[nodeIndex] = ci * box.area();
        nodeStates[nodeIndex] = 0;

        // Actual node index.
        nodeIndex = nodeParentIndices[nodeIndex];

        // Go up to the root.
        while (atomicAdd(&termCounters[nodeIndex], 1) > 0) {

            // Sync. global memory writes.
            __threadfence();

            // Box.
            AABB box;
            box.grow(Vec3f(nodeBoxesMin[nodeIndex]));
            box.grow(Vec3f(nodeBoxesMax[nodeIndex]));

            // Node.
            int nodeLeftIndex = nodeLeftIndices[nodeIndex];
            int nodeRightIndex = nodeRightIndices[nodeIndex];
            int nodeSize = nodeSizes[nodeIndex];

            // Cost.
            float area = box.area();
            float cost = ct * area + nodeCosts[nodeLeftIndex] + nodeCosts[nodeRightIndex];
            float costAsLeaf = ci * area * nodeSize;

            // Leaf.
            if (costAsLeaf < cost) {
                nodeCosts[nodeIndex] = costAsLeaf;
                nodeStates[nodeIndex] = 0;
            }

            // Interior.
            else {
                nodeCosts[nodeIndex] = cost;
                nodeStates[nodeIndex] = 1;
            }

            // Root.
            if (nodeIndex == 0) break;

            // Go to the parent.
            nodeIndex = nodeParentIndices[nodeIndex];

        }

    }

}

extern "C" GLOBAL void computeNodeStates(
    const int numberOfReferences,
    const int maxLeafSize,
    int * termCounters,
    int * nodeParentIndices,
    int * nodeSizes,
    int * nodeStates
) {

    // Reference index.
    const int referenceIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (referenceIndex < numberOfReferences) {

        // Node index.
        int nodeIndex = referenceIndex + numberOfReferences - 1;

        // Leaf.
        nodeStates[nodeIndex] = 0;

        // Actual node index.
        nodeIndex = nodeParentIndices[nodeIndex];

        // Go up to the root.
        while (atomicAdd(&termCounters[nodeIndex], 1) > 0) {

            // Leaf.
            if (nodeSizes[nodeIndex] <= maxLeafSize) {
                nodeStates[nodeIndex] = 0;
            }

            // Interior.
            else {
                nodeStates[nodeIndex] = 1;
            }

            // Root.
            if (nodeIndex == 0) break;

            // Go to the parent.
            nodeIndex = nodeParentIndices[nodeIndex];

        }

    }

}

extern "C" GLOBAL void computeLeafIndices(
    const int numberOfReferences,
    int * leafIndices,
    int * nodeParentIndices,
    int * nodeStates
) {

    // Reference index.
    const int referenceIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (referenceIndex < numberOfReferences) {

        // Node.
        int nodeIndex = referenceIndex + numberOfReferences - 1;

        // Find leaf index.
        int leafIndex = nodeIndex;
        int parentIndex = nodeParentIndices[leafIndex];
        int parentState = nodeStates[parentIndex];
        while (parentIndex > 0) {
            if (parentState == 0) leafIndex = parentIndex;
            nodeIndex = parentIndex;
            parentIndex = nodeParentIndices[nodeIndex];
            parentState = nodeStates[parentIndex];
        }

        // Write leaf index.
        leafIndices[referenceIndex] = leafIndex;

    }

}

extern "C" GLOBAL void invalidateCollapsedNodes(
    const int numberOfReferences,
    int * termCounters,
    int * leafIndices,
    int * nodeParentIndices,
    int * nodeStates
) {

    // Reference index.
    const int referenceIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (referenceIndex < numberOfReferences) {

        // Leaf index.
        int leafIndex = leafIndices[referenceIndex];

        // Node index.
        int nodeIndex = referenceIndex + numberOfReferences - 1;

        // Leaf reached.
        if (nodeIndex == leafIndex) return;

        // Invalidate node.
        nodeStates[nodeIndex] = -1;

        // Actual node index.
        nodeIndex = nodeParentIndices[nodeIndex];

        // Go up to the root.
        while (atomicAdd(&termCounters[nodeIndex], 1) > 0) {

            // Leaf reached.
            if (nodeIndex == leafIndex) break;

            // Invalidate node.
            nodeStates[nodeIndex] = -1;

            // Root.
            if (nodeIndex == 0) break;

            // Go to the parent.
            nodeIndex = nodeParentIndices[nodeIndex];

        }

    }

}

extern "C" GLOBAL void computeNodeOffsets(
    const int taskOffset,
    const int numberOfTasks,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    int * nodeIndices,
    int * nodeOffsets,
    int * nodeStates
) {

    // Task index.
    const int taskIndex = taskOffset + blockDim.x * blockIdx.x + threadIdx.x;
    const int taskEnd = (divCeilLog(numberOfTasks, LOG_WARP_THREADS) << LOG_WARP_THREADS) + taskOffset;

    // Warp thread index.
    const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

    if (taskIndex < taskEnd) {

        int nodeIndex = -1;
        int numberOfLeaves = 0;
        int numberOfInteriors = 0;
        int nodeLeftIndex = -1;
        int nodeRightIndex = -1;
        int nodeLeftState = -1;
        int nodeRightState = -1;

        if (taskIndex < taskOffset + numberOfTasks) {

            // Node index.
            nodeIndex = nodeIndices[taskIndex];

            // Children.
            nodeLeftIndex = nodeLeftIndices[nodeIndex];
            nodeRightIndex = nodeRightIndices[nodeIndex];
            nodeLeftState = nodeStates[nodeLeftIndex];
            nodeRightState = nodeStates[nodeRightIndex];

            if (nodeLeftState > 0) ++numberOfInteriors;
            if (nodeLeftState == 0) ++numberOfLeaves;
            if (nodeRightState > 0) ++numberOfInteriors;
            if (nodeRightState == 0) ++numberOfLeaves;

        }

        // Prefix scan.
        int warpSum = warpScan(warpThreadIndex, numberOfInteriors);

        // Add count to the global counter.
        int warpOffset = 0;
        if (warpThreadIndex == 31)
            warpOffset = atomicAdd(&interiorPrefixScanOffset, warpSum);
        warpSum -= numberOfInteriors;

        // Exchange offset between threads.
        warpOffset = __shfl(warpOffset, 31);

        if (taskIndex < taskOffset + numberOfTasks) {
            if (nodeLeftState > 0) {
                nodeOffsets[nodeLeftIndex] = warpOffset + warpSum;
                nodeIndices[warpOffset + warpSum] = nodeLeftIndex;
                ++warpSum;
            }
            if (nodeRightState > 0) {
                nodeOffsets[nodeRightIndex] = warpOffset + warpSum;
                nodeIndices[warpOffset + warpSum] = nodeRightIndex;
            }
        }

        // Prefix scan.
        warpSum = warpScan(warpThreadIndex, numberOfLeaves);

        // Add count to the global counter.
        if (warpThreadIndex == 31)
            warpOffset = atomicAdd(&leafPrefixScanOffset, warpSum);
        warpSum -= numberOfLeaves;

        // Exchange offset between threads.
        warpOffset = __shfl(warpOffset, 31);

        if (taskIndex < taskOffset + numberOfTasks) {
            if (nodeLeftState == 0) nodeOffsets[nodeLeftIndex] = warpOffset + warpSum++;
            if (nodeRightState == 0) nodeOffsets[nodeRightIndex] = warpOffset + warpSum;
        }

    }

}

extern "C" GLOBAL void computeReferenceOffsets(
    const int numberOfNodes,
    int * nodeStates,
    int * nodeSizes,
    int * referenceOffsets
) {

    // Node index.
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int nodeEnd = divCeilLog(numberOfNodes, LOG_WARP_THREADS) << LOG_WARP_THREADS;

    // Warp thread index.
    const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

    if (nodeIndex < nodeEnd) {

        // Node state.
        int nodeState;

        // Leaf size.
        int leafSize = 0;

        // Valid node index.
        if (nodeIndex < numberOfNodes) {
            nodeState = nodeStates[nodeIndex];
            if (nodeState == 0) leafSize = nodeSizes[nodeIndex];
        }

        // Leaf size prefix scan.
        int warpSum = warpScan(warpThreadIndex, leafSize);

        // Add count to the global counter.
        int warpOffset = 0;
        if (warpThreadIndex == 31)
            warpOffset = atomicAdd(&prefixScanOffset, warpSum);
        warpSum -= leafSize;

        // Exchange offset between threads.
        warpOffset = __shfl(warpOffset, 31);

        // Leaf.
        if (nodeIndex < numberOfNodes && nodeState == 0) {
            referenceOffsets[nodeIndex] = warpOffset + warpSum;
        }

    }

}

extern "C" GLOBAL void reorderTriangleIndices(
    const int numberOfReferences,
    int * nodeLeftIndices,
    int * referenceOffsets,
    int * inputTriangleIndices,
    int * outputTriangleIndices,
    int * leafIndices
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (taskIndex < numberOfReferences) {

        // Node index.
        const int nodeIndex = taskIndex + numberOfReferences - 1;

        // Reference index.
        const int referenceIndex = nodeLeftIndices[nodeIndex];

        // Write triangle index.
        const int leafIndex = leafIndices[taskIndex];
        const int referenceOffset = atomicAdd(&referenceOffsets[leafIndex], 1);
        outputTriangleIndices[referenceOffset] = inputTriangleIndices[referenceIndex];

    }

}

#define DEFINE_COMPACT(HipBVHNode, SUFFIX)                                                                                  \
extern "C" GLOBAL void compact ## SUFFIX(                                                                                   \
    const int numberOfNodes,                                                                                                \
    const int newNumberOfInteriorNodes,                                                                                     \
    int * nodeStates,                                                                                                       \
    int * nodeOffsets,                                                                                                      \
    int * nodeParentIndices,                                                                                                \
    int * nodeLeftIndices,                                                                                                  \
    int * nodeRightIndices,                                                                                                 \
    int * nodeSizes,                                                                                                        \
    int * triangleOffsets,                                                                                                  \
    Vec4f * nodeBoxesMin,                                                                                                   \
    Vec4f * nodeBoxesMax,                                                                                                   \
    HipBVHNode * nodes                                                                                                      \
) {                                                                                                                         \
                                                                                                                            \
    /* Node index. */                                                                                                       \
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;                                                            \
                                                                                                                            \
    if (nodeIndex < numberOfNodes) {                                                                                        \
                                                                                                                            \
        /* Node state. */                                                                                                   \
        const int nodeState = nodeStates[nodeIndex];                                                                        \
                                                                                                                            \
        /* Valid node. */                                                                                                   \
        if (nodeState != -1) {                                                                                              \
                                                                                                                            \
            /* New node index. */                                                                                           \
            int newNodeIndex = nodeOffsets[nodeIndex];                                                                      \
            newNodeIndex = nodeState > 0 ? newNodeIndex : newNodeIndex + newNumberOfInteriorNodes;                          \
                                                                                                                            \
            /* Parent. */                                                                                                   \
            int nodeParentIndex = nodeParentIndices[nodeIndex];                                                             \
                                                                                                                            \
            /* Size. */                                                                                                     \
            int nodeSize = nodeSizes[nodeIndex];                                                                            \
                                                                                                                            \
            /* Child indices. */                                                                                            \
            int nodeLeftIndex = nodeLeftIndices[nodeIndex];                                                                 \
            int nodeRightIndex = nodeRightIndices[nodeIndex];                                                               \
                                                                                                                            \
            /* States. */                                                                                                   \
            int nodeLeftState = nodeStates[nodeLeftIndex];                                                                  \
            int nodeRightState = nodeStates[nodeRightIndex];                                                                \
                                                                                                                            \
            /* Remap parent index. */                                                                                       \
            if (nodeParentIndex >= 0)                                                                                       \
                nodeParentIndex = nodeOffsets[nodeParentIndex];                                                             \
                                                                                                                            \
            /* Boxes. */                                                                                                    \
            Vec4f leftBoxMin, leftBoxMax;                                                                                   \
            Vec4f rightBoxMin, rightBoxMax;                                                                                 \
                                                                                                                            \
            /* Interior. */                                                                                                 \
            if (nodeState > 0) {                                                                                            \
                int nodeLeftOffset = nodeOffsets[nodeLeftIndex];                                                            \
                int nodeRightOffset = nodeOffsets[nodeRightIndex];                                                          \
                leftBoxMin = nodeBoxesMin[nodeLeftIndex];                                                                   \
                leftBoxMax = nodeBoxesMax[nodeLeftIndex];                                                                   \
                rightBoxMin = nodeBoxesMin[nodeRightIndex];                                                                 \
                rightBoxMax = nodeBoxesMax[nodeRightIndex];                                                                 \
                nodeLeftIndex = nodeLeftState > 0 ? nodeLeftOffset : ~(nodeLeftOffset + newNumberOfInteriorNodes);          \
                nodeRightIndex = nodeRightState > 0 ? nodeRightOffset : ~(nodeRightOffset + newNumberOfInteriorNodes);      \
            }                                                                                                               \
                                                                                                                            \
            /* Leaf. */                                                                                                     \
            else {                                                                                                          \
                int triangleOffset = triangleOffsets[nodeIndex];                                                            \
                leftBoxMin = rightBoxMin = nodeBoxesMin[nodeIndex];                                                         \
                leftBoxMax = rightBoxMax = nodeBoxesMax[nodeIndex];                                                         \
                nodeLeftIndex = triangleOffset;                                                                             \
                nodeRightIndex = nodeLeftIndex + nodeSize;                                                                  \
                nodeSize = ~nodeSize;                                                                                       \
            }                                                                                                               \
                                                                                                                            \
            /* Output node. */                                                                                              \
            HipBVHNode node;                                                                                                \
            node.setChildBoundingBox(0, AABB(Vec3f(leftBoxMin), Vec3f(leftBoxMax)));                                        \
            node.setChildBoundingBox(1, AABB(Vec3f(rightBoxMin), Vec3f(rightBoxMax)));                                      \
            node.setChildIndex(0, nodeLeftIndex);                                                                           \
            node.setChildIndex(1, nodeRightIndex);                                                                          \
            node.setSize(nodeSize);                                                                                         \
            node.setParentIndex(nodeParentIndex);                                                                           \
            nodes[newNodeIndex] = node;                                                                                     \
                                                                                                                            \
        }                                                                                                                   \
                                                                                                                            \
    }                                                                                                                       \
                                                                                                                            \
}

DEFINE_COMPACT(HipBVHNodeBin, Bin)
