/**
 * \file	HLBVHBuilderKernels.cu
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	HLBVHBuilder kernels soruce file.
 */

#include "rt/bvh/HipBVHUtil.h"
#include "rt/bvh/HLBVHBuilderKernels.h"
#include "rt/bvh/HLBVHBin.h"

DEVICE_INLINE Vec4i mortonCodeToCoords(unsigned int code) {
    Vec4i coords;
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        coords.x += ((code >> (3 * i + 0)) & 1) << i;
        coords.y += ((code >> (3 * i + 1)) & 1) << i;
        coords.z += ((code >> (3 * i + 2)) & 1) << i;
    }
    return coords;
}

DEVICE_INLINE void updateBin(Vec4f * binMin, Vec4f * binMax, const HLBVHBin & bin) {
    Vec4f min = *binMin;
    if (bin.mn.x < min.x) atomicMin((int*)&binMin->x, __float_as_int(bin.mn.x));
    if (bin.mn.y < min.y) atomicMin((int*)&binMin->y, __float_as_int(bin.mn.y));
    if (bin.mn.z < min.z) atomicMin((int*)&binMin->z, __float_as_int(bin.mn.z));
    Vec4f max = *binMax;
    if (bin.mx.x > max.x) atomicMax((int*)&binMax->x, __float_as_int(bin.mx.x));
    if (bin.mx.y > max.y) atomicMax((int*)&binMax->y, __float_as_int(bin.mx.y));
    if (bin.mx.z > max.z) atomicMax((int*)&binMax->z, __float_as_int(bin.mx.z));
    atomicAdd((int*)&binMin->w, 1);
}

extern "C" GLOBAL void computeNodeStates(
    const int numberOfReferences,
    const int mortonCodeBits,
    const int mortonCodeSAHBits,
    int * termCounters,
    int * nodeParentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    int * nodeStates,
    unsigned long long * mortonCodes
) {

    // Reference index.
    const int referenceIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Morton code offset.
    const int mortonCodeOffset = 8 * sizeof(unsigned long long) - mortonCodeBits + mortonCodeSAHBits;

    if (referenceIndex < numberOfReferences) {

        // Node index.
        int nodeIndex = referenceIndex + numberOfReferences - 1;

        // Leaf.
        nodeStates[nodeIndex] = 0;

        // Actual node index.
        nodeIndex = nodeParentIndices[nodeIndex];

        // Go up to the root.
        while (atomicAdd(&termCounters[nodeIndex], 1) > 0) {

            // Child indices.
            int leftIndex = nodeLeftIndices[nodeIndex];
            if (leftIndex >= numberOfReferences - 1) leftIndex -= numberOfReferences - 1;
            int rightIndex = nodeRightIndices[nodeIndex];
            if (rightIndex >= numberOfReferences - 1) rightIndex -= numberOfReferences - 1;

            // Node delta.
            int deltaNode = delta(leftIndex, rightIndex, numberOfReferences, mortonCodes);

            // Cluster or subset of cluster.
            if (deltaNode >= mortonCodeOffset) {
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

extern "C" GLOBAL void computeLeafClusterIndices(
    const int numberOfReferences,
    int * leafClusterIndices,
    int * nodeParentIndices,
    int * nodeStates
) {

    // Reference index.
    const int referenceIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (referenceIndex < numberOfReferences) {

        // Node.
        int nodeIndex = referenceIndex + numberOfReferences - 1;

        // Find cluster index.
        int clusterIndex = nodeIndex;
        int parentIndex = nodeParentIndices[clusterIndex];
        int parentState = nodeStates[parentIndex];
        while (parentIndex > 0) {
            if (parentState == 0)
                clusterIndex = parentIndex;
            nodeIndex = parentIndex;
            parentIndex = nodeParentIndices[nodeIndex];
            parentState = nodeStates[parentIndex];
        }

        // Write cluster index.
        leafClusterIndices[referenceIndex] = clusterIndex;

    }

}

extern "C" GLOBAL void invalidateIntermediateClusters(
    const int numberOfReferences,
    int * termCounters,
    int * leafClusterIndices,
    int * nodeParentIndices,
    int * nodeStates
) {

    // Reference index.
    const int referenceIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (referenceIndex < numberOfReferences) {

        // Cluster index.
        int clusterIndex = leafClusterIndices[referenceIndex];

        // Node index.
        int nodeIndex = referenceIndex + numberOfReferences - 1;

        // Leaf reached.
        if (nodeIndex == clusterIndex) return;

        // Invalidate node.
        nodeStates[nodeIndex] = -1;

        // Actual node index.
        nodeIndex = nodeParentIndices[nodeIndex];

        // Go up to the root.
        while (atomicAdd(&termCounters[nodeIndex], 1) > 0) {

            // Leaf reached.
            if (nodeIndex == clusterIndex) break;

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
    const int numberOfReferences,
    int * nodeOffsets,
    int * nodeStates
) {

    // Node index.
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Warp thread index.
    const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

    if (nodeIndex < numberOfReferences - 1) {

        // Node state.
        int nodeState = nodeStates[nodeIndex];

        // Prefix scan.
        unsigned int warpBallot = __ballot(nodeState <= 0);
        int warpCount = __popc(warpBallot);
        int warpIndex = __popc(warpBallot & ((1u << warpThreadIndex) - 1));

        // Add count of components to the global counter.
        int warpOffset = 0;
        if (warpThreadIndex == 0)
            warpOffset = atomicAdd(&prefixScanOffset, warpCount);

        // Exchange offset between threads.
        warpOffset = __shfl(warpOffset, 0);

        // Node offset.
        int nodeOffset = warpOffset + warpIndex;

        // Node offset.
        nodeOffsets[nodeIndex] = numberOfReferences - nodeOffset - 2;

    }

}

extern "C" GLOBAL void compact(
    const int numberOfNodes,
    int * nodeOffsets,
    int * nodeStates,
    int * inputParentIndices,
    int * inputLeftIndices,
    int * inputRightIndices,
    int * outputParentIndices,
    int * outputLeftIndices,
    int * outputRightIndices
) {

    // Node index.
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Number of references.
    const int numberOfReferences = (numberOfNodes >> 1) + 1;

    if (nodeIndex < numberOfNodes) {

        // Node state.
        int nodeState = nodeStates[nodeIndex];

        if (nodeState <= 0) {

            // Node offset.
            int nodeOffset = nodeIndex;

            // Child indices.
            int nodeLeftIndex = inputLeftIndices[nodeIndex];
            int nodeRightIndex = inputRightIndices[nodeIndex];

            // Interior node.
            if (nodeIndex < numberOfReferences - 1) {

                // Node offset.
                nodeOffset = nodeOffsets[nodeIndex];

                // Left node leaf => Update parent.
                if (nodeLeftIndex < numberOfReferences - 1) nodeLeftIndex = nodeOffsets[nodeLeftIndex];

                // Right node leaf => Update parent.
                if (nodeRightIndex < numberOfReferences - 1) nodeRightIndex = nodeOffsets[nodeRightIndex];

            }

            // Parent and size.
            int nodeParentIndex = nodeState == 0 ? -1 : nodeOffsets[inputParentIndices[nodeIndex]];

            // Output node.
            outputLeftIndices[nodeOffset] = nodeLeftIndex;
            outputRightIndices[nodeOffset] = nodeRightIndex;
            outputParentIndices[nodeOffset] = nodeParentIndex;

        }

    }

}

extern "C" GLOBAL void computeClusters(
    const int numberOfNodes,
    const int mortonCodeBits,
    const int mortonCodeSAHBits,
    int * nodeStates,
    int * nodeOffsets,
    int * nodeLeftIndices,
    int * clusterNodeIndices,
    Vec4i * clusterBinIndices,
    unsigned long long * mortonCodes
) {

    // Node index.
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Warp thread index.
    const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

    // Number of references.
    const int numberOfReferences = (numberOfNodes >> 1) + 1;

    // Morton code offset.
    const int mortonCodeOffset = mortonCodeBits - mortonCodeSAHBits;

    if (nodeIndex < numberOfNodes) {

        // Node state.
        int nodeState = nodeStates[nodeIndex];

        // Prefix scan.
        unsigned int warpBallot = __ballot(nodeState == 0);
        int warpCount = __popc(warpBallot);
        int warpIndex = __popc(warpBallot & ((1u << warpThreadIndex) - 1));

        // Add count of components to the global counter.
        int warpOffset = 0;
        if (warpThreadIndex == 0)
            warpOffset = atomicAdd(&prefixScanOffset, warpCount);

        // Exchange offset between threads.
        warpOffset = __shfl(warpOffset, 0);

        // Cluster index.
        int clusterIndex = warpOffset + warpIndex;

        // Node offset.
        int nodeOffset = nodeIndex < numberOfReferences - 1 ? nodeOffsets[nodeIndex] : nodeIndex;

        // Leaf index.
        int leafIndex = nodeLeftIndices[nodeIndex];
        if (leafIndex >= numberOfReferences - 1) leafIndex -= numberOfReferences - 1;

        // Morton code.
        unsigned long long mortonCode = mortonCodes[leafIndex];

        // Setup cluster.
        if (nodeState == 0) {
            clusterNodeIndices[clusterIndex] = nodeOffset;
            clusterBinIndices[clusterIndex] = mortonCodeToCoords(mortonCode >> mortonCodeOffset);
        }

    }

}

extern "C" GLOBAL void resetBins(
    const int numberOfAllBins,
    Vec4f * binBoxesMin,
    Vec4f * binBoxesMax
) {

    // Bin index.
    const int binIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Reset bins.
    if (binIndex < numberOfAllBins) {
        binBoxesMin[binIndex] = Vec4f(MAX_FLOAT, MAX_FLOAT, MAX_FLOAT, __int_as_float(0));
        binBoxesMax[binIndex] = Vec4f(-MAX_FLOAT, -MAX_FLOAT, -MAX_FLOAT, __int_as_float(0));
    }

}

extern "C" GLOBAL void binClusters(
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
) {

    // Cluster index.
    const int clusterIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (clusterIndex < numberOfClusters) {

        // Split task index.
        int clusterTaskIndex = clusterTaskIndices[clusterIndex];

        // Process only not assigned clusters.
        if (clusterTaskIndex >= 0) {

            // Node index.
            int nodeIndex = clusterNodeIndices[clusterIndex];

            // Bin index to the bin set.
            Vec4i binIndex = clusterBinIndices[clusterIndex];

            // Absolute bin indices.
            int binOffsetX = clusterTaskIndex + numberOfTasks * (binIndex.x + 0 * numberOfBins);
            int binOffsetY = clusterTaskIndex + numberOfTasks * (binIndex.y + 1 * numberOfBins);
            int binOffsetZ = clusterTaskIndex + numberOfTasks * (binIndex.z + 2 * numberOfBins);

            // Update bins.
            HLBVHBin clusterBox = HLBVHBin(nodeBoxesMin[nodeIndex], nodeBoxesMax[nodeIndex]);
            updateBin(binBoxesMin + binOffsetX, binBoxesMax + binOffsetX, clusterBox);
            updateBin(binBoxesMin + binOffsetY, binBoxesMax + binOffsetY, clusterBox);
            updateBin(binBoxesMin + binOffsetZ, binBoxesMax + binOffsetZ, clusterBox);

        }

    }

}

extern "C" GLOBAL void split(
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
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int taskEnd = divCeilLog(inputQueueSize, LOG_WARP_THREADS) << LOG_WARP_THREADS;

    // Warp thread index.
    const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

    if (taskIndex < taskEnd) {

        // The best split plane SAH cost.
        float bestCost = MAX_FLOAT;

        // The best split plane index and best axis index.
        int bestAxis = -1;
        int bestBinIndex = -1;

        // Interior flags.
        bool interior = false;
        bool leftInterior = false;
        bool rightInterior = false;

        // The best bounding boxes.
        HLBVHBin leftBin, rightBin;

        // Input task.
        HLBVHTask task;

        // Only valid tasks.
        if (taskIndex < inputQueueSize) {

            // Task.
            task = inputQueue[taskIndex];

            // Bins.
            HLBVHBin bins[MAX_BINS];

            // Bin.
            HLBVHBin bin;

            // Find best split plane.
            for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {

                // Right prefix scan.
                int binOffset = taskIndex + inputQueueSize * (numberOfBins - 1 + axisIndex * numberOfBins);
                bins[numberOfBins - 1] = HLBVHBin(binBoxesMin[binOffset], binBoxesMax[binOffset]);
                for (int binIndex = numberOfBins - 2; binIndex >= 0; --binIndex) {
                    binOffset = taskIndex + inputQueueSize * (binIndex + axisIndex * numberOfBins);
                    bins[binIndex] = bins[binIndex + 1];
                    bins[binIndex].include(binBoxesMin[binOffset], binBoxesMax[binOffset]);
                }

                // Left prefix scan.
                binOffset = taskIndex + inputQueueSize * axisIndex * numberOfBins;
                bin = HLBVHBin(binBoxesMin[binOffset], binBoxesMax[binOffset]);
                for (int binIndex = 0; binIndex < numberOfBins - 1; ++binIndex) {

                    // Only valid splits.
                    if (bin.clusterCounter > 0 && bins[binIndex + 1].clusterCounter > 0) {

                        // Update the best cost and split index.
                        float cost = bin.cost() + bins[binIndex + 1].cost();
                        if (cost < bestCost) {
                            bestCost = cost;
                            bestAxis = axisIndex;
                            bestBinIndex = binIndex;
                            leftBin = bin;
                            rightBin = bins[binIndex + 1];
                            interior = true;
                            leftInterior = bin.clusterCounter > 1;
                            rightInterior = bins[binIndex + 1].clusterCounter > 1;
                        }

                    }

                    // Add next bin from the right.
                    binOffset = taskIndex + inputQueueSize * (binIndex + 1 + axisIndex * numberOfBins);
                    bin.include(binBoxesMin[binOffset], binBoxesMax[binOffset]);

                }

            }

            // All clusters fell in a single bin.
            if (bestCost == MAX_FLOAT && bin.clusterCounter > 1) {
                bestAxis = 3;
                bestBinIndex = bin.clusterCounter / 2;
                leftBin = bin;
                rightBin = bin;
                interior = true;
            }

        }

        // Warp wide prefix scan of interior nodes.
        int outputNodeCount = leftInterior + rightInterior;
        int warpSum = warpScan(warpThreadIndex, outputNodeCount);

        // Add count of interior nodes to the global counter.
        int warpOffset = 0;
        if (warpThreadIndex == 31)
            warpOffset = atomicAdd(&prefixScanOffset, warpSum);
        warpSum -= outputNodeCount;

        // Exchange offset between threads.
        warpOffset = __shfl(warpOffset, 31);

        // Indices for children nodes.
        int nodeOffset = warpOffset + warpSum;

        // Warp wide prefix scan of output tasks.
        const unsigned int warpBallot = __ballot(interior);
        const int warpCount = __popc(warpBallot) << 1;
        const int warpIndex = __popc(warpBallot & ((1u << warpThreadIndex) - 1)) << 1;

        // Add count to the global counter.
        if (warpThreadIndex == 0)
            warpOffset = atomicAdd(outputQueueSizeLoc, warpCount);

        // Exchange offset between threads.
        warpOffset = __shfl(warpOffset, 0);

        // Index to the input task queue.
        int taskOffset = warpOffset + warpIndex;

        // Only valid tasks.
        if (taskIndex < inputQueueSize) {

            // Cluster leaf.
            if (!interior) {

                // No new cluster tasks.
                newTaskIndices[taskIndex] = ~taskOffset;

            }

            // Cluster interior node.
            else {

                // Node order.
                bool nodeOrder = leftInterior || !rightInterior;

                // New task index.
                newTaskIndices[taskIndex] = taskOffset;

                // Split index.
                if (bestAxis < 3) splitIndices[taskIndex] = bestBinIndex + bestAxis * numberOfBins;
                else splitIndices[taskIndex] = ~bestBinIndex;

                // Output tasks.
                outputQueue[taskOffset] = HLBVHTask(nodeOffset + !nodeOrder, task.nodeIndex);
                outputQueue[taskOffset + 1] = HLBVHTask(nodeOffset + nodeOrder, task.nodeIndex);

                // Box.
                AABB box;
                box.grow(leftBin.mn);
                box.grow(leftBin.mx);
                box.grow(rightBin.mn);
                box.grow(rightBin.mx);

                // Write node.
                nodeLeftIndices[task.nodeIndex] = nodeOffset + !nodeOrder;
                nodeRightIndices[task.nodeIndex] = nodeOffset + nodeOrder;
                nodeParentIndices[task.nodeIndex] = task.parentIndex;
                nodeBoxesMin[task.nodeIndex] = Vec4f(box.mn, 0.0f);
                nodeBoxesMax[task.nodeIndex] = Vec4f(box.mx, 0.0f);

            }

        }

    }

}

extern "C" GLOBAL void distributeClusters(
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
    Vec4f * binBoxesMin,
    Vec4f * binBoxesMax,
    HLBVHTask * inputQueue
) {

    // Cluster index.
    const int clusterIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (clusterIndex < numberOfClusters) {

        // Task index.
        int clusterTaskIndex = clusterTaskIndices[clusterIndex];

        // Process only not assigned clusters.
        if (clusterTaskIndex >= 0) {

            // New task index.
            int newTaskIndex = newTaskIndices[clusterTaskIndex];

            // Cluster leaf.
            if (newTaskIndex < 0) {

                // Task.
                HLBVHTask task = inputQueue[clusterTaskIndex];

                // Node index.
                int nodeIndex = clusterNodeIndices[clusterIndex];
                
                // Node is left child => correct parent's left index.
                if (nodeLeftIndices[task.parentIndex] == task.nodeIndex)
                    nodeLeftIndices[task.parentIndex] = nodeIndex;

                // Node is right child => correct parent's right index.
                else
                    nodeRightIndices[task.parentIndex] = nodeIndex;

                // Parent index.
                nodeParentIndices[nodeIndex] = task.parentIndex;

                // Mark as assigned.
                clusterTaskIndices[clusterIndex] = -1;

            }

            // Cluster interior node.
            else {

                // Split index and axis encoded.
                int splitIndex = splitIndices[clusterTaskIndex];

                // Best split index.
                int bestSplitIndex = -1;

                // Split bin index.
                int clusterBinIndexOverall = -1;

                // Object median split.
                if (splitIndex < 0) {

                    // Split bin index.
                    clusterBinIndexOverall = atomicAdd((int*)&binBoxesMax[clusterTaskIndex].w, 1);

                    // Best split index.
                    bestSplitIndex = (~splitIndex) - 1;

                }

                // Regular split.
                else {

                    // Bin index.
                    Vec4i clusterBinIndex = clusterBinIndices[clusterIndex];

                    // Split bin index.
                    clusterBinIndexOverall = splitIndex < numberOfBins ? clusterBinIndex.x :
                        (splitIndex < 2 * numberOfBins ? clusterBinIndex.y : clusterBinIndex.z);

                    // Best split index.
                    bestSplitIndex = splitIndex & (numberOfBins - 1);

                }

                // Update task index.
                clusterTaskIndices[clusterIndex] = clusterBinIndexOverall <= bestSplitIndex ? newTaskIndex : newTaskIndex + 1;

            }

        }

    }

}
