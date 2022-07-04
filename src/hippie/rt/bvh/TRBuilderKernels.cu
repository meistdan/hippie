/**
 * \file	TRBuilderKernels.cu
 * \author	Daniel Meister
 * \date	2016/02/17
 * \brief	TRBuilder kernels soruce file.
 */

#include "rt/bvh/TRBuilderKernels.h"
#include "rt/bvh/TRBuilderUtil.h"
#include "rt/HipUtil.h"

DEVICE void reduceOptimal(float & optimalCost, int & optimalMask, int numberOfValues) {
    for (int i = numberOfValues >> 1; i > 0; i = (i >> 1)) {
        float otherValue = __shfl_down(optimalCost, i);
        int otherMask = __shfl_down(optimalMask, i);
        if (otherValue < optimalCost) {
            optimalCost = otherValue;
            optimalMask = otherMask;
        }
    }
}

DEVICE void calculateSubsetSurfaceAreas(
    int treeletSize,
    int * treeletLeaves,
    float * subsetAreas,
    float4 * subsetBoxesMin,
    float4 * subsetBoxesMax,
    float4 * nodeBoxesMin,
    float4 * nodeBoxesMax,
    float * costs
) {
    float bbMin[3], bbMax[3];
    if (THREAD_WARP_INDEX < treeletSize) {
        floatArrayFromFloat4(nodeBoxesMin[treeletLeaves[THREAD_WARP_INDEX]], bbMin);
        floatArrayFromFloat4(nodeBoxesMax[treeletLeaves[THREAD_WARP_INDEX]], bbMax);
    }

    // The 5 most significative bits are common ammong the thread's subsets.
    int subset = THREAD_WARP_INDEX * 4;
    float3 baseMin, baseMax;
    baseMin.x = MAX_FLOAT;
    baseMin.y = MAX_FLOAT;
    baseMin.z = MAX_FLOAT;
    baseMax.x = -MAX_FLOAT;
    baseMax.y = -MAX_FLOAT;
    baseMax.z = -MAX_FLOAT;
    for (int i = (treeletSize - 5); i < treeletSize; ++i) {
        float3 leafBbMin, leafBbMax;
        SHFL_FLOAT3(leafBbMin, bbMin, i);
        SHFL_FLOAT3(leafBbMax, bbMax, i);
        if (subset & (1 << i))
            expandBoundingBox(baseMin, baseMax, leafBbMin, leafBbMax);
    }

    int iterations = max(1, 1 << (treeletSize - 5)); // Num elements / 32, rounded up.
    for (int j = 0; j < iterations; ++j) {
        float3 subsetMin, subsetMax;
        subsetMin.x = baseMin.x;
        subsetMin.y = baseMin.y;
        subsetMin.z = baseMin.z;
        subsetMax.x = baseMax.x;
        subsetMax.y = baseMax.y;
        subsetMax.z = baseMax.z;
        for (int i = 0; i < (treeletSize - 5); ++i) {
            float3 leafBbMin, leafBbMax;
            SHFL_FLOAT3(leafBbMin, bbMin, i);
            SHFL_FLOAT3(leafBbMax, bbMax, i);
            if (subset & (1 << i))
                expandBoundingBox(subsetMin, subsetMax, leafBbMin, leafBbMax);
        }

        // Store bounding boxes and their surface areas.
        int position = (1 << treeletSize) * GLOBAL_WARP_INDEX + subset;
        subsetBoxesMin[position] = float4FromFloat3(subsetMin);
        subsetBoxesMax[position] = float4FromFloat3(subsetMax);
        float subsetArea = calculateBoundingBoxSurfaceArea(subsetMin, subsetMax);
        subsetAreas[position] = subsetArea;
        costs[subset] = subsetArea;

        ++subset;
    }
}

DEVICE void processSchedule(
    int numberOfRounds,
    int * schedule,
    float * costs,
    char * partitionMasks,
    int treeletReferences,
    float ci,
    float ct
) {
    for (int j = 0; j < numberOfRounds; ++j) {
        int subset = schedule[THREAD_WARP_INDEX + j * WARP_THREADS];
        if (subset != 0) {
            // Process all possible partitions of the subset.
            float optimalCost = MAX_FLOAT;
            int optimalPartition = 0;
            int delta = (subset - 1) & subset;
            int partition = (-delta) & subset;
            float partitionCost;
            while (partition != 0) {
                partitionCost = costs[partition] + costs[partition ^ subset];
                if (partitionCost < optimalCost) {
                    optimalCost = partitionCost;
                    optimalPartition = partition;
                }
                partition = (partition - delta) & subset;
            }

            // Calculate subset SAH. Keep whichever has a lower SAH between collapsing 
            // the subset treelet or leaving it as is.
            costs[subset] = min(ct * costs[subset] + optimalCost, ci * costs[subset] * treeletReferences);
            partitionMasks[subset] = static_cast<char>(optimalPartition);
        }

        WARP_SYNC;
    }
}

DEVICE void processSubsets(
    int treeletSize,
    int treeletReferences,
    float * costs,
    char * partitionMasks,
    float ci,
    float ct
) {
    // Process subsets of size treeletSize-1. Each 4 threads will process a subset. There 
    // are treeletSize subsets.
    if (THREAD_WARP_INDEX < 4 * treeletSize) {
        // To find the nth subset of treeletSize-1 elements, start with a sequence of 
        // treeletSize ones and set the nth bit to 0.
        int subset = ((1 << treeletSize) - 1) & (~(1 << (THREAD_WARP_INDEX / 4)));

        // To assemble the partitions of nth subset of size treeletSize-1, we create a 
        // mask to split that subset before the (n-1)th least significant bit. We then 
        // get the left part of the masked base number and shift left by one (thus adding 
        // the partition's 0). The last step is to OR the result with the right part of 
        // the masked number and shift one more time to set the least significant bit to 
        // 0. Below is an example for a treelet size of 7:
        // subset = 1110111 (7 bits)
        // base = abcde (5 bits)
        // partition = abc0de0 (7 bits)
        // The cast to int is required so max does not return the wrong value.
        int leftMask = -(1 << max(static_cast<int>((THREAD_WARP_INDEX / 4) - 1), 0));
        // x & 3 == x % 4.
        int partitionBase = (THREAD_WARP_INDEX & 3) + 1;
        float optimalCost = MAX_FLOAT;
        int optimalPartition = 0;
        int numberOfPartitions = (1 << (treeletSize - 2)) - 1;
        int partition = (((partitionBase & leftMask) << 1) |
            (partitionBase & ~leftMask)) << 1;
        for (int j = (THREAD_WARP_INDEX & 3); j < numberOfPartitions; j += 4)
        {
            float partitionCost = costs[partition] + costs[partition ^ subset];
            if (partitionCost < optimalCost)
            {
                optimalCost = partitionCost;
                optimalPartition = partition;
            }

            partitionBase += 4;
            partition = (((partitionBase & leftMask) << 1) |
                (partitionBase & ~leftMask)) << 1;
        }

        reduceOptimal(optimalCost, optimalPartition, 4);

        if ((THREAD_WARP_INDEX & 3) == 0) {
            // Calculate subset SAH. Keep whichever has a lower SAH between collapsing 
            // the subset treelet or leaving it as is.
            costs[subset] = min(ct * costs[subset] + optimalCost, ci * costs[subset] * treeletReferences);
            partitionMasks[subset] = static_cast<char>(optimalPartition);
        }
    }

    WARP_SYNC;

    // Process subsets of size treeletSize
    float optimalCost = MAX_FLOAT;
    int optimalPartition = 0;
    int subset = (1 << treeletSize) - 1;
    int partition = (THREAD_WARP_INDEX + 1) * 2;
    int numberOfPartitions = (1 << (treeletSize - 1)) - 1;
    for (int j = THREAD_WARP_INDEX; j < numberOfPartitions; j += 32)
    {
        float partitionCost = costs[partition] + costs[partition ^ subset];
        if (partitionCost < optimalCost)
        {
            optimalCost = partitionCost;
            optimalPartition = partition;
        }

        partition += 64;
    }

    reduceOptimal(optimalCost, optimalPartition, WARP_THREADS);
    if (THREAD_WARP_INDEX == 0) {
        // Calculate subset SAH. Keep whichever has a lower SAH between collapsing 
        // the subset treelet or leaving it as is
        costs[subset] = min(ct * costs[subset] + optimalCost, ci * costs[subset] * treeletReferences);
        partitionMasks[subset] = static_cast<char>(optimalPartition);
    }
}

DEVICE void updateTreelet(
    int treeletSize,
    int * parentIndices,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    float4 * nodeBoxesMin,
    float4 * nodeBoxesMax,
    float * surfaceAreas,
    int * treeletInternalNodes,
    int * treeletLeaves,
    float * subsetAreas,
    float * costs,
    char * partitionMasks,
    float* nodesSah,
    float4 * subsetBoxesMin,
    float4 * subsetBoxesMax,
    int * stackNode,
    char * stackMask,
    int * stackSize,
    int * currentInternalNode
) {
    int globalWarpIndex = GLOBAL_WARP_INDEX;
    if (costs[(1 << treeletSize) - 1] < nodesSah[treeletInternalNodes[0]]) {
        if (THREAD_WARP_INDEX == 0) {
            stackNode[globalWarpIndex * (treeletSize - 1)] = treeletInternalNodes[0];
            stackMask[globalWarpIndex * (treeletSize - 1)] =
                static_cast<char>((1 << treeletSize) - 1);
            stackSize[globalWarpIndex] = 1;
            currentInternalNode[globalWarpIndex] = 1;
        }

        while (stackSize[globalWarpIndex] > 0) {
            int lastStackSize = stackSize[globalWarpIndex];
            if (THREAD_WARP_INDEX == 0) {
                stackSize[globalWarpIndex] = 0;
            }

            if (THREAD_WARP_INDEX < lastStackSize) {
                int nodeSubset = stackMask
                    [globalWarpIndex * (treeletSize - 1) + THREAD_WARP_INDEX];
                char partition = partitionMasks[nodeSubset];
                char partitionComplement = partition ^ nodeSubset;
                int subsetRoot =
                    stackNode[globalWarpIndex * (treeletSize - 1) + THREAD_WARP_INDEX];

                int childIndex;
                if (__popc(partition) > 1) {
                    // Update node pointers.
                    int currentNode = atomicAdd(currentInternalNode + globalWarpIndex, 1);
                    WARP_SYNC;
                    childIndex = treeletInternalNodes[currentNode];
                    nodeLeftIndices[subsetRoot] = childIndex;
                    parentIndices[childIndex] = subsetRoot;

                    int position = (1 << treeletSize) * globalWarpIndex +
                        partition;
                    float4 bbMin = subsetBoxesMin[position];
                    float4 bbMax = subsetBoxesMax[position];
                    float area = calculateBoundingBoxSurfaceArea(bbMin, bbMax);

                    // Update node area and bounding box.   
                    nodeBoxesMin[childIndex] = bbMin;
                    nodeBoxesMax[childIndex] = bbMax;
                    surfaceAreas[childIndex] = area;
                    nodesSah[childIndex] = costs[partition];

                    // Add child to stack.
                    int stackIndex = atomicAdd(stackSize + globalWarpIndex, 1);
                    WARP_SYNC;
                    stackNode[globalWarpIndex * (treeletSize - 1) + stackIndex] =
                        childIndex;
                    stackMask[globalWarpIndex * (treeletSize - 1) + stackIndex] =
                        partition;
                }
                else {
                    childIndex = treeletLeaves[__ffs(partition) - 1];
                    nodeLeftIndices[subsetRoot] = childIndex;
                    parentIndices[childIndex] = subsetRoot;
                }

                if (__popc(partitionComplement) > 1) {
                    // Update node pointers.
                    int currentNode = atomicAdd(currentInternalNode + globalWarpIndex, 1);
                    WARP_SYNC;
                    int childIndex = treeletInternalNodes[currentNode];
                    nodeRightIndices[subsetRoot] = childIndex;
                    parentIndices[childIndex] = subsetRoot;

                    int position = (1 << treeletSize) * globalWarpIndex +
                        partitionComplement;
                    float4 bbMin = subsetBoxesMin[position];
                    float4 bbMax = subsetBoxesMax[position];
                    float area = calculateBoundingBoxSurfaceArea(bbMin, bbMax);

                    // Update node area and bounding box.
                    nodeBoxesMin[childIndex] = bbMin;
                    nodeBoxesMax[childIndex] = bbMax;
                    surfaceAreas[childIndex] = area;
                    nodesSah[childIndex] = costs[partition];

                    // Add child to stack.
                    int stackIndex = atomicAdd(stackSize + globalWarpIndex, 1);
                    WARP_SYNC;
                    stackNode[globalWarpIndex * (treeletSize - 1) + stackIndex] =
                        childIndex;
                    stackMask[globalWarpIndex * (treeletSize - 1) + stackIndex] =
                        partitionComplement;
                }
                else {
                    int childIndex = treeletLeaves[__ffs(partitionComplement) - 1];
                    nodeRightIndices[subsetRoot] = childIndex;
                    parentIndices[childIndex] = subsetRoot;
                }
            }
        }
    }
}

extern "C" GLOBAL void optimize(
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
) {

    // Split the pre-allocated shared memory into distinct arrays for our treelet.
    extern __shared__ int sharedMemory[];
    __shared__ int * treeletInternalNodes;
    __shared__ int * treeletLeaves;
    __shared__ float * treeletLeavesAreas;
    __shared__ float * costs;
    __shared__ char * partitionMasks;

    // Having only the first thread perform this assignments and then
    // synchronizing is actually slower than issuing the assignments on all threads.
    int numberOfWarps = blockDim.x / WARP_THREADS;
    if (THREAD_WARP_INDEX == 0) {
        treeletInternalNodes = sharedMemory;
        treeletLeaves = treeletInternalNodes + (treeletSize - 1) * numberOfWarps;
        treeletLeavesAreas = (float*)(treeletLeaves + treeletSize * numberOfWarps);
        costs = treeletLeavesAreas + treeletSize * numberOfWarps;
        partitionMasks = (char*)(costs + (1 << treeletSize) * numberOfWarps);
    }
    __syncthreads();

    // If this flag is set, the thread will be excluded from the bottom up traversal, but will 
    // still be available to help form and optimiza treelets.
    const int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize leaves.
    int currentNodeIndex;
    if (threadIndex < numberOfReferences) {
        int leafIndex = threadIndex + numberOfReferences - 1;
        float area = surfaceAreas[leafIndex];
        currentNodeIndex = parentIndices[leafIndex];
        subtreeReferences[leafIndex] = 1;
        nodesSah[leafIndex] = ci * area;
    }
    else {
        currentNodeIndex = -1;
    }

    while (__ballot(currentNodeIndex >= 0) != 0) {

        // Number of threads who already have processed the current node.
        unsigned int counter = 0;

        if (currentNodeIndex >= 0) {
            counter = atomicAdd(&termCounters[currentNodeIndex], 1);
            WARP_SYNC;
            // Only the last thread to arrive is allowed to process the current node. This ensures
            // that both its children will already have been processed.
            if (counter == 0) currentNodeIndex = -1;
        }

        // How many references can be reached by the subtree with root at the current node.
        int referenceCount = 0;
        if (counter != 0) {
            // Throughout the code, blocks that have loads separated from stores are so organized 
            // in order to increase ILP (Instruction level parallelism).
            int left = nodeLeftIndices[currentNodeIndex];
            int right = nodeRightIndices[currentNodeIndex];
            float area = surfaceAreas[currentNodeIndex];
            int referencesLeft = subtreeReferences[left];
            float sahLeft = nodesSah[left];
            int referencesRight = subtreeReferences[right];
            float sahRight = nodesSah[right];

            referenceCount = referencesLeft + referencesRight;
            subtreeReferences[currentNodeIndex] = referenceCount;
            nodesSah[currentNodeIndex] = ct * area + sahLeft + sahRight;
        }

        // Check which threads in the warp have treelets to be processed. We are only going to 
        // process a treelet if the current node is the root of a subtree with at least gamma references.
        unsigned int vote = __ballot(referenceCount >= gamma);

        while (vote != 0) {

            // Get the thread index for the treelet that will be processed.         
            int rootThreadIndex = __ffs(vote) - 1;

            // Get the treelet root by reading the corresponding thread's currentNodeIndex private variable.
            int treeletRootIndex = __shfl(currentNodeIndex, rootThreadIndex);

            formTreelet(treeletRootIndex, numberOfReferences, treeletSize, nodeLeftIndices, nodeRightIndices, surfaceAreas,
                WARP_ARRAY(treeletInternalNodes, treeletSize - 1), WARP_ARRAY(treeletLeaves, treeletSize), WARP_ARRAY(treeletLeavesAreas, treeletSize));

            // Optimize treelet.
            calculateSubsetSurfaceAreas(treeletSize, WARP_ARRAY(treeletLeaves, treeletSize), subsetAreas,
                subsetBoxesMin, subsetBoxesMax, nodeBoxesMin, nodeBoxesMax, WARP_ARRAY(costs, (1 << treeletSize)));

            // Set leaves cost.
            if (THREAD_WARP_INDEX < treeletSize) {
                int leafIndex = WARP_ARRAY_INDEX(1 << THREAD_WARP_INDEX, 1 << treeletSize);
                int treeletLeafIndex = WARP_ARRAY_INDEX(THREAD_WARP_INDEX, treeletSize);
                costs[leafIndex] = nodesSah[treeletLeaves[treeletLeafIndex]];
            }

            WARP_SYNC;

            int treeletReferences = subtreeReferences[treeletInternalNodes[WARP_ARRAY_INDEX(0, treeletSize - 1)]];

            // Process subsets of sizes 2 to treeletSize-2 using the schedule.
            processSchedule(numberOfRounds, schedule, WARP_ARRAY(costs, (1 << treeletSize)),
                WARP_ARRAY(partitionMasks, (1 << treeletSize)), treeletReferences, ci, ct);

            WARP_SYNC;

            // Procecss remaining subsets.
            processSubsets(treeletSize, treeletReferences, WARP_ARRAY(costs, (1 << treeletSize)),
                WARP_ARRAY(partitionMasks, (1 << treeletSize)), ci, ct);

            WARP_SYNC;

            updateTreelet(treeletSize, parentIndices, nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax,
                surfaceAreas, WARP_ARRAY(treeletInternalNodes, treeletSize - 1), WARP_ARRAY(treeletLeaves, treeletSize),
                subsetAreas, WARP_ARRAY(costs, (1 << treeletSize)), WARP_ARRAY(partitionMasks, (1 << treeletSize)),
                nodesSah, subsetBoxesMin, subsetBoxesMax, stackNode, stackMask, stackSize, currentInternalNode);

            // Update vote so each treelet is only processed once (set the bit that represents the 
            // treelet that will be processed back to 0).
            vote &= ~(1 << rootThreadIndex);
        }

        // Update current node pointer.
        if (currentNodeIndex >= 0) currentNodeIndex = parentIndices[currentNodeIndex];

    }

}

extern "C" GLOBAL void computeSurfaceAreas(
    const int numberOfNodes,
    float * surfaceAreas,
    float4 * nodeBoxesMin,
    float4 * nodeBoxesMax
) {

    // Node index.
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (nodeIndex < numberOfNodes) {

        // Box.
        float4 nodeBoxMin = nodeBoxesMin[nodeIndex];
        float4 nodeBoxMax = nodeBoxesMax[nodeIndex];

        // Surface Area.
        surfaceAreas[nodeIndex] = calculateBoundingBoxSurfaceArea(nodeBoxMin, nodeBoxMax);

    }

}
