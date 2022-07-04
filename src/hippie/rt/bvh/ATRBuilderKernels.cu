/**
 * \file	ATRBuilderKernels.cu
 * \author	Daniel Meister
 * \date	2016/02/11
 * \brief	ATRBuilder kernels soruce file.
 */

#include "rt/bvh/ATRBuilderKernels.h"
#include "rt/bvh/TRBuilderUtil.h"

#define LOWER_TRM_ROW(index) rowIndex((index))
DEVICE_INLINE int rowIndex(int index) {
    return static_cast<int>((-1 + sqrtf(8.0f * index + 1)) / 2) + 1;
}

#define LOWER_TRM_COL(index) columnIndex((index))
DEVICE_INLINE int columnIndex(int index) {
    int y = rowIndex(index);
    return (index)-(y * (y - 1)) / 2;
}

DEVICE void calculateDistancesMatrix(
    int * schedule,
    const int scheduleSize,
    float * distancesMatrix,
    int distanceMatrixSize,
    float * bbMin,
    float * bbMax
) {
    int numberOfIterations = (scheduleSize + (WARP_THREADS - 1)) / WARP_THREADS;
    for (int j = 0; j < numberOfIterations; ++j) {
        int element = 0;
        int elementIndex = THREAD_WARP_INDEX + j * WARP_THREADS;
        if (elementIndex < scheduleSize)
            element = schedule[elementIndex];
        int a = element >> 24;
        int b = ((element >> 16) & 0xFF);

        // Read bounding boxes.                    
        float3 bbMinA, bbMaxA, bbMinB, bbMaxB;
        SHFL_FLOAT3(bbMinA, bbMin, a);
        SHFL_FLOAT3(bbMaxA, bbMax, a);
        SHFL_FLOAT3(bbMinB, bbMin, b);
        SHFL_FLOAT3(bbMaxB, bbMax, b);

        if (a != 0 || b != 0) {
            float distance = calculateBoundingBoxAndSurfaceArea(bbMinA, bbMaxA, bbMinB, bbMaxB);
            distancesMatrix[LOWER_TRM_INDEX(a, b)] = distance;
        }

        a = ((element >> 8) & 0xFF);
        b = (element & 0xFF);

        // Read bounding boxes.
        SHFL_FLOAT3(bbMinA, bbMin, a);
        SHFL_FLOAT3(bbMaxA, bbMax, a);
        SHFL_FLOAT3(bbMinB, bbMin, b);
        SHFL_FLOAT3(bbMaxB, bbMax, b);

        if (a != 0 || b != 0) {
            float distance = calculateBoundingBoxAndSurfaceArea(bbMinA, bbMaxA, bbMinB, bbMaxB);
            distancesMatrix[LOWER_TRM_INDEX(a, b)] = distance;
        }

    }
}

DEVICE void findMinimum(int numberOfElements, int & index, float & distance) {
    int shflAmount = numberOfElements / 2;
    while (numberOfElements > 1) {
        int otherIndex = __shfl_down(index, shflAmount);
        float otherArea = __shfl_down(distance, shflAmount);
        if (otherArea < distance) {
            distance = otherArea;
            index = otherIndex;
        }
        numberOfElements = (numberOfElements + 1) / 2;
        shflAmount = numberOfElements / 2;
    }
}

DEVICE void findMinimumDistance(float * distancesMatrix, int lastRow, int & minIndex) {
    float minDistance = MAX_FLOAT;
    int matrixSize = sumArithmeticSequence(lastRow, 1, lastRow);
    for (int j = THREAD_WARP_INDEX; j < matrixSize; j += WARP_THREADS) {
        float distance = distancesMatrix[j];
        if (distance < minDistance) {
            minDistance = distance;
            minIndex = j;
        }
    }
    findMinimum(WARP_THREADS, minIndex, minDistance);
    minIndex = __shfl(minIndex, 0);
}

DEVICE void updateState(
    int joinRow,
    int joinCol,
    int lastRow,
    int & threadNode,
    float & threadSah,
    int * treeletInternalNodes,
    int * treeletLeaves,
    float * treeletLeavesAreas,
    float * bbMin,
    float * bbMax,
    float ct
) {

    // Update 'joinCol' bounding box and update treelet. The left and right indices 
    // and the bounding boxes must be read outside the conditional or else __shfl is 
    // not going to work.
    float3 bbMinA, bbMaxA, bbMinB, bbMaxB;
    float sah = __shfl(threadSah, joinRow) + __shfl(threadSah, joinCol);
    int leftIndex = __shfl(threadNode, joinRow);
    int rightIndex = __shfl(threadNode, joinCol);
    SHFL_FLOAT3(bbMinA, bbMin, joinRow);
    SHFL_FLOAT3(bbMaxA, bbMax, joinRow);
    SHFL_FLOAT3(bbMinB, bbMin, joinCol);
    SHFL_FLOAT3(bbMaxB, bbMax, joinCol);
    expandBoundingBox(bbMinA, bbMaxA, bbMinB, bbMaxB);
    if (THREAD_WARP_INDEX == joinCol) {
        threadNode = treeletInternalNodes[lastRow - 1];
        floatArrayFromFloat3(bbMinA, bbMin);
        floatArrayFromFloat3(bbMaxA, bbMax);
        float area = calculateBoundingBoxSurfaceArea(bbMinA, bbMaxA);
        treeletLeavesAreas[lastRow] = sah + ct * area;
        threadSah = treeletLeavesAreas[lastRow];
    }

    // Update 'joinRow' node and bounding box. The last block only modified 'joinCol', 
    // which won't conflict with this block, so we can synchronize only once after 
    // both blocks.
    int lastIndex = __shfl(threadNode, lastRow);
    float sahLast = __shfl(threadSah, lastRow);
    SHFL_FLOAT3(bbMinB, bbMin, lastRow);
    SHFL_FLOAT3(bbMaxB, bbMax, lastRow);
    if (THREAD_WARP_INDEX == joinRow) {
        threadNode = lastIndex;
        threadSah = sahLast;
        floatArrayFromFloat3(bbMinB, bbMin);
        floatArrayFromFloat3(bbMaxB, bbMax);
    }

    // Update lastRow with the information required to update the treelet.
    if (THREAD_WARP_INDEX == lastRow) {
        threadNode = leftIndex;
        treeletLeaves[lastRow] = rightIndex;
        floatArrayFromFloat3(bbMinA, bbMin);
        floatArrayFromFloat3(bbMaxA, bbMax);
    }

}

DEVICE void updateDistancesMatrix(
    int joinRow,
    int joinCol,
    int lastRow,
    float * distancesMatrix,
    float * bbMin,
    float * bbMax
) {

    // Copy last row to 'joinRow' row and columns.
    int destinationRow = THREAD_WARP_INDEX;
    int destinationCol = destinationRow;
    if (THREAD_WARP_INDEX < lastRow && THREAD_WARP_INDEX != joinRow)
    {
        destinationRow = max(joinRow, destinationRow);
        destinationCol = min(joinRow, destinationCol);
        int indexSource = LOWER_TRM_INDEX(lastRow, THREAD_WARP_INDEX);
        float distance = distancesMatrix[indexSource];
        int indexDestination = LOWER_TRM_INDEX(destinationRow, destinationCol);
        distancesMatrix[indexDestination] = distance;
    }

    // Update row and column 'joinCol'.
    destinationRow = THREAD_WARP_INDEX;
    destinationCol = destinationRow;
    if (THREAD_WARP_INDEX < lastRow && THREAD_WARP_INDEX != joinCol) {
        destinationRow = max(joinCol, destinationRow);
        destinationCol = min(joinCol, destinationCol);
    }
    float3 bbMinA, bbMaxA, bbMinB, bbMaxB;
    SHFL_FLOAT3(bbMinA, bbMin, destinationRow);
    SHFL_FLOAT3(bbMaxA, bbMax, destinationRow);
    SHFL_FLOAT3(bbMinB, bbMin, destinationCol);
    SHFL_FLOAT3(bbMaxB, bbMax, destinationCol);
    float distance = calculateBoundingBoxAndSurfaceArea(bbMinA, bbMaxA, bbMinB, bbMaxB);
    if (THREAD_WARP_INDEX < lastRow && THREAD_WARP_INDEX != joinCol) {
        int indexDestination = LOWER_TRM_INDEX(destinationRow, destinationCol);
        distancesMatrix[indexDestination] = distance;
    }

}

DEVICE void updateTreelet(
    int treeletSize,
    int threadNode,
    int * treeletInternalNodes,
    int * treeletLeaves,
    float * treeletLeavesAreas,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    float4 * nodeBoxesMin,
    float4 * nodeBoxesMax,
    int * parentIndices,
    float * surfaceAreas,
    float * costs,
    float * bbMin,
    float * bbMax
) {
    if (treeletLeavesAreas[1] < costs[treeletInternalNodes[0]]) {
        if (THREAD_WARP_INDEX >= 1 && THREAD_WARP_INDEX < treeletSize) {
            int nodeIndex = treeletInternalNodes[THREAD_WARP_INDEX - 1];
            nodeLeftIndices[nodeIndex] = threadNode;
            nodeRightIndices[nodeIndex] = treeletLeaves[THREAD_WARP_INDEX];
            parentIndices[threadNode] = nodeIndex;
            parentIndices[treeletLeaves[THREAD_WARP_INDEX]] = nodeIndex;
            costs[nodeIndex] = treeletLeavesAreas[THREAD_WARP_INDEX];
            float4 bbMin4, bbMax4;
            float4FromFloatArray(bbMin, bbMin4);
            float4FromFloatArray(bbMax, bbMax4);
            nodeBoxesMin[nodeIndex] = bbMin4;
            nodeBoxesMax[nodeIndex] = bbMax4;
            surfaceAreas[nodeIndex] = calculateBoundingBoxSurfaceArea(bbMin4, bbMax4);
        }
    }
}

extern "C" GLOBAL void optimize(
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
) {

    const int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // Split the pre-allocated shared memory into distinc arrays for our treelet.
    extern __shared__ int sharedMemory[];
    __shared__ int * treeletInternalNodes;
    __shared__ int * treeletLeaves;
    __shared__ float * treeletLeavesAreas;

    // Initialize shared variables
    if (threadIdx.x == 0) {
        int numberOfWarps = blockDim.x / WARP_THREADS;
        treeletInternalNodes = sharedMemory;
        treeletLeaves = treeletInternalNodes + (treeletSize - 1) * numberOfWarps;
        treeletLeavesAreas = (float*)(treeletLeaves + treeletSize * numberOfWarps);
    }
    __syncthreads();

    // If this flag is set, the thread will be excluded from the bottom up traversal, but will 
    // still be available to help form and optimize treelets.
    int currentNodeIndex;
    if (threadIndex < numberOfReferences) {
        int leafIndex = threadIndex + numberOfReferences - 1;
        subtreeReferences[leafIndex] = 1;
        currentNodeIndex = parentIndices[leafIndex];
        float area = surfaceAreas[leafIndex];
        costs[leafIndex] = ci * area;
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
        int referencesCount = 0;
        if (counter != 0) {
            // Throughout the code, blocks that have loads separated from stores are so organized 
            // in order to increase ILP (Instruction level parallelism).
            int left = nodeLeftIndices[currentNodeIndex];
            int right = nodeRightIndices[currentNodeIndex];
            float area = surfaceAreas[currentNodeIndex];
            int referencesLeft = subtreeReferences[left];
            float sahLeft = costs[left];
            int referencesRight = subtreeReferences[right];
            float sahRight = costs[right];

            referencesCount = referencesLeft + referencesRight;
            subtreeReferences[currentNodeIndex] = referencesCount;
            costs[currentNodeIndex] = ct * area + sahLeft + sahRight;
        }

        // Check which threads in the warp have treelets to be processed. We are only going to 
        // process a treelet if the current node is the root of a subtree with at least gamma references.
        unsigned int vote = __ballot(referencesCount >= gamma);

        while (vote != 0) {

            // Get the thread index for the treelet that will be.
            int rootThreadIndex = __ffs(vote) - 1;

            // Get the treelet root by reading the corresponding thread's currentNodeIndex private variable.
            int treeletRootIndex = __shfl(currentNodeIndex, rootThreadIndex);

            formTreelet(treeletRootIndex, numberOfReferences, treeletSize, nodeLeftIndices, nodeRightIndices, surfaceAreas,
                WARP_ARRAY(treeletInternalNodes, treeletSize - 1), WARP_ARRAY(treeletLeaves, treeletSize), WARP_ARRAY(treeletLeavesAreas, treeletSize));

            // Load bounding boxes.
            float bbMin[3], bbMax[3];
            if (THREAD_WARP_INDEX < treeletSize) {
                int index = WARP_ARRAY_INDEX(THREAD_WARP_INDEX, treeletSize);
                floatArrayFromFloat4(nodeBoxesMin[treeletLeaves[index]], bbMin);
                floatArrayFromFloat4(nodeBoxesMax[treeletLeaves[index]], bbMax);
            }

            calculateDistancesMatrix(schedule, scheduleSize, distanceMatrices +
                distanceMatrixSize * GLOBAL_WARP_INDEX, distanceMatrixSize, bbMin, bbMax);

            int threadNode;
            float threadSah;
            if (THREAD_WARP_INDEX < treeletSize) {
                int index = WARP_ARRAY_INDEX(THREAD_WARP_INDEX, treeletSize);
                threadNode = treeletLeaves[index];
                threadSah = costs[threadNode];
            }

            for (int lastRow = treeletSize - 1; lastRow > 0; --lastRow) {

                // Find pair with minimum distance.
                int minIndex = 0;
                findMinimumDistance(distanceMatrices + distanceMatrixSize * GLOBAL_WARP_INDEX, lastRow, minIndex);

                // Update treelet.
                int joinCol = LOWER_TRM_COL(minIndex);
                int joinRow = LOWER_TRM_ROW(minIndex);

                // Copy last row to 'joinRow' row and columns.
                if (THREAD_WARP_INDEX < lastRow && THREAD_WARP_INDEX != joinRow && lastRow > 1) {
                    int destinationRow = max(joinRow, THREAD_WARP_INDEX);
                    int destinationCol = min(joinRow, THREAD_WARP_INDEX);
                    int indexSource = distanceMatrixSize * GLOBAL_WARP_INDEX +
                        LOWER_TRM_INDEX(lastRow, THREAD_WARP_INDEX);
                    float distance = distanceMatrices[indexSource];
                    int indexDestination = distanceMatrixSize * GLOBAL_WARP_INDEX +
                        LOWER_TRM_INDEX(destinationRow, destinationCol);
                    distanceMatrices[indexDestination] = distance;
                }

                updateState(joinRow, joinCol, lastRow, threadNode, threadSah, WARP_ARRAY(treeletInternalNodes, treeletSize - 1),
                    WARP_ARRAY(treeletLeaves, treeletSize), WARP_ARRAY(treeletLeavesAreas, treeletSize), bbMin, bbMax, ct);

                // Update row and column 'joinCol'.
                if (lastRow > 1) {
                    updateDistancesMatrix(joinRow, joinCol, lastRow,
                        distanceMatrices + distanceMatrixSize * GLOBAL_WARP_INDEX, bbMin, bbMax);
                }
            }

            WARP_SYNC;

            updateTreelet(treeletSize, threadNode, WARP_ARRAY(treeletInternalNodes, treeletSize - 1),
                WARP_ARRAY(treeletLeaves, treeletSize), WARP_ARRAY(treeletLeavesAreas, treeletSize), nodeLeftIndices,
                nodeRightIndices, nodeBoxesMin, nodeBoxesMax, parentIndices, surfaceAreas, costs, bbMin, bbMax);

            // Update vote so each treelet is only processed once (set the bit that represents the 
            // treelet that will be processed back to 0).
            vote &= ~(1 << rootThreadIndex);

            WARP_SYNC;
        }

        // Update current node pointer.
        if (currentNodeIndex >= 0) currentNodeIndex = parentIndices[currentNodeIndex];

    }

}

extern "C" GLOBAL void optimizeSmall(
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
) {

    const int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // Split the pre-allocated shared memory into distinc arrays for our treelet.
    extern __shared__ int sharedMemory[];
    __shared__ int * treeletInternalNodes;
    __shared__ int * treeletLeaves;
    __shared__ float * treeletLeavesAreas;
    __shared__ float * distanceMatrices;

    // Initialize shared variables.
    if (threadIdx.x == 0) {
        int numberOfWarps = blockDim.x / WARP_THREADS;
        treeletInternalNodes = sharedMemory;
        treeletLeaves = treeletInternalNodes + (treeletSize - 1) * numberOfWarps;
        treeletLeavesAreas = (float*)(treeletLeaves + treeletSize * numberOfWarps);
        distanceMatrices = treeletLeavesAreas + treeletSize * numberOfWarps;
    }
    __syncthreads();

    // If this flag is set, the thread will be excluded from the bottom up traversal, but will 
    // still be available to help form and optimize treelets.
    int currentNodeIndex;
    if (threadIndex < numberOfReferences) {
        int leafIndex = threadIndex + numberOfReferences - 1;
        subtreeReferences[leafIndex] = 1;
        currentNodeIndex = parentIndices[leafIndex];
        float area = surfaceAreas[leafIndex];
        costs[leafIndex] = ci * area;
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
            float sahLeft = costs[left];
            int referencesRight = subtreeReferences[right];
            float sahRight = costs[right];

            referenceCount = referencesLeft + referencesRight;
            subtreeReferences[currentNodeIndex] = referenceCount;
            costs[currentNodeIndex] = ct * area + sahLeft + sahRight;
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

            // Load bounding boxes.
            float bbMin[3], bbMax[3];
            if (THREAD_WARP_INDEX < treeletSize) {
                int index = WARP_ARRAY_INDEX(THREAD_WARP_INDEX, treeletSize);
                floatArrayFromFloat4(nodeBoxesMin[treeletLeaves[index]], bbMin);
                floatArrayFromFloat4(nodeBoxesMax[treeletLeaves[index]], bbMax);
            }

            calculateDistancesMatrix(schedule, scheduleSize, WARP_ARRAY(distanceMatrices,
                distanceMatrixSize), distanceMatrixSize, bbMin, bbMax);

            int threadNode;
            float threadSah;
            if (THREAD_WARP_INDEX < treeletSize) {
                int index = WARP_ARRAY_INDEX(THREAD_WARP_INDEX, treeletSize);
                threadNode = treeletLeaves[index];
                threadSah = costs[threadNode];
            }

            for (int lastRow = treeletSize - 1; lastRow > 0; --lastRow) {
                // Find pair with minimum distance.              
                int minIndex = 0;
                findMinimumDistance(WARP_ARRAY(distanceMatrices, distanceMatrixSize), lastRow, minIndex);

                // Add modifications to a list.
                int joinCol = LOWER_TRM_COL(minIndex);
                int joinRow = LOWER_TRM_ROW(minIndex);
                updateState(joinRow, joinCol, lastRow, threadNode, threadSah,
                    WARP_ARRAY(treeletInternalNodes, treeletSize - 1),
                    WARP_ARRAY(treeletLeaves, treeletSize),
                    WARP_ARRAY(treeletLeavesAreas, treeletSize), bbMin, bbMax, ct);

                // Update distances matrix.
                if (lastRow > 1)
                    updateDistancesMatrix(joinRow, joinCol, lastRow, WARP_ARRAY(distanceMatrices, distanceMatrixSize), bbMin, bbMax);
            }

            WARP_SYNC;

            updateTreelet(treeletSize, threadNode, WARP_ARRAY(treeletInternalNodes, treeletSize - 1),
                WARP_ARRAY(treeletLeaves, treeletSize), WARP_ARRAY(treeletLeavesAreas, treeletSize), nodeLeftIndices,
                nodeRightIndices, nodeBoxesMin, nodeBoxesMax, parentIndices, surfaceAreas, costs, bbMin, bbMax);

            // Update vote so each treelet is only processed once (set the bit that represents the 
            // treelet that will be processed back to 0).
            vote &= ~(1 << rootThreadIndex);

            WARP_SYNC;
        }

        // Update current node pointer.
        if (currentNodeIndex >= 0)
            currentNodeIndex = parentIndices[currentNodeIndex];

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
