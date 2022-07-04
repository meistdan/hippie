/**
 * \file	TRBuilderUtil.h
 * \author	Daniel Meister
 * \date	2016/02/17
 * \brief	A header file containing functions for both TRBuilder and ATRBuilder.
 */

#include "Globals.h"

 // Get the global warp index.
#define GLOBAL_WARP_INDEX static_cast<int>((threadIdx.x + blockIdx.x * blockDim.x) / WARP_THREADS)

// Get the block warp index.
#define WARP_INDEX static_cast<int>(threadIdx.x / WARP_THREADS)

// Get a pointer to the beginning of a warp area in an array that stores a certain number of elements for each warp.
#define WARP_ARRAY(source, elementsPerWarp) ((source) + WARP_INDEX * (elementsPerWarp))

// Calculate the index of a value in an array that stores a certain number of elements for each warp.
#define WARP_ARRAY_INDEX(index, elementsPerWarp) (WARP_INDEX * (elementsPerWarp) + (index))

// Index of the thread in the warp, from 0 to WARP_SIZE - 1.
#define THREAD_WARP_INDEX (threadIdx.x & (WARP_THREADS - 1))

// Read a vector of 3 elements using shuffle operations.
#define SHFL_FLOAT3(destination, source, index) \
do { \
    (destination).x = __shfl((source)[0], (index)); \
    (destination).y = __shfl((source)[1], (index)); \
    (destination).z = __shfl((source)[2], (index)); \
} while (0);

// Get the number of elements which can be stored in a diagonal square matrix of size 'dim,
// excluding the main diagonal.
#define TRM_SIZE(dim) (((dim - 1) * (dim)) / 2)

// Get the index of an element in an array that represents a lower triangular matrix.
// The main diagonal elements are not included in the array.
#define LOWER_TRM_INDEX(row, column) (TRM_SIZE((row)) + (column))

// Uncomment this to enforce warp synchronization.
#define SAFE_WARP_SYNCHRONY 1

// Synchronize warp. This protects the code from future compiler optimization that 
// involves instructions reordering, possibly leading to race conditions. 
// __syncthreads() could be used instead, at a slight performance penalty.
#if SAFE_WARP_SYNCHRONY
#define WARP_SYNC \
__threadfence(); \
do { \
    int _sync = 0; \
    __shfl(_sync, 0); \
} while (0);
#else
#define WARP_SYNC \
do { \
} while (0);
#endif

DEVICE_INLINE void floatArrayFromFloat4(float4 source, float * destination) {
    destination[0] = source.x;
    destination[1] = source.y;
    destination[2] = source.z;
}

DEVICE_INLINE void floatArrayFromFloat3(float3 source, float * destination)
{
    destination[0] = source.x;
    destination[1] = source.y;
    destination[2] = source.z;
}

DEVICE_INLINE void float4FromFloatArray(const float* source, float4 & destination) {
    destination.x = source[0];
    destination.y = source[1];
    destination.z = source[2];
}

DEVICE_INLINE float4 float4FromFloat3(float3 source) {
    float4 temp;
    temp.x = source.x;
    temp.y = source.y;
    temp.z = source.z;
    return temp;
}

DEVICE_INLINE float calculateBoundingBoxAndSurfaceArea(
    const float3 bbMin1,
    const float3 bbMax1,
    const float3 bbMin2,
    const float3 bbMax2
) {
    float3 size;
    size.x = max(bbMax1.x, bbMax2.x) - min(bbMin1.x, bbMin2.x);
    size.y = max(bbMax1.y, bbMax2.y) - min(bbMin1.y, bbMin2.y);
    size.z = max(bbMax1.z, bbMax2.z) - min(bbMin1.z, bbMin2.z);
    return 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
}

DEVICE_INLINE float calculateBoundingBoxSurfaceArea(float3 bbMin, float3 bbMax) {
    float3 size;
    size.x = bbMax.x - bbMin.x;
    size.y = bbMax.y - bbMin.y;
    size.z = bbMax.z - bbMin.z;
    return 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
}

DEVICE_INLINE float calculateBoundingBoxSurfaceArea(float4 bbMin, float4 bbMax) {
    float3 size;
    size.x = bbMax.x - bbMin.x;
    size.y = bbMax.y - bbMin.y;
    size.z = bbMax.z - bbMin.z;
    return 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
}

DEVICE_INLINE void expandBoundingBox(
    float3 & groupBbMin,
    float3 & groupBbMax,
    const float3 & newBbMin, const float3& newBbMax
) {
    groupBbMin.x = min(newBbMin.x, groupBbMin.x);
    groupBbMin.y = min(newBbMin.y, groupBbMin.y);
    groupBbMin.z = min(newBbMin.z, groupBbMin.z);
    groupBbMax.x = max(newBbMax.x, groupBbMax.x);
    groupBbMax.y = max(newBbMax.y, groupBbMax.y);
    groupBbMax.z = max(newBbMax.z, groupBbMax.z);
}

DEVICE_INLINE bool isInternalNode(unsigned int index, unsigned int numberOfReferences) {
    return (index < numberOfReferences - 1);
}

DEVICE_INLINE bool isLeaf(unsigned int index, unsigned int numberOfReferences) {
    return !isInternalNode(index, numberOfReferences);
}

DEVICE_INLINE bool isTreeletRoot(int size) {
    return size == 1;
}

DEVICE_INLINE void findLeafToExpand(int numberOfElements, int & index, float & area) {
    int shflAmount = numberOfElements / 2;
    while (numberOfElements > 1) {
        int otherIndex = __shfl_down(index, shflAmount);
        float otherArea = __shfl_down(area, shflAmount);
        if (otherArea > area) {
            area = otherArea;
            index = otherIndex;
        }
        numberOfElements = (numberOfElements + 1) / 2;
        shflAmount = numberOfElements / 2;
    }
}

DEVICE_INLINE void formTreelet(
    int treeletRootIndex,
    int numberOfReferences,
    int treeletSize,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    float * surfaceAreas,
    int * treeletInternalNodes,
    int * treeletLeaves,
    float * treeletLeavesAreas
) {

    // Initialize treelet.
    int left = nodeLeftIndices[treeletRootIndex];
    int right = nodeRightIndices[treeletRootIndex];
    float areaLeft = surfaceAreas[left];
    float areaRight = surfaceAreas[right];
    if (THREAD_WARP_INDEX == 0) {
        treeletInternalNodes[0] = treeletRootIndex;
        treeletLeaves[0] = left;
        treeletLeaves[1] = right;
        treeletLeavesAreas[0] = areaLeft;
        treeletLeavesAreas[1] = areaRight;
    }

    WARP_SYNC;

    // Find the treelet's internal nodes. On each iteration we choose the leaf with 
    // largest surface area and add move it to the list of internal nodes, adding its
    // two children as leaves in its place.
    for (int iteration = 0; iteration < treeletSize - 2; ++iteration) {

        // Choose leaf with the largest area.
        int largestLeafIndex;
        float largestLeafArea;
        if (THREAD_WARP_INDEX < 2 + iteration) {
            largestLeafIndex = THREAD_WARP_INDEX;
            largestLeafArea = treeletLeavesAreas[THREAD_WARP_INDEX];
            if (isLeaf(treeletLeaves[largestLeafIndex], numberOfReferences))
                largestLeafArea = -MAX_FLOAT;
        }
        findLeafToExpand(2 + iteration, largestLeafIndex, largestLeafArea);

        // Update treelet.
        if (THREAD_WARP_INDEX == 0) {
            int replace = treeletLeaves[largestLeafIndex];
            int left = nodeLeftIndices[replace];
            int right = nodeRightIndices[replace];
            float areaLeft = surfaceAreas[left];
            float areaRight = surfaceAreas[right];

            treeletInternalNodes[iteration + 1] = replace;
            treeletLeaves[largestLeafIndex] = left;
            treeletLeaves[iteration + 2] = right;
            treeletLeavesAreas[largestLeafIndex] = areaLeft;
            treeletLeavesAreas[iteration + 2] = areaRight;
        }

        WARP_SYNC;
    }

}

DEVICE_INLINE void formTreeletFast(
    int treeletRootIndex,
    int numberOfReferences,
    int treeletSize,
    int * nodeLeftIndices,
    int * nodeRightIndices,
    int * treeletSizes,
    float * surfaceAreas,
    int * treeletInternalNodes,
    int * treeletLeaves,
    float * treeletLeavesAreas,
    int * treeletLeavesSizes
) {

    // Initialize treelet.
    int left = nodeLeftIndices[treeletRootIndex];
    int right = nodeRightIndices[treeletRootIndex];
    int sizeLeft = treeletSizes[left];
    int sizeRight = treeletSizes[right];
    float areaLeft = surfaceAreas[left];
    float areaRight = surfaceAreas[right];
    if (THREAD_WARP_INDEX == 0) {
        treeletInternalNodes[0] = treeletRootIndex;
        treeletLeaves[0] = left;
        treeletLeaves[1] = right;
        treeletLeavesSizes[0] = sizeLeft;
        treeletLeavesSizes[1] = sizeRight;
        treeletLeavesAreas[0] = areaLeft;
        treeletLeavesAreas[1] = areaRight;
    }

    WARP_SYNC;

    // Find the treelet's internal nodes. On each iteration we choose the leaf with 
    // largest surface area and add move it to the list of internal nodes, adding its
    // two children as leaves in its place.
    for (int iteration = 0; iteration < treeletSize - 2; ++iteration) {

        // Choose leaf with the largest area.
        int largestLeafIndex;
        float largestLeafArea;
        if (THREAD_WARP_INDEX < 2 + iteration) {
            largestLeafIndex = THREAD_WARP_INDEX;
            largestLeafArea = treeletLeavesAreas[THREAD_WARP_INDEX];
            if (isLeaf(treeletLeaves[largestLeafIndex], numberOfReferences)
                || isTreeletRoot(treeletLeavesSizes[largestLeafIndex]))
                largestLeafArea = -MAX_FLOAT;
        }
        findLeafToExpand(2 + iteration, largestLeafIndex, largestLeafArea);

        // Update treelet.
        if (THREAD_WARP_INDEX == 0) {
            int replace = treeletLeaves[largestLeafIndex];
            int left = nodeLeftIndices[replace];
            int right = nodeRightIndices[replace];
            int sizeLeft = treeletSizes[left];
            int sizeRight = treeletSizes[right];
            float areaLeft = surfaceAreas[left];
            float areaRight = surfaceAreas[right];
            treeletInternalNodes[iteration + 1] = replace;
            treeletLeaves[largestLeafIndex] = left;
            treeletLeaves[iteration + 2] = right;
            treeletLeavesSizes[largestLeafIndex] = sizeLeft;
            treeletLeavesSizes[iteration + 2] = sizeRight;
            treeletLeavesAreas[largestLeafIndex] = areaLeft;
            treeletLeavesAreas[iteration + 2] = areaRight;
        }

        WARP_SYNC;
    }

}
