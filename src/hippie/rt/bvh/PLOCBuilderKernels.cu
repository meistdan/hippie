/**
 * \file	PLOCBuilderKernels.cu
 * \author	Daniel Meister
 * \date	2015/10/29
 * \brief	PLOCBuilder kernels soruce file.
 */

#include "rt/bvh/PLOCBuilderKernels.h"
#include "rt/bvh/HipBVHUtil.h"

extern "C" GLOBAL void generateNeighboursCached(
    const int numberOfClusters,
    const int radius,
    float * neighbourDistances,
    int * neighbourIndices,
    Vec4f * nodeBoxesMin,
    Vec4f * nodeBoxesMax,
    int * nodeIndices
) {

    // Shared memory cache.
    __shared__ char cache[sizeof(AABB) * 2 * PLOC_GEN_BLOCK_THREADS];
    AABB * boxes = ((AABB*)cache) + blockDim.x / 2;

    // Thread index.
    const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Block offset.
    const int blockOffset = blockDim.x * blockIdx.x;

    // Load boxes.
    for (int neighbourIndex = int(threadIdx.x) - radius; neighbourIndex < int(blockDim.x) + radius; neighbourIndex += blockDim.x) {

        // Cluster index.
        int clusterIndex = neighbourIndex + blockOffset;

        // Valid threads.
        if (clusterIndex >= 0 && clusterIndex < numberOfClusters) {

            // Node index.
            int nodeIndex = nodeIndices[clusterIndex];

            // Cluster bounding box.
            const Vec4f boxMin = nodeBoxesMin[nodeIndex];
            const Vec4f boxMax = nodeBoxesMax[nodeIndex];
            boxes[neighbourIndex] = AABB(Vec3f(boxMin), Vec3f(boxMax));

        }

        // Dummy large boxes.
        else {
            boxes[neighbourIndex] = AABB(Vec3f(-MAX_FLOAT), Vec3f(MAX_FLOAT));
        }

    }

    // Sync.
    __syncthreads();

    // Nearest neighbour.
    int minIndex = -1;
    float minDistance = MAX_FLOAT;

    // Cluster box.
    AABB box = boxes[threadIdx.x];

    // Search left.
    for (int neighbourIndex = int(threadIdx.x) - radius; neighbourIndex < int(threadIdx.x); ++neighbourIndex) {

        // Box.
        AABB neighbourBox = boxes[neighbourIndex];

        // Grow.
        neighbourBox.grow(box);

        // Surface area.
        const float distance = neighbourBox.area();

        // Update distance.
        if (minDistance > distance) {
            minIndex = blockOffset + neighbourIndex;
            minDistance = distance;
        }

    }

    // Search right.
    for (int neighbourIndex = threadIdx.x + 1; neighbourIndex < threadIdx.x + radius + 1; ++neighbourIndex) {

        // Box.
        AABB neighbourBox = boxes[neighbourIndex];

        // Grow.
        neighbourBox.grow(box);

        // Surface area.
        const float distance = neighbourBox.area();

        // Update distance.
        if (minDistance > distance) {
            minIndex = blockOffset + neighbourIndex;
            minDistance = distance;
        }

    }

    // Save proposal.
    if (threadIndex < numberOfClusters) {
        const int nodeIndex = nodeIndices[threadIndex];
        neighbourDistances[nodeIndex] = minDistance;
        neighbourIndices[nodeIndex] = minIndex;
    }

}

extern "C" GLOBAL void generateNeighbours(
    const int numberOfClusters,
    const int radius,
    float * neighbourDistances,
    int * neighbourIndices,
    Vec4f * nodeBoxesMin,
    Vec4f * nodeBoxesMax,
    int * nodeIndices
) {

    // Thread index.
    const int clusterIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (clusterIndex < numberOfClusters) {

        // Node index.
        const int nodeIndex = nodeIndices[clusterIndex];

        // Cluster bounding box.
        const Vec4f boxMin = nodeBoxesMin[nodeIndex];
        const Vec4f boxMax = nodeBoxesMax[nodeIndex];
        AABB box = AABB(Vec3f(boxMin), Vec3f(boxMax));

        // Nearest neighbour.
        int minIndex = -1;
        float minDistance = MAX_FLOAT;

        // Search left.
        for (int neighbourIndex = max(0, clusterIndex - radius); neighbourIndex < clusterIndex; ++neighbourIndex) {

            // Neighbour node index.
            const int neighbourNodeIndex = nodeIndices[neighbourIndex];

            // Box.
            Vec4f neighbourBoxMin = nodeBoxesMin[neighbourNodeIndex];
            Vec4f neighbourBoxMax = nodeBoxesMax[neighbourNodeIndex];
            AABB neighbourBox = AABB(Vec3f(neighbourBoxMin), Vec3f(neighbourBoxMax));
            neighbourBox.grow(box);

            // Surface area.
            const float distance = neighbourBox.area();

            // Update distance.
            if (minDistance > distance) {
                minDistance = distance;
                minIndex = neighbourIndex;
            }

        }

        // Search right.
        for (int neighbourIndex = clusterIndex + 1; neighbourIndex < min(numberOfClusters, clusterIndex + radius + 1); ++neighbourIndex) {

            // Neighbour node index.
            const int neighbourNodeIndex = nodeIndices[neighbourIndex];

            // Box.
            Vec4f neighbourBoxMin = nodeBoxesMin[neighbourNodeIndex];
            Vec4f neighbourBoxMax = nodeBoxesMax[neighbourNodeIndex];
            AABB neighbourBox = AABB(Vec3f(neighbourBoxMin), Vec3f(neighbourBoxMax));
            neighbourBox.grow(box);

            // Surface area.
            const float distance = neighbourBox.area();

            // Update distance.
            if (minDistance > distance) {
                minDistance = distance;
                minIndex = neighbourIndex;
            }

        }

        // Save proposal.
        neighbourDistances[nodeIndex] = minDistance;
        neighbourIndices[nodeIndex] = minIndex;

    }

}

extern "C" GLOBAL void merge(
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
) {

    // Thread index.
    const int clusterIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Warp thread index.
    const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

    if (clusterIndex < numberOfClusters) {

        // Merging flag.
        bool merging = false;

        // Neighbour indices.
        const int leftNodeIndex = nodeIndices0[clusterIndex];
        const int neighbourIndex = neighbourIndices[leftNodeIndex];
        const int rightNodeIndex = nodeIndices0[neighbourIndex];
        const int neighbourNeighbourIndex = neighbourIndices[rightNodeIndex];

        // Merge only mutually paired clusters.
        if (clusterIndex == neighbourNeighbourIndex) {
            if (clusterIndex < neighbourIndex) merging = true;
        }

        // Just copy the node index.
        else {
            nodeIndices1[clusterIndex] = leftNodeIndex;
        }

        // Prefix scan.
        const unsigned int warpBallot = __ballot(merging);
        const int warpCount = __popc(warpBallot);
        const int warpIndex = __popc(warpBallot & ((1u << warpThreadIndex) - 1));

        // Add count of components to the global counter.
        int warpOffset = 0;
        if (warpThreadIndex == 0)
            warpOffset = atomicAdd(&prefixScanOffset, warpCount);

        // Exchange offset between threads.
        warpOffset = __shfl(warpOffset, 0);

        // Node index.
        const int nodeIndex = nodeOffset - warpOffset - warpIndex;

        // Merge.
        if (merging) {

            // Box min.
            Vec4f leftNodeBoxMin = nodeBoxesMin[leftNodeIndex];
            Vec4f rightNodeBoxMin = nodeBoxesMin[rightNodeIndex];
            leftNodeBoxMin.x = fminf(leftNodeBoxMin.x, rightNodeBoxMin.x);
            leftNodeBoxMin.y = fminf(leftNodeBoxMin.y, rightNodeBoxMin.y);
            leftNodeBoxMin.z = fminf(leftNodeBoxMin.z, rightNodeBoxMin.z);
            nodeBoxesMin[nodeIndex] = leftNodeBoxMin;

            // Box max.
            Vec4f leftNodeBoxMax = nodeBoxesMax[leftNodeIndex];
            Vec4f rightNodeBoxMax = nodeBoxesMax[rightNodeIndex];
            leftNodeBoxMax.x = fmaxf(leftNodeBoxMax.x, rightNodeBoxMax.x);
            leftNodeBoxMax.y = fmaxf(leftNodeBoxMax.y, rightNodeBoxMax.y);
            leftNodeBoxMax.z = fmaxf(leftNodeBoxMax.z, rightNodeBoxMax.z);
            nodeBoxesMax[nodeIndex] = leftNodeBoxMax;

            // Parent indices.
            nodeParentIndices[leftNodeIndex] = nodeIndex;
            nodeParentIndices[rightNodeIndex] = nodeIndex;

            // Node.
            nodeLeftIndices[nodeIndex] = leftNodeIndex;
            nodeRightIndices[nodeIndex] = rightNodeIndex;

            // Update node index.
            nodeIndices1[clusterIndex] = nodeIndex;
            nodeIndices1[neighbourIndex] = -1;

        }

    }

}

extern "C" GLOBAL void localPrefixScan(
    const int numberOfClusters,
    int * nodeIndices,
    int * threadOffsets,
    int * blockOffsets
) {

    // Thread index.
    const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Cache.
    __shared__ volatile int blockCache[2 * PLOC_SCAN_BLOCK_THREADS];

    // Read value.
    int threadValue = 0;
    if (threadIndex < numberOfClusters)
        threadValue = nodeIndices[threadIndex] >= 0;

    // Block scan.
    int blockSum = threadValue;
    blockScan<PLOC_SCAN_BLOCK_THREADS>(blockSum, blockCache);
    blockSum -= threadValue;

    // Write value.
    if (threadIndex < numberOfClusters)
        threadOffsets[threadIndex] = blockSum;

    // Write block value.
    if (threadIdx.x == 0)
        blockOffsets[blockIdx.x] = blockCache[2 * PLOC_SCAN_BLOCK_THREADS - 1];

}

extern "C" GLOBAL void globalPrefixScan(
    const int numberOfBlocks,
    int * blockOffsets
) {

    // Block end.
    const int blockEnd = divCeil(numberOfBlocks, PLOC_SCAN_BLOCK_THREADS) * PLOC_SCAN_BLOCK_THREADS;

    // Cache.
    __shared__ volatile int blockCache[2 * PLOC_SCAN_BLOCK_THREADS];

    if (blockIdx.x == 0) {

        // Block offset.
        int blockOffset = 0;

        for (int blockIndex = threadIdx.x; blockIndex < blockEnd; blockIndex += PLOC_SCAN_BLOCK_THREADS) {

            // Sync. global memory writes.
            __threadfence();

            // Read value.
            int blockValue = 0;
            if (blockIndex < numberOfBlocks)
                blockValue = blockOffsets[blockIndex];

            // Block scan.
            int blockSum = blockValue;
            blockScan<PLOC_SCAN_BLOCK_THREADS>(blockSum, blockCache);
            blockSum -= blockValue;

            // Write value.
            if (blockIndex < numberOfBlocks)
                blockOffsets[blockIndex] = blockSum + blockOffset;

            // Update block offset.
            blockOffset += blockCache[2 * PLOC_SCAN_BLOCK_THREADS - 1];

        }

    }

}

extern "C" GLOBAL void compact(
    const int numberOfClusters,
    int * nodeIndices0,
    int * nodeIndices1,
    int * blockOffsets,
    int * threadOffsets
) {

    // Thread index.
    const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Only valid clusters.
    if (threadIndex < numberOfClusters) {

        // Compact.
        const int nodeIndex = nodeIndices0[threadIndex];
        const int newClusterIndex = blockOffsets[blockIdx.x] + threadOffsets[threadIndex];
        if (nodeIndex >= 0)
            nodeIndices1[newClusterIndex] = nodeIndex;

    }

}
