/**
 * \file	PresplitterKenrels.cu
 * \author	Daniel Meister
 * \date	2019/07/06
 * \brief	Presplitter kernels soruce file.
 */

#include "rt/bvh/PresplitterKernels.h"
#include "rt/bvh/HipBVHUtil.h"

extern "C" GLOBAL void computePriorities(
    const int numberOfTriangles,
    Vec3i * triangles,
    Vec3f * vertices,
    float * priorities
) {

    // Triangle index.
    const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Scene box.
    AABB _sceneBox = *(AABB*)sceneBox;
    Vec3f scale = 1.0f / _sceneBox.diagonal();

    // Priority parameters.
    const float X = 2.0f;
    const float Y = 1.0f / 3.0f;

    if (triangleIndex < numberOfTriangles) {

        // Triangle.		
        Vec3i triangle = triangles[triangleIndex];
        Vec3f v0 = vertices[triangle.x];
        Vec3f v1 = vertices[triangle.y];
        Vec3f v2 = vertices[triangle.z];

        // Box.
        AABB box;
        box.grow(v0);
        box.grow(v1);
        box.grow(v2);

        // Quantized extreme points of bounding box.
        Vec3i imn = (box.mn - _sceneBox.mn) * scale * 2097151.9f;
        Vec3i imx = (box.mx - _sceneBox.mn) * scale * 2097151.9f;

        // Morton codes.
        unsigned long long mortonCodesMin = 0;
        unsigned long long mortonCodesMax = 0;
        for (int i = 0; i < 21; ++i) {
            mortonCodesMin |= (unsigned long long)((imn.x >> i) & 1) << (3 * i + 0);
            mortonCodesMin |= (unsigned long long)((imn.y >> i) & 1) << (3 * i + 1);
            mortonCodesMin |= (unsigned long long)((imn.z >> i) & 1) << (3 * i + 2);
            mortonCodesMax |= (unsigned long long)((imx.x >> i) & 1) << (3 * i + 0);
            mortonCodesMax |= (unsigned long long)((imx.y >> i) & 1) << (3 * i + 1);
            mortonCodesMax |= (unsigned long long)((imx.z >> i) & 1) << (3 * i + 2);
        }

        // Most significant bit in which codes differ.
        int diffBit = 8 * sizeof(unsigned long long) - __clzll(mortonCodesMin ^ mortonCodesMax) - 1;
        float i = (diffBit / 3) - 21;
        
        // Ideal surface area.
        Vec3f d1 = v0 - v1;
        Vec3f d2 = v0 - v2;
        Vec3f d1Xd2 = cross(d1, d2);
        float areaIdeal = fabs(d1Xd2.x) + fabs(d1Xd2.y) + fabs(d1Xd2.z);

        // Priority.
        priorities[triangleIndex] = powf(powf(X, i) * (box.area() - areaIdeal), Y);

    }

}

extern "C" GLOBAL void sumPriorities(
    const int numberOfTriangles,
    const float D,
    float * priorities
) {

    // Triangle index.
    const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Priority.
    float priority = 0.0f;

    // Fetch priority.
    if (triangleIndex < numberOfTriangles) {
        priority = priorities[triangleIndex];
    }

    // Cache.
    __shared__ volatile float cache[REDUCTION_BLOCK_THREADS];

    // Cost reduction.
    cache[threadIdx.x] = D * priority;
    cache[threadIdx.x] += cache[threadIdx.x ^ 1];
    cache[threadIdx.x] += cache[threadIdx.x ^ 2];
    cache[threadIdx.x] += cache[threadIdx.x ^ 4];
    cache[threadIdx.x] += cache[threadIdx.x ^ 8];
    cache[threadIdx.x] += cache[threadIdx.x ^ 16];

    __syncthreads();
    if ((threadIdx.x & 32) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 32];

    __syncthreads();
    if ((threadIdx.x & 64) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 64];

    __syncthreads();
    if ((threadIdx.x & 128) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 128];

    // Update total cost.
    if (threadIdx.x == 0) {
        atomicAdd((float*)&S, cache[threadIdx.x]);
    }
}

extern "C" GLOBAL void sumPrioritiesRound(
    const int numberOfTriangles,
    const float D,
    float * priorities
) {

    // Triangle index.
    const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int triangleEnd = (divCeilLog(numberOfTriangles, LOG_WARP_THREADS) << LOG_WARP_THREADS);

    // Warp thread index.
    const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

    if (triangleIndex < triangleEnd) {

        // Priority.
        float priority = 0.0f;

        // Fetch priority.
        if (triangleIndex < numberOfTriangles)
            priority = priorities[triangleIndex];

        // Prefix scan.
        int warpSum = warpScan(warpThreadIndex, int(D * priority));

        // Add count to the global counter.
        if (warpThreadIndex == 31)
            atomicAdd(&S, warpSum);

    }

}

extern "C" GLOBAL void initSplitTasks(
    const int numberOfTriangles,
    const float D,
    float * priorities,
    Vec3i * triangles,
    Vec3f * vertices,
    Vec4f * boxesMin,
    Vec4f * boxesMax,
    SplitTask * queue
) {

    // Triangle index.
    const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (triangleIndex < numberOfTriangles) {

        // Split.
        int splitCount = D * priorities[triangleIndex];

        // Triangle.		
        Vec3i triangle = triangles[triangleIndex];
        Vec3f v0 = vertices[triangle.x];
        Vec3f v1 = vertices[triangle.y];
        Vec3f v2 = vertices[triangle.z];

        // Box.
        AABB box;
        box.grow(v0);
        box.grow(v1);
        box.grow(v2);

        // Write task.
        queue[triangleIndex] = SplitTask(triangleIndex, splitCount);
        boxesMin[triangleIndex] = Vec4f(box.mn, 0.0f);
        boxesMax[triangleIndex] = Vec4f(box.mx, 0.0f);

    }

}

extern "C" GLOBAL void split(
    const int inputQueueSize,
    int * outputQueueSizeLoc,
    int * triangleIndices,
    Vec3i * triangles,
    Vec3f * vertices,
    Vec4f * referenceBoxesMin,
    Vec4f * referenceBoxesMax,
    Vec4f * inputBoxesMin,
    Vec4f * inputBoxesMax,
    Vec4f * outputBoxesMin,
    Vec4f * outputBoxesMax,
    SplitTask * inputQueue,
    SplitTask * outputQueue
) {

    // Task index.
    const int taskIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int taskEnd = (divCeilLog(inputQueueSize, LOG_WARP_THREADS) << LOG_WARP_THREADS);

    // Scene box.
    AABB _sceneBox = *(AABB*)sceneBox;

    // Warp thread index.
    const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);
    
    if (taskIndex < taskEnd) {

        // Input task.
        SplitTask task;

        // Split counts.
        int leftSplitCount, rightSplitCount;

        // Boxes.
        AABB box = AABB(Vec3f(inputBoxesMin[taskIndex]), Vec3f(inputBoxesMax[taskIndex]));

        // Child boxes.
        AABB leftBox, rightBox;

        // Flags.
        bool done = false;
        bool split = false;

        // Only valid tasks.
        if (taskIndex < inputQueueSize) {

            // Task.
            task = inputQueue[taskIndex];

            // Stop splitting.
            if (task.splitCount == 0) {

                // Done flag.
                done = true;

            }

            // Split.
            else {

                // Quantized extreme points of bounding box.
                Vec3i imn = (box.mn - _sceneBox.mn) / _sceneBox.diagonal() * 2097151.9f; // 21b
                Vec3i imx = (box.mx - _sceneBox.mn) / _sceneBox.diagonal() * 2097151.9f; // 21b

                // Morton codes.
                unsigned long long mortonCodesMin = 0;
                unsigned long long mortonCodesMax = 0;
                for (int i = 0; i < 21; ++i) {
                    mortonCodesMin |= (unsigned long long)((imn.x >> i) & 1) << (3 * i + 0);
                    mortonCodesMin |= (unsigned long long)((imn.y >> i) & 1) << (3 * i + 1);
                    mortonCodesMin |= (unsigned long long)((imn.z >> i) & 1) << (3 * i + 2);
                    mortonCodesMax |= (unsigned long long)((imx.x >> i) & 1) << (3 * i + 0);
                    mortonCodesMax |= (unsigned long long)((imx.y >> i) & 1) << (3 * i + 1);
                    mortonCodesMax |= (unsigned long long)((imx.z >> i) & 1) << (3 * i + 2);
                }

                // Split axis and position.
                int axis;
                float pos;

                // Most significant bit in which codes differ.
                int diffBit = 8 * sizeof(unsigned long long) - __clzll(mortonCodesMin ^ mortonCodesMax) - 1;

                // Morton code splits.
                if (diffBit >= 0) {

                    // Axis.
                    axis = diffBit % 3;

                    // Split position.
                    pos = 0.0f;
                    float delta = 0.5f;
                    for (int i = 20; i >= diffBit / 3; --i) {
                        if ((imx[axis] >> i) & 1)
                            pos += delta;
                        delta *= 0.5f;
                    }
                    pos = pos * _sceneBox.diagonal()[axis] + _sceneBox.mn[axis];

                    // Split position coincides with one of the extreme points => Spatial media.
                    if (box.mn[axis] >= pos || box.mx[axis] <= pos)
                        pos = box.centroid()[axis];

                }

                // Longest axis split.
                else {

                    // Diagonal.
                    Vec3f d = box.diagonal();

                    // Axis.
                    if (d.x > d.y && d.x > d.z) axis = 0;
                    else if (d.y > d.z) axis = 1;
                    else axis = 2;

                    // Split position.
                    pos = box.centroid()[axis];

                }

                // Triangle.		
                Vec3i triangle = triangles[task.triangleIndex];
                Vec3f v[3];
                v[0] = vertices[triangle.x];
                v[1] = vertices[triangle.y];
                v[2] = vertices[triangle.z];
                const Vec3f * v1 = &v[2];

                for (int i = 0; i < 3; i++) {
                    const Vec3f * v0 = v1;
                    v1 = &v[i];
                    float v0p = (*v0)[axis];
                    float v1p = (*v1)[axis];

                    // Insert vertex to the boxes it belongs to.
                    if (v0p <= pos)
                        leftBox.grow(*v0);
                    if (v0p >= pos)
                        rightBox.grow(*v0);

                    // Edge intersects the plane => insert intersection to both boxes.
                    if ((v0p < pos && v1p > pos) || (v0p > pos && v1p < pos)) {
                        Vec3f t = mix(*v0, *v1, clamp((pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
                        leftBox.grow(t);
                        rightBox.grow(t);
                    }
                }

                // Intersect with original bounds.
                leftBox.mx[axis] = pos;
                rightBox.mn[axis] = pos;
                leftBox.intersect(box);
                rightBox.intersect(box);

                // Distribute splits.
                float leftLongest = leftBox.longest();
                float rightLongest = rightBox.longest();
                leftSplitCount = (task.splitCount - 1) * leftLongest / (leftLongest + rightLongest) + 0.5f;
                rightSplitCount = task.splitCount - 1 - leftSplitCount;

                // Split flag.
                split = true;


            }

        }

        // Warp wide prefix scan of output tasks.
        unsigned int warpBallot = __ballot(done);
        int warpCount = __popc(warpBallot);
        int warpIndex = __popc(warpBallot & ((1u << warpThreadIndex) - 1));

        // Add count to the global counter.
        int warpOffset = 0;
        if (warpThreadIndex == 0)
            warpOffset = atomicAdd(&prefixScanOffset, warpCount);

        // Exchange offset between threads.
        warpOffset = __shfl(warpOffset, 0);

        // Index to reference buffer.
        int referenceOffset = warpOffset + warpIndex;

        // Warp wide prefix scan of output tasks.
        warpBallot = __ballot(split);
        warpCount = __popc(warpBallot) << 1;
        warpIndex = __popc(warpBallot & ((1u << warpThreadIndex) - 1)) << 1;

        // Add count to the global counter.
        if (warpThreadIndex == 0)
            warpOffset = atomicAdd(outputQueueSizeLoc, warpCount);

        // Exchange offset between threads.
        warpOffset = __shfl(warpOffset, 0);

        // Index to the input task queue.
        int taskOffset = warpOffset + warpIndex;

        // Only valid tasks.
        if (taskIndex < inputQueueSize) {

            // Write reference.
            if (task.splitCount == 0) {
                referenceBoxesMin[referenceOffset] = Vec4f(box.mn, 0.0f);
                referenceBoxesMax[referenceOffset] = Vec4f(box.mx, 0.0f);
                if (task.triangleIndex < 0)
                    task.triangleIndex = 0;
                triangleIndices[referenceOffset] = task.triangleIndex;
            }

            // Split.
            else if (split){
                outputQueue[taskOffset + 0] = SplitTask(task.triangleIndex, leftSplitCount);
                outputBoxesMin[taskOffset + 0] = Vec4f(leftBox.mn, 0.0f);
                outputBoxesMax[taskOffset + 0] = Vec4f(leftBox.mx, 0.0f);
                outputQueue[taskOffset + 1] = SplitTask(task.triangleIndex, rightSplitCount);
                outputBoxesMin[taskOffset + 1] = Vec4f(rightBox.mn, 0.0f);
                outputBoxesMax[taskOffset + 1] = Vec4f(rightBox.mx, 0.0f);
            }

        }

    }

}
