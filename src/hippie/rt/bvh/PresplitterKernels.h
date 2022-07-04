/**
 * \file	PresplitterKernels.h
 * \author	Daniel Meister
 * \date	2019/07/06
 * \brief	Presplitter kernels header file.
 */

#ifndef _BVH_PRESPLITTER_KERNELS_H_
#define _BVH_PRESPLITTER_KERNELS_H_

#include "Globals.h"
#include "util/Math.h"

#define REDUCTION_BLOCK_THREADS 256

struct SplitTask {
    int triangleIndex;
    int splitCount;
    HOST_DEVICE_INLINE SplitTask(void) : triangleIndex(-1), splitCount(0) {}
    HOST_DEVICE_INLINE SplitTask(int triangleIndex) : triangleIndex(triangleIndex), splitCount(0) {}
    HOST_DEVICE_INLINE SplitTask(int triangleIndex, int splitCount) : triangleIndex(triangleIndex), splitCount(splitCount) {}
};

#ifdef __KERNELCC__
extern "C" {

    DEVICE int S;
    DEVICE int prefixScanOffset;
    CONSTANT float sceneBox[6];

    GLOBAL void computePriorities(
        const int numberOfTriangles,
        Vec3i * triangles,
        Vec3f * vertices,
        float * priorities
    );

    GLOBAL void sumPriorities(
        const int numberOfTriangles,
        const float D,
        float * priorities
    );

    GLOBAL void sumPrioritiesRound(
        const int numberOfTriangles,
        const float D,
        float * priorities
    );

    GLOBAL void initSplitTasks(
        const int numberOfTriangles,
        const float D,
        float * priorities,
        Vec3i * triangles,
        Vec3f * vertices,
        Vec4f * boxesMin,
        Vec4f * boxesMax,
        SplitTask * queue
    );

    GLOBAL void split(
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
    );

}
#endif

#endif /* _BVH_PRESPLITTER_KERNELS_H_ */