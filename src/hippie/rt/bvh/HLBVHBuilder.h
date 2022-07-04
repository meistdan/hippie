/**
 * \file	HLBVHBuilder.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	HLBVHBuilder class header file.
 */

#ifndef _HLBVH_BUILDER_H_
#define _HLBVH_BUILDER_H_

#include "LBVHBuilder.h"
#include "HipBVH.h"
#include "HLBVHBuilderKernels.h"
#include "gpu/HipCompiler.h"
#include "rt/TaskQueue.h"
#include "rt/scene/Scene.h"

class HLBVHQueue : public TaskQueue<HLBVHTask> {

protected:

    Buffer bins[2];
    Buffer newTaskIndices;
    Buffer splitIndices;

public:

    HLBVHQueue(void);
    virtual ~HLBVHQueue(void);

    void init(int size, int binSize);
    virtual void clear(void);

    Buffer & getBinsBuffer(int i);
    Buffer & getNewTaskIndicesBuffer(void);
    Buffer & getSplitIndicesBuffer(void);

};

class HLBVHBuilder : public LBVHBuilder {

private:

    HipCompiler hlbvhCompiler;

    Buffer leafClusterIndices;
    Buffer nodeStates;
    Buffer nodeOffsets;

    Buffer nodeParentIndices;
    Buffer nodeLeftIndices;
    Buffer nodeRightIndices;

    Buffer clusterTaskIndices;
    Buffer clusterNodeIndices;
    Buffer clusterBinIndices;

    HLBVHQueue queue;

    int mortonCodeSAHBits;
    int numberOfClusters;

    void allocate(int numberOfReferences);

    float computeNodeStates(HipBVH & bvh, int numberOfReferences);
    float computeLeafClusterIndices(int numberOfReferences);
    float invalidateIntermediateClusters(HipBVH & bvh, int numberOfReferences);
    float computeNodeOffsets(int numberOfReferences);
    float compact(int numberOfReferences);

    float computeClusters(int numberOfReferences);
    float refit(HipBVH & bvh, int numberOfReferences);
    float split(void);
    float build(HipBVH & bvh, Scene * scene);

public:

    HLBVHBuilder(void);
    virtual ~HLBVHBuilder(void);

    int getMortonCodeSAHBits(void);
    void setMortonCodeSAHBits(int mortonCodeSAHBits);

    virtual HipBVH * build(Scene * scene);
    virtual HipBVH * build(Scene * scene, float & time);
    virtual float rebuild(HipBVH & bvh);

    virtual void clear(void);

};

#endif /* __HLBVH_BUILDER_H_ */
