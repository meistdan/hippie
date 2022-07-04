/**
 * \file	TRBuilder.h
 * \author	Daniel Meister
 * \date	2016/03/14
 * \brief	TRBuilder class header file.
 */

#ifndef _TR_BUILDER_H_
#define _TR_BUILDER_H_

#include "LBVHBuilder.h"

#define TR_TREELET_SIZE 7 // Must be greater or equal to 7.
#define TR_ITERATIONS 2

class TRBuilder : public LBVHBuilder {

private:

    HipCompiler trCompiler;
    HipBVH * bvh;

    Buffer schedule;
    Buffer costs;
    Buffer surfaceAreas;
    Buffer subtreeReferences;

    Buffer subsetAreas;
    Buffer subsetBoxesMin;
    Buffer subsetBoxesMax;

    Buffer stackNode;
    Buffer stackMask;
    Buffer stackSize;
    Buffer currentInternalNode;

    int iterations;
    int treeletSize;

    void allocate(int numberOfReferences);

    int populationCount(int x);
    void processSubset(int subset, int superset, QVector<QSet<int>> & dependencies, QVector<QSet<int>> & subsetsBySize);
    void generateSchedule(void);

    float computeSurfaceAreas(int numberOfReferences);
    float optimize(HipBVH & bvh, int numberOfReferences);
    float build(HipBVH & bvh, Scene * scene);

public:

    TRBuilder(void);
    virtual ~TRBuilder(void);

    virtual HipBVH * build(Scene * scene);
    virtual HipBVH * build(Scene * scene, float & time);
    virtual float rebuild(HipBVH & bvh);

    int getTreeletSize(void);
    void setTreeletSize(int treeletSize);
    int getIterations(void);
    void setIterations(int iterations);

    virtual void clear(void);

};

#endif /* _TR_BUILDER_H_ */
