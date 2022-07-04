/**
 * \file	ATRBuilder.h
 * \author	Daniel Meister
 * \date	2016/02/11
 * \brief	ATRBuilder class header file.
 */

#ifndef _ATR_BUILDER_H_
#define _ATR_BUILDER_H_

#include "LBVHBuilder.h"

#define ATR_TREELET_SIZE 20
#define ATR_ITERATIONS 2
#define ATR_MAX_TREELET_SIZE 20

class ATRBuilder : public LBVHBuilder {

protected:

    HipCompiler atrCompiler;
    HipBVH * bvh;

    Buffer schedule;
    Buffer distanceMatrices;
    Buffer costs;
    Buffer surfaceAreas;
    Buffer subtreeReferences;

    int iterations;
    int treeletSize;

    void allocate(int numberOfReferences);

    void generateSchedule(void);

    float computeSurfaceAreas(int numberOfReferences);
    float optimize(HipBVH & bvh, int numberOfReferences);
    float build(HipBVH & bvh, Scene * scene);

public:

    ATRBuilder(void);
    virtual ~ATRBuilder(void);

    virtual HipBVH * build(Scene * scene);
    virtual HipBVH * build(Scene * scene, float & time);
    virtual float rebuild(HipBVH & bvh);

    int getTreeletSize(void);
    void setTreeletSize(int treeletSize);
    int getIterations(void);
    void setIterations(int iterations);

    virtual void clear(void);

};

#endif /* _ATR_BUILDER_H_ */
