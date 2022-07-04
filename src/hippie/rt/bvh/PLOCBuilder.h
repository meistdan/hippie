/**
 * \file	PLOCBuilder.h
 * \author	Daniel Meister
 * \date	2015/10/29
 * \brief	PLOCBuilder class header file.
 */

#ifndef _PLOC_BUILDER_H_
#define _PLOC_BUILDER_H_

#include "BVHCollapser.h"
#include "LBVHBuilder.h"
#include "PLOCBuilderKernels.h"

class PLOCBuilder : public LBVHBuilder {

private:

    HipCompiler plocCompiler;

    Buffer blockOffsets;
    Buffer prefixScanOffset;
    Buffer nodeIndices[2];
    Buffer neighbourDistances;
    Buffer neighbourIndices;

    bool sortSwap;
    bool adaptive;
    int radius;
    int maxRadius;

    void allocate(int numberOfReferences);

    float clustering(HipBVH & bvh, int numberOfReferences);
    float build(HipBVH & bvh, Scene * scene);

public:

    PLOCBuilder(void);
    virtual ~PLOCBuilder(void);

    virtual HipBVH * build(Scene * scene);
    virtual HipBVH * build(Scene * scene, float & time);
    virtual float rebuild(HipBVH & bvh);

    bool isAdaptive(void);
    void setAdaptive(bool adaptive);
    int getRadius(void);
    void setRadius(int radius);
    int getMaxRadius(void);
    void setMaxRadius(int maxRadius);

    virtual void clear(void);

};

#endif /* _PLOC_BUILDER_H_ */
