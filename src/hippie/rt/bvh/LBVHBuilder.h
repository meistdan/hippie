/**
 * \file	LBVHBuilder.h
 * \author	Daniel Meister
 * \date	2015/11/27
 * \brief	LBVHBuilder class header file.
 */

#ifndef _LBVH_BULDER_H_
#define _LBVH_BULDER_H_

#include "BVHBuilder.h"
#include "BVHCollapser.h"

class LBVHBuilder : public BVHBuilder {

protected:

    HipCompiler compiler;
    BVHCollapser collapser;

    Buffer spine;
    Buffer triangleIndices;
    Buffer mortonCodes[2];

    Buffer nodeLeftIndices;
    Buffer nodeRightIndices;
    Buffer nodeBoxesMin;
    Buffer nodeBoxesMax;
    Buffer nodeParentIndices;

    Buffer referenceBoxesMin;
    Buffer referenceBoxesMax;
    Buffer referenceIndices[2];

    int mortonCodeBits;

    void allocate(int numberOfReferences);

    float setupReferences(HipBVH & bvh, Scene * scene, int & numberOfReferences);
    float computeMortonCodes(Scene * scene, int numberOfReferences);
    float sortReferences(int numberOfReferences);
    float setupLeaves(HipBVH & bvh, int numberOfReferences);
    float construct(int numberOfReferences);
    float refit(HipBVH & bvh, int numberOfReferences);
    virtual float build(HipBVH & bvh, Scene * scene);

public:

    LBVHBuilder(void);
    virtual ~LBVHBuilder(void);

    virtual HipBVH * build(Scene * scene);
    virtual HipBVH * build(Scene * scene, float & time);
    virtual float rebuild(HipBVH & bvh);

    int getMortonCodeBits(void);
    void setMortonCodeBits(int mortonCodeBits);

    virtual void clear(void);

};

#endif /* _LBVH_BULDER_H_ */
