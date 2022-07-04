/**
 * \file	InsertionBuilder.h
 * \author	Daniel Meister
 * \date	2017/01/30
 * \brief	InsertionBuilder class header file.
 */

#ifndef _INSERTION_BUILDER_H_
#define _INSERTION_BUILDER_H_

#include "ATRBuilder.h"
#include "BVHCollapser.h"
#include "InsertionBuilderKernels.h"

class InsertionBuilder : public ATRBuilder {

protected:

    HipCompiler insertionCompiler;

    Buffer locks;
    Buffer areaReductions;
    Buffer outNodeIndices;

    Buffer bestNodeParentIndices;
    Buffer bestNodeLeftIndices;
    Buffer bestNodeRightIndices;

    bool sbvh;
    int mod;
    
    void allocate(int numberOfReferences);

    float optimizeInsertion(HipBVH & bvh, int numberOfReferences, float initTime);

    float buildSBVH(HipBVH& bvh, Scene* scene, int & numberOfReferences);
    float buildLBVH(HipBVH& bvh, Scene* scene, int & numberOfReferences);
    float build(HipBVH & bvh, Scene * scene);

public:

    InsertionBuilder(void);
    virtual ~InsertionBuilder(void);

    virtual HipBVH * build(Scene * scene);
    virtual HipBVH * build(Scene * scene, float & time);
    virtual float rebuild(HipBVH & bvh);

    bool isAtr(void);
    void setAtr(bool atr);
    bool isSbvh(void);
    void setSbvh(bool sbvh);
    int getMod(void);
    void setMod(int mod);
    int getMortonCodeBits(void);
    void setMortonCodeBits(int mortonCodeBits);

    virtual void clear(void);

};

#endif /* _INSERTION_BUILDER_H_ */
