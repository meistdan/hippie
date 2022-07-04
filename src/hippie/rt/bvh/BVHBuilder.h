/**
 * \file	BVHBuilder.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	BVHBuilder interface header file.
 */

#ifndef _BVH_BUILDER_H_
#define _BVH_BUILDER_H_

#include "environment/AppEnvironment.h"
#include "HipBVH.h"
#include "util/Logger.h"
#include <QStack>

class BVHBuilder {

protected:

    bool presplitting;
    bool adaptiveLeafSize;
    int maxLeafSize;

public:

    BVHBuilder(void) : maxLeafSize(8), adaptiveLeafSize(true), presplitting(false) {
        int _maxLeafSize;
        Environment::getInstance()->getIntValue("Bvh.maxLeafSize", _maxLeafSize);
        setMaxLeafSize(_maxLeafSize);
        Environment::getInstance()->getBoolValue("Bvh.adaptiveLeafSize", adaptiveLeafSize);
        Environment::getInstance()->getBoolValue("Bvh.presplitting", presplitting);
    }

    virtual ~BVHBuilder(void) {
    }

    virtual HipBVH * build(Scene * scene) = 0;
    virtual HipBVH * build(Scene * scene, float & time) = 0;
    virtual float rebuild(HipBVH & bvh) = 0;

    bool getPresplitting(void) const {
        return presplitting;
    }

    void setPresplitting(bool presplitting) {
        this->presplitting = presplitting;
    }

    bool getAdaptiveLeafSize(void) const {
        return adaptiveLeafSize;
    }

    void setAdaptiveLeafSize(bool adaptiveLeafSize) {
        this->adaptiveLeafSize = adaptiveLeafSize;
    }

    int getMaxLeafSize(void) const {
        return maxLeafSize;
    }

    void setMaxLeafSize(int maxLeafSize) {
        if (maxLeafSize > 0 && maxLeafSize <= 1024) this->maxLeafSize = maxLeafSize;
        else logger(LOG_WARN) << "WARN <BVHBuilder> Maximum leaf size must be in range (0,1024].\n";
    }

    virtual void clear(void) = 0;

};

#endif /* _BVH_BUILDER_H_ */ 
