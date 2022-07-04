/**
 * \file	HLBVHBin.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	HLBVHBin class header file.
 */

#ifndef _HLBVH_BIN_H_
#define _HLBVH_BIN_H_

#include "Globals.h"
#include "rt/HipUtil.h"

struct HLBVHBin {

    Vec3f mn;
    Vec3f mx;
    int clusterCounter;

    HOST_DEVICE HLBVHBin(void) {}

    HOST_DEVICE HLBVHBin(const Vec4f & mn, const Vec4f & mx) {
        this->mn = Vec3f(mn);
        this->mx = Vec3f(mx);
        clusterCounter = floatToBits(mn.w);
    }

    HOST_DEVICE_INLINE  HLBVHBin & include(const Vec4f & mn, const Vec4f & mx) {
        this->mn = min(this->mn, Vec3f(mn));
        this->mx = max(this->mx, Vec3f(mx));
        clusterCounter += floatToBits(mn.w);
        return *this;
    }

    HOST_DEVICE_INLINE  float area(void) const {
        Vec3f d = mx - mn;
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    HOST_DEVICE_INLINE  float cost(void) const {
        return clusterCounter * area();
    }

};

#endif /* _HLBVH_BIN_H_ */
