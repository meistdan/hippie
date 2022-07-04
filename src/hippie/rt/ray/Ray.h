/**
  * \file	Ray.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	Ray structure header file.
  */

#ifndef _RAY_H_
#define _RAY_H_

#include "Globals.h"
#include "util/Math.h"

#define RAY_NO_HIT (-1)

struct Ray {

    HOST_DEVICE_INLINE Ray(void) : origin(0.0f), tmin(0.0f), direction(0.0f), tmax(0.0f) {}
    HOST_DEVICE_INLINE Ray(const Vec3f & origin, const Vec3f & direction) : origin(origin), tmin(0.0f), direction(direction), tmax(0.0f) {}

    Vec3f origin;
    float tmin;
    Vec3f direction;
    float tmax;

};

struct RayResult {
    HOST_DEVICE_INLINE RayResult(int id = RAY_NO_HIT, float t = 0.0f) : id(id), t(t) {}
    HOST_DEVICE_INLINE bool hit(void) const { return (id != RAY_NO_HIT); }
    HOST_DEVICE_INLINE void clear(void) { id = RAY_NO_HIT; }
    int id;
    float t;
    float u;
    float v;
};

struct RayStats {
    int traversedNodes;
    int testedTriangles;
};

#endif /* _RAY_H_ */
