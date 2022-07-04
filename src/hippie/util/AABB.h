/**
 * \file	AABB.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Axis aligned bounding box structure header file.
 */

#ifndef _AABB_H_
#define _AABB_H_

#include "Globals.h"
#include "util/Math.h"

class AABB {

public:

    Vec3f mn;
    Vec3f mx;

    HOST_DEVICE_INLINE AABB(void) : mn(MAX_FLOAT), mx(-MAX_FLOAT) {}
    HOST_DEVICE_INLINE AABB(const AABB & a, const AABB & b) : mn(MAX_FLOAT), mx(-MAX_FLOAT) { grow(a); grow(b); }
    HOST_DEVICE_INLINE AABB(const Vec3f & _mn, const Vec3f & _mx) : mn(_mn), mx(_mx) {}
    HOST_DEVICE_INLINE void reset(void) { mn = Vec3f(MAX_FLOAT); mx = Vec3f(-MAX_FLOAT); }

    HOST_DEVICE_INLINE void grow(const Vec3f & pt) {
        mn = min(mn, pt);
        mx = max(mx, pt);
    }

    HOST_DEVICE_INLINE void grow(const AABB & box) { grow(box.mn); grow(box.mx); }
    HOST_DEVICE_INLINE void intersect(const AABB & box) { mn = max(mn, box.mn); mx = min(mx, box.mx); }
    HOST_DEVICE_INLINE void enlarge(const Vec3f & add) { mn -= add; mx += add; }
    HOST_DEVICE_INLINE float volume(void) const { if (!valid()) return 0.0f; return (mx.x - mn.x) * (mx.y - mn.y) * (mx.z - mn.z); }

    HOST_DEVICE_INLINE float area(void) const { if (!valid()) return 0.0f; Vec3f d = diagonal(); return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x); }
    HOST_DEVICE_INLINE float radius(void) const { if (!valid()) return 0.0f; return 0.5f * length(diagonal()); }

    HOST_DEVICE_INLINE bool longest(void) const { Vec3f d = diagonal(); return fmaxf(fmaxf(d.x, d.y), d.z); }
    HOST_DEVICE_INLINE bool valid(void) const { return mn.x <= mx.x && mn.y <= mx.y && mn.z <= mx.z; }
    HOST_DEVICE_INLINE Vec3f centroid(void) const { return (mn + mx) * 0.5f; }
    HOST_DEVICE_INLINE Vec3f diagonal(void) const { return mx - mn; }

    HOST_DEVICE_INLINE void scale(float alpha) {
        Vec3f center = 0.5f * (mx + mn);
        Vec3f tran = alpha * 0.5f * (mx - mn);
        mn = center - tran;
        mx = center + tran;
    }

    HOST_DEVICE_INLINE bool operator==(const AABB & box) const {
        return mn.x == box.mn.x && mn.y == box.mn.y && mn.z == box.mn.z
            && mx.x == box.mx.x && mx.y == box.mx.y && mx.z == box.mx.z;
    }

    HOST_DEVICE_INLINE bool operator!=(const AABB & box) const {
        return !(*this == box);
    }

    HOST_DEVICE_INLINE friend bool overlap(const AABB & a, const AABB & b) {
        if (a.mx.x < b.mn.x ||
            a.mn.x > b.mx.x ||
            a.mx.y < b.mn.y ||
            a.mn.y > b.mx.y ||
            a.mx.z < b.mn.z ||
            a.mn.z > b.mx.z)
            return false;
        return true;
    }

};

#endif /* _AABB_H_ */
