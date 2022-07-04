/**
 * \file	HipBVHNode.h
 * \author	Daniel Meister
 * \date	2016/02/23
 * \brief	HipBVHNode struct header file.
 */

#ifndef _HIP_BVH_NODE_H_
#define _HIP_BVH_NODE_H_

#include "util/AABB.h"

template <int K>
class HipBVHNodeN {

private:

    int content[8 * K];

public:

    const static int N = K;

    HOST_DEVICE_INLINE HipBVHNodeN(void) {}

    HOST_DEVICE_INLINE int getParentIndex(void) {
        return content[8 * N - 1];
    }

    HOST_DEVICE_INLINE void setParentIndex(int parentIndex) {
        content[8 * N - 1] = parentIndex;
    }

    HOST_DEVICE_INLINE int getChildIndex(int i) {
        return content[4 * (N + i) + 2] < 0 ? ~content[4 * (N + i) + 2] : content[4 * (N + i) + 2];
    }

    HOST_DEVICE_INLINE void setChildIndex(int i, int childIndex) {
        content[4 * (N + i) + 2] = childIndex;
    }

    HOST_DEVICE_INLINE int getBegin(void) {
        return content[4 * N + 2];
    }

    HOST_DEVICE_INLINE void setBegin(int begin) {
        content[4 * N + 2] = begin;
    }

    HOST_DEVICE_INLINE int getEnd(void) {
        return content[4 * N + 3];
    }

    HOST_DEVICE_INLINE void setEnd(int end) {
        content[4 * N + 3] = end;
    }

    HOST_DEVICE_INLINE AABB getChildBoundingBox(int i) {
        AABB box;
        box.mn = Vec3f(bitsToFloat(content[4 * i + 0]), bitsToFloat(content[4 * i + 2]), bitsToFloat(content[4 * (N + i) + 0]));
        box.mx = Vec3f(bitsToFloat(content[4 * i + 1]), bitsToFloat(content[4 * i + 3]), bitsToFloat(content[4 * (N + i) + 1]));
        return box;
    }

    HOST_DEVICE_INLINE void setChildBoundingBox(int i, const AABB & box) {
        content[4 * i + 0] = floatToBits(box.mn.x);
        content[4 * i + 2] = floatToBits(box.mn.y);
        content[4 * (N + i) + 0] = floatToBits(box.mn.z);
        content[4 * i + 1] = floatToBits(box.mx.x);
        content[4 * i + 3] = floatToBits(box.mx.y);
        content[4 * (N + i) + 1] = floatToBits(box.mx.z);
    }

    HOST_DEVICE_INLINE AABB getBoundingBox(void) {
        AABB box;
        for (int i = 0; i < N; ++i)
            box.grow(getChildBoundingBox(i));
        return box;
    }

    HOST_DEVICE_INLINE void setBoundingBox(const AABB & box) {
        for (int i = 0; i < N; ++i)
            setChildBoundingBox(i, box);
    }

    HOST_DEVICE_INLINE bool isLeaf(void) {
        return content[8 * N - 5] < 0;
    }

    HOST_DEVICE_INLINE bool isChildLeaf(int i) {
        return getChildIndex(i) < 0;
    }

    HOST_DEVICE_INLINE float getSurfaceArea(void) {
        return getBoundingBox().area();
    }

    HOST_DEVICE_INLINE int getSize(void) {
        return content[8 * N - 5] < 0 ? ~content[8 * N - 5] : content[8 * N - 5];
    }

    HOST_DEVICE_INLINE void setSize(int size) {
        content[8 * N - 5] = size;
    }

    HOST_DEVICE_INLINE int getNumberOfChildren(void) {
        return content[8 * N - 9];
    }

    HOST_DEVICE_INLINE void setNumberOfChildren(int numberOfChildren) {
        content[8 * N - 9] = numberOfChildren;
    }

};

template<>
class HipBVHNodeN<2> {

private:

    int content[8 * 2];

public:

    const static int N = 2;

    HOST_DEVICE_INLINE HipBVHNodeN(void) {}

    HOST_DEVICE_INLINE bool isLeaf(void) {
        return content[14] < 0;
    }

    HOST_DEVICE_INLINE int getParentIndex(void) {
        return content[15];
    }

    HOST_DEVICE_INLINE void setParentIndex(int parentIndex) {
        content[15] = parentIndex;
    }

    HOST_DEVICE_INLINE int getChildIndex(int i) {
        return content[12 + i] < 0 ? ~content[12 + i] : content[12 + i];
    }

    HOST_DEVICE_INLINE void setChildIndex(int i, int childIndex) {
        content[12 + i] = childIndex;
    }

    HOST_DEVICE_INLINE int getBegin(void) {
        return getChildIndex(0);
    }

    HOST_DEVICE_INLINE void setBegin(int begin) {
        setChildIndex(0, begin);
    }

    HOST_DEVICE_INLINE int getEnd(void) {
        return getChildIndex(1);
    }

    HOST_DEVICE_INLINE void setEnd(int end) {
        setChildIndex(1, end);
    }

    HOST_DEVICE_INLINE AABB getChildBoundingBox(int i) {
        AABB box;
        box.mn = Vec3f(bitsToFloat(content[4 * i + 0]), bitsToFloat(content[4 * i + 2]), bitsToFloat(content[2 * i + 8]));
        box.mx = Vec3f(bitsToFloat(content[4 * i + 1]), bitsToFloat(content[4 * i + 3]), bitsToFloat(content[2 * i + 9]));
        return box;
    }

    HOST_DEVICE_INLINE void setChildBoundingBox(int i, const AABB & box) {
        content[4 * i + 0] = floatToBits(box.mn.x);
        content[4 * i + 2] = floatToBits(box.mn.y);
        content[2 * i + 8] = floatToBits(box.mn.z);
        content[4 * i + 1] = floatToBits(box.mx.x);
        content[4 * i + 3] = floatToBits(box.mx.y);
        content[2 * i + 9] = floatToBits(box.mx.z);
    }

    HOST_DEVICE_INLINE AABB getBoundingBox(void) {
        AABB box;
        for (int i = 0; i < 2; ++i)
            box.grow(getChildBoundingBox(i));
        return box;
    }

    HOST_DEVICE_INLINE void setBoundingBox(const AABB & box) {
        for (int i = 0; i < 2; ++i)
            setChildBoundingBox(i, box);
    }

    HOST_DEVICE_INLINE bool isChildLeaf(int i) {
        return getChildIndex(i) < 0;
    }

    HOST_DEVICE_INLINE float getSurfaceArea(void) {
        return getBoundingBox().area();
    }

    HOST_DEVICE_INLINE int getSize(void) {
        return content[14] < 0 ? ~content[14] : content[14];
    }

    HOST_DEVICE_INLINE void setSize(int size) {
        content[14] = size;
    }

    HOST_DEVICE_INLINE int getNumberOfChildren(void) {
        return 2;
    }

    HOST_DEVICE_INLINE void setNumberOfChildren(int numberOfChildren) {
    }

};

typedef HipBVHNodeN<2> HipBVHNodeBin;
typedef HipBVHNodeN<4> HipBVHNodeQuad;
typedef HipBVHNodeN<8> HipBVHNodeOct;

#endif /* _HIP_BVH_NODE_H_ */
