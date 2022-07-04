/**
 * \file	BVHConverter.h
 * \author	Daniel Meister
 * \date	2014/08/19
 * \brief	BVHConverter class header file.
 */

#ifndef _BVH_CONVERTER_H_
#define _BVH_CONVERTER_H_

#include "BVH.h"
#include "HipBVH.h"
#include "rt/scene/Scene.h"
#include <QQueue>

class BVHConverter {

private:

    Vec4f woopifiedTriangle[3];

    template <typename HipBVHNode>
    void convert(HipBVH & cbvh, BVH & bvh);

    void woopifyTriangle(BVH & bvh, Scene * scene, int index);
    void woopifyTriangles(HipBVH & cbvh, BVH & bvh);

    void writeTriangleIndices(BVH::Node * node, const int * triangleIndicesSrc,
        int * triangleIndicesDst, int & offset);

public:

    BVHConverter(void);
    ~BVHConverter(void);

    void convertAdaptive(BVH & bvh, HipBVH & cbvh);
    void convert(BVH & bvh, HipBVH & cbvh, int maxLeafSize);

};

#endif /* _BVH_CONVERTER_H_ */
