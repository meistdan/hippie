/**
* \file	  SBVHBuilder.h
* \author Daniel Meister
* \date	  2019/04/17
* \brief  SBVHBuilder class header file.
*/

#ifndef _SBVH_BUILDER_H_
#define _SBVH_BUILDER_H_

#include "BVH.h"
#include "BVHBuilder.h"

#define ENABLE_SPATIAL_SPLITS 1
#define ENABLE_PRINT_PROGRESS 1
#define MAX_DEPTH 48
#define NUMBER_OF_BINS 128

class SBVHBuilder : public BVHBuilder {

private:

    struct Reference {
        int index;
        AABB box;
        Reference(void) : index(-1) {}
    };

    struct NodeSpec {
        int size;
        AABB box;
        NodeSpec(void) : size(0) {}
    };

    struct ObjectSplit {
        float sah;
        int axis;
        int leftCount;
        AABB leftBox;
        AABB rightBox;
        ObjectSplit(void) : sah(MAX_FLOAT), axis(0), leftCount(0) {}
    };

    struct SpatialSplit {
        float sah;
        int axis;
        float position;
        SpatialSplit(void) : sah(MAX_FLOAT), axis(0), position(0.0f) {}
    };

    struct Bin {
        AABB box;
        int enter;
        int exit;
    };

    struct Comparator {
        unsigned char axis;
        Comparator(unsigned char axis) : axis(axis) {}
        bool operator() (const Reference & left, const Reference & right) {
            float leftValue = left.box.mn[axis] + left.box.mx[axis];
            float rightValue = right.box.mn[axis] + right.box.mx[axis];
            return leftValue < rightValue || (leftValue == rightValue && left.index < right.index);
        }
    };

    Scene * scene;
    BVH * bvh;

    QVector<int> triangleIndices;
    QVector<Reference> references;
    QVector<AABB> rightBoxes;
    Bin bins[3][NUMBER_OF_BINS];
    int progress;
    int duplicates;

    float alpha;

    ObjectSplit findObjectSplit(const NodeSpec & spec, float nodeSAH);
    void performObjectSplit(NodeSpec & left, NodeSpec & right, const NodeSpec & spec, const ObjectSplit & split);

    SpatialSplit findSpatialSplit(const NodeSpec & spec, float nodeSAH);
    void performSpatialSplit(NodeSpec & left, NodeSpec & right, const NodeSpec & spec, const SpatialSplit & split);
    void splitReference(Reference & left, Reference & right, Reference & ref, int dim, float pos);

    BVH::LeafNode * buildLeaf(const NodeSpec & spec);
    BVH::Node * buildNode(NodeSpec spec, int level, float progressStart, float progressEnd);

    void reorderTriangleIndices(BVH::Node * node);

public:

    SBVHBuilder(void);
    virtual ~SBVHBuilder(void);

    BVH * buildSBVH(Scene * scene);
    virtual HipBVH * build(Scene * scene);
    virtual HipBVH * build(Scene * scene, float & time);
    virtual float rebuild(HipBVH & bvh);

    float getAlpha(void);
    void setAlpha(float alpha);

    virtual void clear(void);

};

#endif /* _SBVH_BUILDER_H_ */
