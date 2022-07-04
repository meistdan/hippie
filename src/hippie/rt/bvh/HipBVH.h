/**
 * \file	HipBVH.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	HipBVH class header file.
 */

#ifndef _HIP_BVH_H_
#define _HIP_BVH_H_

#include "gpu/Buffer.h"
#include "gpu/HipCompiler.h"
#include "rt/scene/Scene.h"
#include "util/AABB.h"

#define CT 3.0f
#define CI 3.0f

class HipBVH {

public:

    enum Layout {
        BIN = 2,
        QUAD = 4,
        OCT = 8
    };

private:

    Layout layout;
    HipCompiler compiler;
    Scene * scene;

    float ct;
    float ci;

    QVector<float> leafSizeHistogram;

    int numberOfLeafNodes;
    int numberOfInteriorNodes;

    int lastNodeSizeThreshold;

    Buffer nodes;
    Buffer woopTriangles;
    Buffer triangleIndices;
    Buffer termCounters;

    void resize(int numberOfReferences);

    QString getLayoutString(void);
    int getNodeSize(void);

    float refitLeaves(void);
    float refitInteriors(void);
    float refit(void);
    float woopifyTriangles(void);

    void generateColors(Buffer & nodeColors);
    void colorizeTriangles(Buffer & nodeColors, int nodeSizeThreshold);

    template <typename HipBVHNode>
    bool validate(void);

public:

    HipBVH(Scene * scene);
    ~HipBVH(void);

    Layout getLayout(void);

    float getCost(void);
    float getCt(void);
    float getCi(void);

    float getAvgLeafSize(void);
    const QVector<float> & getLeafSizeHistogram(void);

    int getNumberOfNodes(void);
    int getNumberOfLeafNodes(void);
    int getNumberOfInteriorNodes(void);
    int getNumberOfReferences(void);

    Buffer & getNodes(void);
    Buffer & getWoopTriangles(void);
    Buffer & getTriangleIndices(void);

    float update(void);
    void colorizeScene(int nodeSizeThreshold);

    bool validate(void);
    bool spatialSplits(void);

    friend class ATRBuilder;
    friend class BVHBuilder;
    friend class BVHCollapser;
    friend class BVHConverter;
    friend class HLBVHBuilder;
    friend class InsertionBuilder;
    friend class LBVHBuilder;
    friend class PLOCBuilder;
    friend class SBVHBuilder;
    friend class TRBuilder;

};

#endif /* _HIP_BVH_H_ */
