/**
 * \file	BVHCollapser.h
 * \author	Daniel Meister
 * \date	2016/03/15
 * \brief	BVHCollapser class header file.
 */

#ifndef _BVH_COLLAPSER_H_
#define _BVH_COLLAPSER_H_

#include "BVH.h"
#include "HipBVH.h"

class BVHCollapser {

private:

    HipCompiler compiler;

    Buffer nodeCosts;
    Buffer nodeStates;
    Buffer nodeOffsets;
    Buffer nodeSizes;

    Buffer leafIndices;
    Buffer referenceOffsets;

    Buffer nodeIndices;

    float computeSizes(
        int numberOfReferences,
        Buffer & nodeParentIndices,
        Buffer & nodeLeftIndices,
        Buffer & nodeRightIndices,
        HipBVH & bvh
    );

    float computeNodeStatesAdaptive(
        int numberOfReferences,
        Buffer & nodeParentIndices,
        Buffer & nodeLeftIndices,
        Buffer & nodeRightIndices,
        Buffer & nodeBoxesMin,
        Buffer & nodeBoxesMax,
        HipBVH & bvh
    );

    float computeNodeStates(
        int numberOfReferences,
        int maxLeafSize,
        Buffer & nodeParentIndices,
        HipBVH & bvh
    );

    float computeLeafIndices(
        int numberOfReferences,
        Buffer & nodeParents
    );

    float invalidateCollapsedNodes(
        int numberOfReferences,
        Buffer & nodeParentIndices,
        HipBVH & bvh
    );

    float computeNodeOffsets(
        int numberOfReferences,
        Buffer & nodeLeftIndices,
        Buffer & nodeRightIndices
    );

    float computeReferenceOffsets(
        int numberOfReferences
    );

    float compact(
        int numberOfReferences,
        Buffer & nodeParentIndices,
        Buffer & nodeLeftIndices,
        Buffer & nodeRightIndices,
        Buffer & nodeBoxesMin,
        Buffer & nodeBoxesMax,
        HipBVH & bvh
    );

    float reorderTriangleIndices(
        int numberOfReferences,
        Buffer & nodeLeftIndices,
        Buffer & trinagleIndices,
        HipBVH & bvh
    );

    BVH * convert(
        int numberOfReferences,
        Buffer & nodeParentIndices,
        Buffer & nodeLeftIndices,
        Buffer & nodeRightIndices,
        Buffer & nodeBoxesMin,
        Buffer & nodeBoxesMax,
        Buffer & triangleIndices
    );

    BVH* convertWide(
        int numberOfReferences,
        int n,
        Buffer& nodeParentIndices,
        Buffer& nodeChildIndices,
        Buffer& nodeChildCounts,
        Buffer& nodeBoxesMin,
        Buffer& nodeBoxesMax,
        Buffer& triangleIndices
    );

    void writeTriangleIndices(BVH::Node * node, const int * triangleIndicesSrc, int * triangleIndicesDst, int & offset);

public:

    BVHCollapser(void);
    ~BVHCollapser(void);

    float collapseAdaptive(
        int numberOfReferences,
        Buffer & nodeParentIndices,
        Buffer & nodeLeftIndices,
        Buffer & nodeRightIndices,
        Buffer & nodeBoxesMin,
        Buffer & nodeBoxesMax,
        Buffer & triangleIndices,
        HipBVH & bvh
    );

    float BVHCollapser::collapse(
        int numberOfReferences,
        int maxLeafSize,
        Buffer & nodeParentIndices,
        Buffer & nodeLeftIndices,
        Buffer & nodeRightIndices,
        Buffer & nodeBoxesMin,
        Buffer & nodeBoxesMax,
        Buffer & triangleIndices,
        HipBVH & bvh
    );

    float collapseAdaptiveWide(
        int numberOfReferences,
        Buffer& nodeParentIndices,
        Buffer& nodeChildIndices,
        Buffer& nodeChildCounts,
        Buffer& nodeBoxesMin,
        Buffer& nodeBoxesMax,
        Buffer& triangleIndices,
        HipBVH& bvh
    );

    float BVHCollapser::collapseWide(
        int numberOfReferences,
        int maxLeafSize,
        Buffer& nodeParentIndices,
        Buffer& nodeChildIndices,
        Buffer& nodeChildCounts,
        Buffer& nodeBoxesMin,
        Buffer& nodeBoxesMax,
        Buffer& triangleIndices,
        HipBVH& bvh
    );

    void clear(void);

};

#endif /* _BVH_COLLAPSER_H_ */
