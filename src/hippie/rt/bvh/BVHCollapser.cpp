/**
 * \file	BVHCollapser.cpp
 * \author	Daniel Meister
 * \date	2016/03/15
 * \brief	BVHCollapser class source file.
 */

#include "BVHCollapser.h"
#include "BVHCollapserKernels.h"
#include "BVHConverter.h"
#include "util/Logger.h"
#include <QStack>

float BVHCollapser::computeSizes(
    int numberOfReferences,
    Buffer & nodeParentIndices,
    Buffer & nodeLeftIndices,
    Buffer & nodeRightIndices,
    HipBVH & bvh
) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computeSizes");

    // Clear termination counters.
    bvh.termCounters.clear();

    // Resize buffer.
    nodeSizes.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));

    // Set params.
    kernel.setParams(
        2 * numberOfReferences - 1,
        bvh.termCounters,
        nodeParentIndices,
        nodeLeftIndices,
        nodeRightIndices,
        nodeSizes
    );

    // Launch.
    float time = kernel.launchTimed(numberOfReferences);

    // Kernel time.
    return time;

}

float BVHCollapser::computeNodeStatesAdaptive(
    int numberOfReferences,
    Buffer & nodeParentIndices,
    Buffer & nodeLeftIndices,
    Buffer & nodeRightIndices,
    Buffer & nodeBoxesMin,
    Buffer & nodeBoxesMax,
    HipBVH & bvh
) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computeNodeStatesAdaptive");

    // Resize buffers.
    nodeStates.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
    nodeCosts.resizeDiscard(sizeof(float) * (2 * numberOfReferences - 1));

    // Clear termination counters.
    bvh.termCounters.clear();

    // Set params.
    kernel.setParams(
        numberOfReferences,
        bvh.getCi(),
        bvh.getCt(),
        bvh.termCounters,
        nodeCosts,
        nodeParentIndices,
        nodeLeftIndices,
        nodeRightIndices,
        nodeSizes,
        nodeStates,
        nodeBoxesMin,
        nodeBoxesMax
    );

    // Launch.
    float time = kernel.launchTimed(numberOfReferences);

    // Kernel time.
    return time;

}

float BVHCollapser::computeNodeStates(
    int numberOfReferences,
    int maxLeafSize,
    Buffer & nodeParentIndices,
    HipBVH & bvh
) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computeNodeStates");

    // Resize buffers.
    nodeStates.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));

    // Clear termination counters.
    bvh.termCounters.clear();

    // Set params.
    kernel.setParams(
        numberOfReferences,
        maxLeafSize,
        bvh.termCounters,
        nodeParentIndices,
        nodeSizes,
        nodeStates
    );

    // Launch.
    float time = kernel.launchTimed(numberOfReferences);

    // Kernel time.
    return time;

}


float BVHCollapser::computeLeafIndices(
    int numberOfReferences,
    Buffer & nodeParentIndices
) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computeLeafIndices");

    // Resize buffer.
    leafIndices.resizeDiscard(sizeof(int) * numberOfReferences);

    // Set params.
    kernel.setParams(
        numberOfReferences,
        leafIndices,
        nodeParentIndices,
        nodeStates
    );

    // Launch.
    float time = kernel.launchTimed(numberOfReferences);

    return time;

}

float BVHCollapser::invalidateCollapsedNodes(
    int numberOfReferences,
    Buffer & nodeParentIndices,
    HipBVH & bvh
) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("invalidateCollapsedNodes");

    // Clear termination counters.
    bvh.termCounters.clear();

    // Set params.
    kernel.setParams(
        numberOfReferences,
        bvh.termCounters,
        leafIndices,
        nodeParentIndices,
        nodeStates
    );

    // Launch.
    float time = kernel.launchTimed(numberOfReferences);

    // Kernel time.
    return time;

}

float BVHCollapser::computeNodeOffsets(
    int numberOfReferences,
    Buffer & nodeLeftIndices,
    Buffer & nodeRightIndices
) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computeNodeOffsets");

    // Clear prefix scan offsets.
    module->getGlobal("leafPrefixScanOffset").clear();
    *(int*)module->getGlobal("interiorPrefixScanOffset").getMutablePtr() = 1;

    // Resize buffer.
    nodeOffsets.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
    nodeIndices.resizeDiscard(sizeof(int) * (numberOfReferences - 1));
    nodeIndices.clearRange(0, 0, sizeof(int));

    int taskOffset = 0;
    int numberOfTasks = 1;
    float time = 0.0f;

    while (numberOfTasks > 0) {

        // Set params.
        kernel.setParams(
            taskOffset,
            numberOfTasks,
            nodeLeftIndices,
            nodeRightIndices,
            nodeIndices,
            nodeOffsets,
            nodeStates
        );

        // Launch.
        time += kernel.launchTimed(numberOfTasks);

        // Update task offset and number of tasks.
        taskOffset += numberOfTasks;
        numberOfTasks = (*(int*)module->getGlobal("interiorPrefixScanOffset").getPtr()) - taskOffset;

    }

    // Kernel time.
    return time;

}

float BVHCollapser::computeReferenceOffsets(
    int numberOfReferences
) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computeReferenceOffsets");

    // Resize buffer.
    referenceOffsets.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));

    // Clear prefix scan offset.
    module->getGlobal("prefixScanOffset").clear();

    // Set params.
    kernel.setParams(
        2 * numberOfReferences - 1,
        nodeStates,
        nodeSizes,
        referenceOffsets
    );

    // Launch.
    float time = kernel.launchTimed(2 * numberOfReferences - 1);

    // Kernel time.
    return time;

}

float BVHCollapser::compact(
    int numberOfReferences,
    Buffer & nodeParentIndices,
    Buffer & nodeLeftIndices,
    Buffer & nodeRightIndices,
    Buffer & nodeBoxesMin,
    Buffer & nodeBoxesMax,
    HipBVH & bvh
) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("compact" + bvh.getLayoutString());

    // Leaf offset.
    bvh.numberOfInteriorNodes = *(int*)module->getGlobal("interiorPrefixScanOffset").getPtr();
    bvh.numberOfLeafNodes = *(int*)module->getGlobal("leafPrefixScanOffset").getPtr();

    // Set params.
    kernel.setParams(
        2 * numberOfReferences - 1,
        bvh.numberOfInteriorNodes,
        nodeStates,
        nodeOffsets,
        nodeParentIndices,
        nodeLeftIndices,
        nodeRightIndices,
        nodeSizes,
        referenceOffsets,
        nodeBoxesMin,
        nodeBoxesMax,
        bvh.getNodes()
    );

    // Launch.
    float time = kernel.launchTimed(2 * numberOfReferences - 1);

    // Kernel time.
    return time;

}

float BVHCollapser::reorderTriangleIndices(
    int numberOfReferences,
    Buffer & nodeLeftIndices,
    Buffer & triangleIndices,
    HipBVH & bvh
) {
    
    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("reorderTriangleIndices");

    // Set params.
    kernel.setParams(
        numberOfReferences,
        nodeLeftIndices,
        referenceOffsets,
        triangleIndices,
        bvh.getTriangleIndices(),
        leafIndices
    );

    // Launch.
    float time = kernel.launchTimed(numberOfReferences);

    // Kernel time.
    return time;

}

BVH * BVHCollapser::convert(
    int numberOfReferences,
    Buffer & nodeParentIndices,
    Buffer & nodeLeftIndices,
    Buffer & nodeRightIndices,
    Buffer & nodeBoxesMin,
    Buffer & nodeBoxesMax,
    Buffer & triangleIndices
) {

    // Nodes.
    int * leftIndices = (int*)nodeLeftIndices.getPtr();
    int * rightIndices = (int*)nodeRightIndices.getPtr();
    Vec4f * boxesMin = (Vec4f*)nodeBoxesMin.getPtr();
    Vec4f * boxesMax = (Vec4f*)nodeBoxesMax.getPtr();

    // BVH.
    BVH * bvh = new BVH();
    bvh->root = new BVH::InteriorNode();
    bvh->numberOfInteriorNodes = numberOfReferences - 1;
    bvh->numberOfLeafNodes = numberOfReferences;

    // Pair of nodes.
    typedef std::pair<int, BVH::Node*> ConverterPair;

    QStack<ConverterPair> stack;
    stack.push_back(ConverterPair(0, bvh->root));
    while (!stack.empty()) {

        // Pop node pair.
        ConverterPair pair = stack.back();
        stack.pop_back();
        int nodeIndex = pair.first;
        BVH::Node * node = pair.second;

        // Bounding box.
        Vec3f mn = Vec3f(boxesMin[nodeIndex]);
        Vec3f mx = Vec3f(boxesMax[nodeIndex]);
        node->box = AABB(mn, mx);

        // Leaf
        if (nodeIndex >= bvh->getNumberOfInteriorNodes()) {
            node->begin = leftIndices[nodeIndex];
            node->end = rightIndices[nodeIndex];
        }

        // Interior.
        else {
            BVH::InteriorNode * interior = dynamic_cast<BVH::InteriorNode*>(node);
            int leftIndex = leftIndices[nodeIndex];
            if (leftIndex >= bvh->getNumberOfInteriorNodes()) interior->children[0] = new BVH::LeafNode();
            else interior->children[0] = new BVH::InteriorNode();
            stack.push_back(ConverterPair(leftIndex, interior->children[0]));
            int rightIndex = rightIndices[nodeIndex];
            if (rightIndex >= bvh->getNumberOfInteriorNodes()) interior->children[1] = new BVH::LeafNode();
            else interior->children[1] = new BVH::InteriorNode();
            stack.push_back(ConverterPair(rightIndex, interior->children[1]));
        }

    }

    // Recompute attributes.
    bvh->recomputeDepth();
    bvh->recomputeIDs();
    bvh->recomputeParents();
    bvh->recomputeBounds();

    // Triangle indices.
    int offset = 0;
    bvh->triangleIndices.resize(numberOfReferences);
    writeTriangleIndices(bvh->root, (int*)triangleIndices.getPtr(), bvh->triangleIndices.data(), offset);

    return bvh;

}

BVH* BVHCollapser::convertWide(
    int numberOfReferences,
    int n,
    Buffer& nodeParentIndices,
    Buffer& nodeChildIndices,
    Buffer& nodeChildCounts,
    Buffer& nodeBoxesMin,
    Buffer& nodeBoxesMax,
    Buffer& triangleIndices
) {

    // Nodes.
    int* childIndices = (int*)nodeChildIndices.getPtr();
    int* childCounts = (int*)nodeChildCounts.getPtr();
    Vec4f* boxesMin = (Vec4f*)nodeBoxesMin.getPtr();
    Vec4f* boxesMax = (Vec4f*)nodeBoxesMax.getPtr();

    // BVH.
    BVH* bvh = new BVH();
    bvh->root = new BVH::InteriorNode();

    // Pair of nodes.
    typedef std::pair<int, BVH::Node*> ConverterPair;

    QStack<ConverterPair> stack;
    stack.push_back(ConverterPair(0, bvh->root));
    while (!stack.empty()) {

        // Pop node pair.
        ConverterPair pair = stack.back();
        stack.pop_back();
        int nodeIndex = pair.first;
        BVH::Node* node = pair.second;

        // Bounding box.
        Vec3f mn = Vec3f(boxesMin[nodeIndex]);
        Vec3f mx = Vec3f(boxesMax[nodeIndex]);
        node->box = AABB(mn, mx);

        // Leaf
        if (nodeIndex >= numberOfReferences - 1) {
            node->begin = childIndices[n * nodeIndex];
            node->end = childIndices[n * nodeIndex + 1];
        }

        // Interior.
        else {
            BVH::InteriorNode* interior = dynamic_cast<BVH::InteriorNode*>(node);
            interior->numberOfChildren = childCounts[nodeIndex];
            for (int i = 0; i < interior->numberOfChildren; ++i) {
                int childIndex = childIndices[n * nodeIndex + i];
                if (childIndex >= numberOfReferences - 1) interior->children[i] = new BVH::LeafNode();
                else interior->children[i] = new BVH::InteriorNode();
                stack.push_back(ConverterPair(childIndex, interior->children[i]));
            }
        }

    }

    // Recompute attributes.
    bvh->recomputeDepth();
    bvh->recomputeIDs();
    bvh->recomputeParents();
    bvh->recomputeNumberOfNodes();
    bvh->recomputeBounds();

    // Triangle indices.
    int offset = 0;
    bvh->triangleIndices.resize(numberOfReferences);
    writeTriangleIndices(bvh->root, (int*)triangleIndices.getPtr(), bvh->triangleIndices.data(), offset);

    // Triangle histogram.
    //int* triangleIndicesPtr = (int*)triangleIndices.getPtr();
    int* triangleIndicesPtr = bvh->triangleIndices.data();
    int numberOfTriangles = numberOfReferences;
    QVector<int> triangleHistogram(numberOfTriangles);
    memset(triangleHistogram.data(), 0, sizeof(int) * numberOfTriangles);

    return bvh;

}

void BVHCollapser::writeTriangleIndices(BVH::Node * node, const int * triangleIndicesSrc, int * triangleIndicesDst, int & offset) {
    if (node->isLeaf()) {
        int offsetTmp = offset;
        for (int i = node->begin; i < node->end; ++i)
            triangleIndicesDst[offset++] = triangleIndicesSrc[i];
        node->begin = offsetTmp;
        node->end = offset;
    }
    else {
        BVH::InteriorNode * interior = dynamic_cast<BVH::InteriorNode*>(node);
        node->begin = offset;
        for (int i = 0; i < interior->getNumberOfChildNodes(); ++i)
            writeTriangleIndices(interior->children[i], triangleIndicesSrc, triangleIndicesDst, offset);
        node->end = offset;
    }
}

BVHCollapser::BVHCollapser() {
    compiler.setSourceFile("../src/hippie/rt/bvh/BVHCollapserKernels.cu");
}

BVHCollapser::~BVHCollapser() {
}

float BVHCollapser::collapseAdaptive(
    int numberOfReferences,
    Buffer & nodeParentIndices,
    Buffer & nodeLeftIndices,
    Buffer & nodeRightIndices,
    Buffer & nodeBoxesMin,
    Buffer & nodeBoxesMax,
    Buffer & triangleIndices,
    HipBVH & cbvh
) {

    if (cbvh.getLayout() == HipBVH::Layout::BIN) {

        // Compute sizes.
        float sizesTime = computeSizes(numberOfReferences, nodeParentIndices, nodeLeftIndices, nodeRightIndices, cbvh);
        logger(LOG_INFO) << "INFO <BVHCollapser> Node sizes computed in " << sizesTime << "s.\n";

        // Node states.
        float nodeStatesTime = computeNodeStatesAdaptive(numberOfReferences, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, cbvh);
        logger(LOG_INFO) << "INFO <BVHCollapser> Node states computed in " << nodeStatesTime << "s.\n";

        // Leaf indices.
        float leafIndicesTime = computeLeafIndices(numberOfReferences, nodeParentIndices);
        logger(LOG_INFO) << "INFO <BVHCollapser> Leaf indices computed in " << leafIndicesTime << "s.\n";

        // Invalidate collapsed leaves.
        float invalidateTime = invalidateCollapsedNodes(numberOfReferences, nodeParentIndices, cbvh);
        logger(LOG_INFO) << "INFO <BVHCollapser> Collapsed nodes invalidated in " << invalidateTime << "s.\n";

        // Node offsets.
        float nodeOffsetsTime = computeNodeOffsets(numberOfReferences, nodeLeftIndices, nodeRightIndices);
        logger(LOG_INFO) << "INFO <BVHCollapser> Node offsets computed in " << nodeOffsetsTime << "s.\n";

        // Triangle offsets.
        float referenceOffsetsTime = computeReferenceOffsets(numberOfReferences);
        logger(LOG_INFO) << "INFO <BVHCollapser> Triangle offsets computed in " << referenceOffsetsTime << "s.\n";

        // Compaction.
        float compactTime = compact(numberOfReferences, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, cbvh);
        logger(LOG_INFO) << "INFO <BVHCollapser> Nodes compacted in " << compactTime << "s.\n";

        // Reorder triangle indices.
        float reorderTrianglesTime = reorderTriangleIndices(numberOfReferences, nodeLeftIndices, triangleIndices, cbvh);
        logger(LOG_INFO) << "INFO <BVHCollapser> Triangles reordered in " << reorderTrianglesTime << "s.\n";

        return sizesTime + nodeStatesTime + leafIndicesTime + invalidateTime + nodeOffsetsTime + referenceOffsetsTime + compactTime + reorderTrianglesTime;

    }

    // QUAD or OCT.
    else {

        // Convert to CPU representation.
        BVH * bvh = convert(
            numberOfReferences,
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            nodeBoxesMin,
            nodeBoxesMax,
            triangleIndices
        );

        // Convert back to HIP representation.
        BVHConverter converter;
        converter.convertAdaptive(*bvh, cbvh);

        // Delete BVH on CPU.
        delete bvh;

        return 0.0f;

    }

}

float BVHCollapser::collapse(
    int numberOfReferences,
    int maxLeafSize,
    Buffer & nodeParentIndices,
    Buffer & nodeLeftIndices,
    Buffer & nodeRightIndices,
    Buffer & nodeBoxesMin,
    Buffer & nodeBoxesMax,
    Buffer & triangleIndices,
    HipBVH & cbvh
) {

    if (cbvh.getLayout() == HipBVH::Layout::BIN) {

        // Compute sizes.
        float sizesTime = computeSizes(numberOfReferences, nodeParentIndices, nodeLeftIndices, nodeRightIndices, cbvh);
        logger(LOG_INFO) << "INFO <BVHCollapser> Node sizes computed in " << sizesTime << "s.\n";

        // Node states.
        float nodeStatesTime = computeNodeStates(numberOfReferences, maxLeafSize, nodeParentIndices, cbvh);
        logger(LOG_INFO) << "INFO <BVHCollapser> Node states computed in " << nodeStatesTime << "s.\n";

        // Leaf indices.
        float leafIndicesTime = computeLeafIndices(numberOfReferences, nodeParentIndices);
        logger(LOG_INFO) << "INFO <BVHCollapser> Leaf indices computed in " << leafIndicesTime << "s.\n";

        // Invalidate collapsed leaves.
        float invalidateTime = invalidateCollapsedNodes(numberOfReferences, nodeParentIndices, cbvh);
        logger(LOG_INFO) << "INFO <BVHCollapser> Collapsed nodes invalidated in " << invalidateTime << "s.\n";

        // Node offsets.
        float nodeOffsetsTime = computeNodeOffsets(numberOfReferences, nodeLeftIndices, nodeRightIndices);
        logger(LOG_INFO) << "INFO <BVHCollapser> Node offsets computed in " << nodeOffsetsTime << "s.\n";

        // Triangle offsets.
        float referenceOffsetsTime = computeReferenceOffsets(numberOfReferences);
        logger(LOG_INFO) << "INFO <BVHCollapser> Triangle offsets computed in " << referenceOffsetsTime << "s.\n";

        // Compaction.
        float compactTime = compact(numberOfReferences, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, cbvh);
        logger(LOG_INFO) << "INFO <BVHCollapser> Nodes compacted in " << compactTime << "s.\n";

        // Reorder triangle indices.
        float reorderTrianglesTime = reorderTriangleIndices(numberOfReferences, nodeLeftIndices, triangleIndices, cbvh);
        logger(LOG_INFO) << "INFO <BVHCollapser> Triangles reordered in " << reorderTrianglesTime << "s.\n";

        return sizesTime + nodeStatesTime + leafIndicesTime + invalidateTime + nodeOffsetsTime + referenceOffsetsTime + compactTime + reorderTrianglesTime;

    }

    // QUAD or OCT.
    else {

        // Convert to CPU representation.
        BVH * bvh = convert(
            numberOfReferences,
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            nodeBoxesMin,
            nodeBoxesMax,
            triangleIndices
        );

        // Convert back to HIP representation.
        BVHConverter converter;
        converter.convert(*bvh, cbvh, maxLeafSize);

        // Delete BVH on CPU.
        delete bvh;

        return 0.0f;

    }

}

float BVHCollapser::collapseAdaptiveWide(
    int numberOfReferences,
    Buffer& nodeParentIndices,
    Buffer& nodeChildIndices,
    Buffer& nodeChildCounts,
    Buffer& nodeBoxesMin,
    Buffer& nodeBoxesMax,
    Buffer& triangleIndices,
    HipBVH& cbvh
) {

    // Convert to CPU representation.
    BVH* bvh = convertWide(
        numberOfReferences,
        cbvh.getLayout(),
        nodeParentIndices,
        nodeChildIndices,
        nodeChildCounts,
        nodeBoxesMin,
        nodeBoxesMax,
        triangleIndices
    );

    // Collapse.
    bvh->collapseAdaptive();

    // Convert back to HIP representation.
    BVHConverter converter;
    converter.convertAdaptive(*bvh, cbvh);
    return 0.0f;

}

float BVHCollapser::collapseWide(
    int numberOfReferences,
    int maxLeafSize,
    Buffer& nodeParentIndices,
    Buffer& nodeChildIndices,
    Buffer& nodeChildCounts,
    Buffer& nodeBoxesMin,
    Buffer& nodeBoxesMax,
    Buffer& triangleIndices,
    HipBVH& cbvh
) {

    // Convert to CPU representation.
    BVH* bvh = convertWide(
        numberOfReferences,
        cbvh.getLayout(),
        nodeParentIndices,
        nodeChildIndices,
        nodeChildCounts,
        nodeBoxesMin,
        nodeBoxesMax,
        triangleIndices
    );

    // Collapse.
    bvh->collapse(maxLeafSize);

    // Convert back to HIP representation.
    BVHConverter converter;
    converter.convert(*bvh, cbvh, maxLeafSize);
    return 0.0f;

}

void BVHCollapser::clear() {
    nodeCosts.free();
    nodeStates.free();
    nodeOffsets.free();
    nodeStates.free();
    nodeSizes.free();
    leafIndices.free();
    referenceOffsets.free();
}
