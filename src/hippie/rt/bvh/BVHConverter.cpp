/**
 * \file	BVHConvernter.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	BVHConvernter class source file.
 */

#include "BVHConverter.h"
#include "HipBVHKernels.h"
#include "util/Logger.h"
#include <QStack>

template <typename HipBVHNode>
void BVHConverter::convert(HipBVH & cbvh, BVH & bvh) {

    struct QueueEntry {
        const BVH::Node * node;
        int index;
        int parent;
        QueueEntry(const BVH::Node * node = nullptr, int index = 0, int parent = -1) : node(node), index(index), parent(parent) {}
        int encodeIdx(void) const { return (node->isLeaf()) ? ~index : index; }
    };

    const BVH::Node * root = bvh.getRoot();
    cbvh.numberOfInteriorNodes = bvh.getNumberOfInteriorNodes();
    cbvh.numberOfLeafNodes = bvh.getNumberOfLeafNodes();
    cbvh.nodes.resizeDiscard(cbvh.getNumberOfNodes() * sizeof(HipBVHNode));

    int nextInteriorIdx = 0;
    int nextLeafIdx = bvh.getNumberOfInteriorNodes();
    QQueue<QueueEntry> queue;
    queue.push_back(QueueEntry(root, nextInteriorIdx++));

    while (!queue.empty()) {

        QueueEntry entry = queue.head();
        queue.pop_front();

        int size = entry.node->end - entry.node->begin;
        int ccnt = 0;

        HipBVHNode node;

        // Leaf.
        if (entry.node->isLeaf()) {
            const BVH::LeafNode * leaf = reinterpret_cast<const BVH::LeafNode*>(entry.node);
            size = ~size;
            node.setBoundingBox(leaf->box);
            node.setBegin(leaf->begin);
            node.setEnd(leaf->end);
        }

        // Interior.
        else {
            
            ccnt = entry.node->getNumberOfChildNodes();

            // Process children.
            for (int i = 0; i < ccnt; i++) {
                const BVH::Node * c = entry.node->getChildNode(i);
                if (c->isLeaf()) queue.push_back(QueueEntry(c, nextLeafIdx++, entry.index));
                else queue.push_back(QueueEntry(c, nextInteriorIdx++, entry.index));
                QueueEntry e = queue.last();
                node.setChildBoundingBox(i, e.node->box);
                node.setChildIndex(i, e.encodeIdx());
            }

            // Dummy boxes.
            AABB dummyBox = AABB(node.getChildBoundingBox(0).mn, node.getChildBoundingBox(0).mn);
            int dummyIndex = node.getChildIndex(0);
            for (int i = ccnt; i < HipBVHNode::N; ++i) {
                node.setChildBoundingBox(i, dummyBox);
                node.setChildIndex(i, dummyIndex);
            }

        }

        node.setNumberOfChildren(ccnt);
        node.setSize(size);
        node.setParentIndex(entry.parent);

        memcpy(cbvh.nodes.getMutablePtr(entry.index * sizeof(HipBVHNode)), &node, sizeof(HipBVHNode));

    }

    Q_ASSERT(nextInteriorIdx == cbvh.getNumberOfInteriorNodes());
    Q_ASSERT(nextLeafIdx == cbvh.getNumberOfNodes());

}

void BVHConverter::woopifyTriangle(BVH & bvh, Scene * scene, int index) {
    Vec3i * vertIndices = (Vec3i*)scene->getTriangleBuffer().getPtr();
    Vec3f * vertices = (Vec3f*)scene->getVertexBuffer().getPtr();
    Vec3i & triangle = vertIndices[bvh.getTriangleIndices()[index]];
    Vec3f & v0 = vertices[triangle.x];
    Vec3f & v1 = vertices[triangle.y];
    Vec3f & v2 = vertices[triangle.z];
    Mat4f matrix;
    matrix[0] = Vec4f(v0 - v2, 0.0f);
    matrix[1] = Vec4f(v1 - v2, 0.0f);
    matrix[2] = Vec4f(cross(v0 - v2, v1 - v2), 0.0f);
    matrix[3] = Vec4f(v2.x, v2.y, v2.z, 1.0f);
    matrix = inverse(matrix);
    woopifiedTriangle[0] = Vec4f(matrix[0][2], matrix[1][2], matrix[2][2], -matrix[3][2]);
    woopifiedTriangle[1] = Vec4f(matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]);
    woopifiedTriangle[2] = Vec4f(matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]);
}

void BVHConverter::woopifyTriangles(HipBVH & cbvh, BVH & bvh) {
    Scene * scene = cbvh.scene;
    cbvh.getWoopTriangles().resizeDiscard((bvh.getTriangleIndices().size() * 64 + TRIANGLE_ALIGN - 1) & -TRIANGLE_ALIGN);
    for (int i = 0; i < bvh.getTriangleIndices().size(); ++i) {
        woopifyTriangle(bvh, scene, i);
        memcpy(cbvh.getWoopTriangles().getMutablePtr() + i * 3 * sizeof(Vec4f), &woopifiedTriangle[0], 3 * sizeof(Vec4f));
    }
    cbvh.getTriangleIndices().resizeDiscard(bvh.getTriangleIndices().size() * sizeof(int));
    memcpy(cbvh.getTriangleIndices().getMutablePtr(), bvh.getTriangleIndices().data(), bvh.getTriangleIndices().size() * sizeof(int));
}

void BVHConverter::writeTriangleIndices(BVH::Node * node, const int * triangleIndicesSrc, int * triangleIndicesDst, int & offset) {
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
        for (int i = 0; i < 2; ++i)
            writeTriangleIndices(interior->children[i], triangleIndicesSrc, triangleIndicesDst, offset);
        node->end = offset;
    }
}

BVHConverter::BVHConverter() {
}

BVHConverter::~BVHConverter() {
}

void BVHConverter::convertAdaptive(BVH & bvh, HipBVH & cbvh) {
    if (cbvh.getLayout() == HipBVH::Layout::BIN) {
        bvh.collapseAdaptive();
        convert<HipBVHNodeBin>(cbvh, bvh);
    }
    else if (cbvh.getLayout() == HipBVH::Layout::QUAD) {
        bvh.collapseAdaptiveWide(HipBVH::Layout::QUAD);
        convert<HipBVHNodeQuad>(cbvh, bvh);
    }
    else {
        bvh.collapseAdaptiveWide(HipBVH::Layout::OCT);
        convert<HipBVHNodeOct>(cbvh, bvh);
    }
    woopifyTriangles(cbvh, bvh);
}

void BVHConverter::convert(BVH & bvh, HipBVH & cbvh, int maxLeafSize) {
    if (cbvh.getLayout() == HipBVH::Layout::BIN) {
        bvh.collapse(maxLeafSize);
        convert<HipBVHNodeBin>(cbvh, bvh);
    }
    else if (cbvh.getLayout() == HipBVH::Layout::QUAD) {
        bvh.collapseWide(HipBVH::Layout::QUAD, maxLeafSize);
        convert<HipBVHNodeQuad>(cbvh, bvh);
    }
    else {
        bvh.collapseWide(HipBVH::Layout::OCT, maxLeafSize);
        convert<HipBVHNodeOct>(cbvh, bvh);
    }
    woopifyTriangles(cbvh, bvh);
}
