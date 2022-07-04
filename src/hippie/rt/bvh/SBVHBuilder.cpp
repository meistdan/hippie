/**
* \file	  SBVHBuilder.h
* \author Daniel Meister
* \date	  2019/04/17
* \brief  SBVHBuilder class source file.
*/

#include <QElapsedTimer>
#include <QQueue>
#include "environment/AppEnvironment.h"
#include "BVHConverter.h"
#include "SBVHBuilder.h"

SBVHBuilder::ObjectSplit SBVHBuilder::findObjectSplit(const NodeSpec & spec, float nodeSAH) {

    ObjectSplit split;
    const Reference * refPtr = references.data() + (references.size() - spec.size);
    float bestTieBreak = MAX_FLOAT;

    // Sort along each axisension.
    for (int axis = 0; axis < 3; axis++) {

        qSort(references.end() - spec.size, references.end(), Comparator(axis));

        // Sweep right to left and determine bounds.
        AABB rightBox;
        for (int i = spec.size - 1; i > 0; i--) {
            rightBox.grow(refPtr[i].box);
            rightBoxes[i - 1] = rightBox;
        }

        // Sweep left to right and select lowest SAH.
        AABB leftBox;
        for (int i = 1; i < spec.size; i++) {
            leftBox.grow(refPtr[i - 1].box);
            float sah = nodeSAH + leftBox.area() * bvh->getCi() * i + rightBoxes[i - 1].area() * bvh->getCi() * (spec.size - i);
            float tieBreak = i * i + (spec.size - i) * (spec.size - i);
            if (sah < split.sah || (sah == split.sah && tieBreak < bestTieBreak)) {
                split.sah = sah;
                split.axis = axis;
                split.leftCount = i;
                split.leftBox = leftBox;
                split.rightBox = rightBoxes[i - 1];
                bestTieBreak = tieBreak;
            }
        }
    }

    return split;
}

void SBVHBuilder::performObjectSplit(NodeSpec & left, NodeSpec & right, const NodeSpec & spec, const ObjectSplit & split) {
    qSort(references.end() - spec.size, references.end(), Comparator(split.axis));
    left.size = split.leftCount;
    left.box = split.leftBox;
    right.size = spec.size - split.leftCount;
    right.box = split.rightBox;
}

SBVHBuilder::SpatialSplit SBVHBuilder::findSpatialSplit(const NodeSpec & spec, float nodeSAH) {

    // Initialize bins.
    Vec3f origin = spec.box.mn;
    Vec3f binSize = (spec.box.mx - origin) * (1.0f / float(NUMBER_OF_BINS));
    Vec3f invBinSize = 1.0f / binSize;
    for (int axis = 0; axis < 3; axis++) {
        for (int i = 0; i < NUMBER_OF_BINS; i++) {
            Bin & bin = bins[axis][i];
            bin.box = AABB();
            bin.enter = 0;
            bin.exit = 0;
        }
    }

    // Chop references into bins.
    for (int refIdx = references.size() - spec.size; refIdx < references.size(); refIdx++) {
        const Reference & ref = references[refIdx];
        Vec3i firstBin = clamp(Vec3i((ref.box.mn - origin) * invBinSize), Vec3i(0), Vec3i(NUMBER_OF_BINS - 1));
        Vec3i lastBin = clamp(Vec3i((ref.box.mx - origin) * invBinSize), firstBin, Vec3i(NUMBER_OF_BINS - 1));

        for (int axis = 0; axis < 3; axis++) {
            Reference currRef = ref;
            for (int i = firstBin[axis]; i < lastBin[axis]; i++) {
                Reference leftRef, rightRef;
                splitReference(leftRef, rightRef, currRef, axis, origin[axis] + binSize[axis] * (float)(i + 1));
                bins[axis][i].box.grow(leftRef.box);
                currRef = rightRef;
            }
            bins[axis][lastBin[axis]].box.grow(currRef.box);
            bins[axis][firstBin[axis]].enter++;
            bins[axis][lastBin[axis]].exit++;
        }
    }

    // Select best split plane.
    SpatialSplit split;
    for (int axis = 0; axis < 3; axis++) {

        // Sweep right to left and determine bounds.
        AABB rightBounds;
        for (int i = NUMBER_OF_BINS - 1; i > 0; i--) {
            rightBounds.grow(bins[axis][i].box);
            rightBoxes[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.
        AABB leftBounds;
        int leftNum = 0;
        int rightNum = spec.size;

        for (int i = 1; i < NUMBER_OF_BINS; i++) {
            leftBounds.grow(bins[axis][i - 1].box);
            leftNum += bins[axis][i - 1].enter;
            rightNum -= bins[axis][i - 1].exit;

            float sah = nodeSAH + leftBounds.area() * bvh->getCi() * (leftNum) + rightBoxes[i - 1].area() * bvh->getCi() * (rightNum);
            if (sah < split.sah) {
                split.sah = sah;
                split.axis = axis;
                split.position = origin[axis] + binSize[axis] * float(i);
            }
        }
    }

    return split;
}

void SBVHBuilder::performSpatialSplit(NodeSpec & left, NodeSpec & right, const NodeSpec & spec, const SpatialSplit & split) {

    // Categorize references and compute bounds.
    // Left-hand side:      [leftStart, leftEnd[
    // Uncategorized/split: [leftEnd, rightStart[
    // Right-hand side:     [rightStart, refs.getSize()[

    QVector<Reference> & refs = references;
    int leftStart = refs.size() - spec.size;
    int leftEnd = leftStart;
    int rightStart = refs.size();
    left.box = right.box = AABB();

    for (int i = leftEnd; i < rightStart; i++) {
        // Entirely on the left-hand side?
        if (refs[i].box.mx[split.axis] <= split.position) {
            left.box.grow(refs[i].box);
            qSwap(refs[i], refs[leftEnd++]);
        }

        // Entirely on the right-hand side?
        else if (refs[i].box.mn[split.axis] >= split.position) {
            right.box.grow(refs[i].box);
            qSwap(refs[i--], refs[--rightStart]);
        }
    }

    // Duplicate or unsplit references intersecting both sides.
    while (leftEnd < rightStart) {

        // Split reference.
        Reference lref, rref;
        splitReference(lref, rref, refs[leftEnd], split.axis, split.position);

        // Compute SAH for duplicate/unsplit candidates.
        AABB lub = left.box;  // Unsplit to left:     new left-hand bounds.
        AABB rub = right.box; // Unsplit to right:    new right-hand bounds.
        AABB ldb = left.box;  // Duplicate:           new left-hand bounds.
        AABB rdb = right.box; // Duplicate:           new right-hand bounds.
        lub.grow(refs[leftEnd].box);
        rub.grow(refs[leftEnd].box);
        ldb.grow(lref.box);
        rdb.grow(rref.box);

        float lac = bvh->getCi() * (leftEnd - leftStart);
        float rac = bvh->getCi() * (refs.size() - rightStart);
        float lbc = bvh->getCi() * (leftEnd - leftStart + 1);
        float rbc = bvh->getCi() * (refs.size() - rightStart + 1);

        float unsplitLeftSAH = lub.area() * lbc + right.box.area() * rac;
        float unsplitRightSAH = left.box.area() * lac + rub.area() * rbc;
        float duplicateSAH = ldb.area() * lbc + rdb.area() * rbc;
        float minSAH = qMin(qMin(unsplitLeftSAH, unsplitRightSAH), duplicateSAH);

        // Unsplit to left?
        if (minSAH == unsplitLeftSAH) {
            left.box = lub;
            leftEnd++;
        }

        // Unsplit to right?
        else if (minSAH == unsplitRightSAH) {
            right.box = rub;
            qSwap(refs[leftEnd], refs[--rightStart]);
        }

        // Duplicate?
        else {
            left.box = ldb;
            right.box = rdb;
            refs[leftEnd++] = lref;
            refs.push_back(rref);
        }
    }

    left.size = leftEnd - leftStart;
    right.size = refs.size() - rightStart;

}

void SBVHBuilder::splitReference(Reference & left, Reference & right, Reference & ref, int axis, float pos) {

    // Initialize references.
    left.index = right.index = ref.index;
    left.box = right.box = AABB();

    // Loop over vertices/edges.
    const Vec3i * tris = (const Vec3i*)scene->getTriangleBuffer().getPtr();
    const Vec3f * verts = (const Vec3f*)scene->getVertexBuffer().getPtr();
    const Vec3i & inds = tris[ref.index];
    const Vec3f * v1 = &verts[inds.z];

    for (int i = 0; i < 3; i++) {
        const Vec3f * v0 = v1;
        v1 = &verts[inds[i]];
        float v0p = (*v0)[axis];
        float v1p = (*v1)[axis];

        // Insert vertex to the boxes it belongs to.
        if (v0p <= pos)
            left.box.grow(*v0);
        if (v0p >= pos)
            right.box.grow(*v0);

        // Edge intersects the plane => insert intersection to both boxes.
        if ((v0p < pos && v1p > pos)|| (v0p > pos && v1p < pos)) {
            Vec3f t = mix(*v0, *v1, clamp((pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
            left.box.grow(t);
            right.box.grow(t);
        }
    }
    
    // Intersect with original bounds.
    left.box.mx[axis] = pos;
    right.box.mn[axis] = pos;
    left.box.intersect(ref.box);
    right.box.intersect(ref.box);

}

BVH::LeafNode * SBVHBuilder::buildLeaf(const NodeSpec & spec) {
    for (int i = 0; i < spec.size; i++) {
        triangleIndices.push_back(references.back().index);
        references.pop_back();
    }
    BVH::LeafNode * leaf = new BVH::LeafNode();
    leaf->box = spec.box;
    leaf->begin = triangleIndices.size() - spec.size;
    leaf->end = triangleIndices.size();
    return leaf;
}

BVH::Node * SBVHBuilder::buildNode(NodeSpec spec, int depth, float progressStart, float progressEnd) {

#if ENABLE_PRINT_PROGRESS
    int currentProgress = 100.0f * progressStart;
    if (progress != currentProgress) {
        progress = currentProgress;
        qDebug() << "INFO <SBVHBuilder> Progress" << progress << "%, duplicates" << 100.0f * duplicates / scene->getNumberOfTriangles() << ".";
    }
#endif

    // Remove degenerates.
    {
        int firstRef = references.size() - spec.size;
        for (int i = references.size() - 1; i >= firstRef; i--) {
            Vec3f size = references[i].box.mx - references[i].box.mn;
            if (qMin(qMin(size.x, size.y), size.z) < 0.0f || (size.x + size.y + size.z) == qMax(qMax(size.x, size.y), size.z)) {
                references[i] = references.back();
                references.pop_back();
            }
        }
        spec.size = references.size() - firstRef;
    }

    // Small enough or too deep => create leaf.
    if (spec.size == 1)
    //if (spec.size <= 8 || depth >= 64)
        return buildLeaf(spec);

    // Find split candidates.
    float area = spec.box.area();
    float leafSAH = area * bvh->getCi() * (spec.size);
    float nodeSAH = area * bvh->getCt();
    ObjectSplit object = findObjectSplit(spec, nodeSAH);

    SpatialSplit spatial;
#if ENABLE_SPATIAL_SPLITS
    if (depth < MAX_DEPTH) {
        AABB overlap = object.leftBox;
        overlap.intersect(object.rightBox);
        if (overlap.area() >= scene->getSceneBox().area() * alpha)
            spatial = findSpatialSplit(spec, nodeSAH);
    }
#endif

    // Leaf SAH is the lowest => create leaf.
    float minSAH = qMin(qMin(leafSAH, object.sah), spatial.sah);
    if (minSAH == leafSAH && spec.size <= maxLeafSize)
        return buildLeaf(spec);

    // Perform split.
    NodeSpec left, right;
    if (minSAH == spatial.sah)
        performSpatialSplit(left, right, spec, spatial);
    if (!left.size || !right.size)
        performObjectSplit(left, right, spec, object);

    // Create inner node.
    duplicates += left.size + right.size - spec.size;
    float progressMid = mix(progressStart, progressEnd, float(right.size) / float(left.size + right.size));
    BVH::Node * rightNode = buildNode(right, depth + 1, progressStart, progressMid);
    BVH::Node * leftNode = buildNode(left, depth + 1, progressMid, progressEnd);
    BVH::InteriorNode * node = new BVH::InteriorNode();
    node->box = spec.box;
    node->children[0] = leftNode;
    node->children[1] = rightNode;
    return node;

}

void SBVHBuilder::reorderTriangleIndices(BVH::Node * node) {
    if (node->isLeaf()) {
        int begin = node->begin;
        int end = node->end;
        node->begin = bvh->triangleIndices.size();
        for (int i = begin; i < end; ++i)
            bvh->triangleIndices.push_back(triangleIndices[i]);
        node->end = bvh->triangleIndices.size();
    }
    else {
        BVH::InteriorNode * interior = dynamic_cast<BVH::InteriorNode*>(node);
        node->begin = triangleIndices.size();
        for (int i = 0; i < 2; ++i)
            reorderTriangleIndices(interior->children[i]);
        node->end = triangleIndices.size();
    }
}

SBVHBuilder::SBVHBuilder() : alpha(1.0e-5f) {
    float _alpha;
    Environment::getInstance()->getFloatValue("Bvh.sbvhAlpha", _alpha);
    setAlpha(_alpha);
}

SBVHBuilder::~SBVHBuilder() {
}

BVH * SBVHBuilder::buildSBVH(Scene * scene) {

    // Initialize reference stack and determine root bounds.
    Vec3i * tris = (Vec3i*)scene->getTriangleBuffer().getPtr();
    Vec3f * verts = (Vec3f*)scene->getVertexBuffer().getPtr();

    NodeSpec rootSpec;
    rootSpec.size = scene->getNumberOfTriangles();
    references.resize(rootSpec.size);

    for (int i = 0; i < rootSpec.size; i++) {
        references[i].index = i;
        for (int j = 0; j < 3; j++)
            references[i].box.grow(verts[tris[i][j]]);
        rootSpec.box.grow(references[i].box);
    }
    rightBoxes.resize(qMax(rootSpec.size, int(NUMBER_OF_BINS)) - 1);

    // Build recursively.
    this->scene = scene;
    progress = 0;
    duplicates = 0;
    bvh = new BVH();
    bvh->root = buildNode(rootSpec, 0, 0.0f, 1.0f);

    // Reorder triangle indices.
    reorderTriangleIndices(bvh->root);

    // Compute other attributes.
    bvh->recomputeBounds();
    bvh->recomputeIDs();
    bvh->recomputeParents();

    bvh->recomputeDepth();
    bvh->recomputeNumberOfNodes();

    // Done.
#if ENABLE_PRINT_PROGRESS
    qDebug() << "INFO <SBVHBuilder> Progress" << 100.0f << "%, duplicates" << 100.0f * duplicates / scene->getNumberOfTriangles() << ".";
#endif
    
    return bvh;

}

HipBVH * SBVHBuilder::build(Scene * scene) {
    float time = 0.0f;
    return build(scene, time);
}

HipBVH * SBVHBuilder::build(Scene * scene, float & time) {
    HipBVH * bvh = new HipBVH(scene);
    time = rebuild(*bvh);
    return bvh;
}

float SBVHBuilder::rebuild(HipBVH & cbvh) {
    QElapsedTimer timer;
    timer.start();
    Scene * scene = cbvh.scene;
    BVHConverter converter;
    BVH * bvh = buildSBVH(scene);
    if (adaptiveLeafSize) converter.convertAdaptive(*bvh, cbvh);
    else converter.convert(*bvh, cbvh, maxLeafSize);
    delete bvh;
    float time = timer.elapsed();
    logger(LOG_INFO) << "INFO <SBVHBuilder> Building BVH from " << scene->getNumberOfTriangles() << " triangles...\n";
    logger(LOG_INFO) << "INFO <SBVHBuilder> " << (scene->getNumberOfTriangles() * 1.0e-3f / time) << " MTriangles/s\n";
    logger(LOG_INFO) << "INFO <SBVHBuilder> BVH built in " << time << " seconds.\n";
    //cbvh.validate();
    return 0.0f;
}

float SBVHBuilder::getAlpha() {
    return alpha;
}

void SBVHBuilder::setAlpha(float alpha) {
    if (alpha >= 0.0f) this->alpha = alpha;
    else logger(LOG_WARN) << "WARN <SBVHBuilder> Alpha must be non-negative.\n";
}

void SBVHBuilder::clear() {
    triangleIndices.clear();
    references.clear();
    rightBoxes.clear();
}
