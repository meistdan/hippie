/**
 * \file	HipBVH.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	HipBVH class source file.
 */

#include "HipBVH.h"
#include "HipBVHKernels.h"
#include "environment/AppEnvironment.h"
#include "util/Logger.h"
#include <QStack>

void HipBVH::resize(int numberOfReferences) {
    if (numberOfReferences <= 0) logger(LOG_WARN) << "WARN <HipBVH> Number of references must be positive.\n";
    else {
        nodes.resizeDiscard(getNodeSize() * (2 * numberOfReferences - 1));
        termCounters.resizeDiscard(sizeof(int) * (numberOfReferences - 1));
        triangleIndices.resizeDiscard(sizeof(int) * numberOfReferences);
        woopTriangles.resizeDiscard((4 * sizeof(Vec4f) * numberOfReferences + TRIANGLE_ALIGN - 1) & -TRIANGLE_ALIGN);
    }
}

QString HipBVH::getLayoutString(void) {
    if (layout == BIN) return "Bin";
    else if (layout == QUAD) return "Quad";
    else return "Oct";
}

int HipBVH::getNodeSize() {
    if (layout == BIN) return sizeof(HipBVHNodeBin);
    else if (layout == QUAD) return sizeof(HipBVHNodeQuad);
    else return sizeof(HipBVHNodeOct);
}

float HipBVH::refitLeaves() {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("refitLeaves" + getLayoutString());

    // Set params.
    kernel.setParams(
        getNumberOfNodes(),
        getNumberOfInteriorNodes(),
        getTriangleIndices(),
        getNodes()
    );

    // Launch.
    float time = kernel.launchTimed(getNumberOfLeafNodes());

    // Kernel time.
    return time;

}

float HipBVH::refitInteriors() {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("refitInteriors" + getLayoutString());

    // Clear term counter.
    termCounters.clear();

    // Set params.
    kernel.setParams(
        getNumberOfNodes(),
        getNumberOfInteriorNodes(),
        termCounters,
        nodes
    );

    // Launch.
    float time = kernel.launchTimed(getNumberOfLeafNodes());

    // Kernel time.
    return time;

}

float HipBVH::refit() {
    float time = 0.0f;
    if (spatialSplits()) {
        logger(LOG_WARN) << "WARN <HipBVH> Spatial splits detected. Refitting of bounding boxes might destroy spatial splits.\n";
    }
    else {
        time += refitLeaves();
        time += refitInteriors();
    }
    return time;
}

float HipBVH::woopifyTriangles() {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("woopifyTriangles");

    // Set params.
    kernel.setParams(
        getNumberOfReferences(),
        getTriangleIndices(),
        scene->getTriangleBuffer(),
        scene->getVertexBuffer(),
        getWoopTriangles().getMutableHipPtr()
    );

    // Launch.
    float time = kernel.launchTimed(getNumberOfReferences());

    // Kernel time.
    return time;

}

void HipBVH::generateColors(Buffer & nodeColors) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("generateColors");

    // Set params.
    kernel.setParams(
        getNumberOfNodes(),
        nodeColors
    );

    // Launch.
    kernel.launch(getNumberOfNodes());

}

void HipBVH::colorizeTriangles(Buffer & nodeColors, int nodeSizeThreshold) {

    // Kernel.
    HipModule* module = compiler.compile();
    HipKernel kernel = module->getKernel("colorizeTriangles" + getLayoutString());

    // Set params.
    kernel.setParams(
        getNumberOfNodes(),
        getNumberOfInteriorNodes(),
        nodeSizeThreshold,
        getTriangleIndices(),
        nodeColors,
        scene->getPseudocolorBuffer(),
        getNodes()
    );

    // Launch.
    kernel.launch(getNumberOfLeafNodes());

}

template <typename HipBVHNode>
bool HipBVH::validate() {

    bool valid = true;

    // Nodes.
    HipBVHNode * nodes = (HipBVHNode*)getNodes().getPtr();

    // Reference indices.
    int * triangleIndices = (int*)getTriangleIndices().getPtr();

    // Triangles.
    Vec3f * vertices = (Vec3f*)scene->getVertexBuffer().getPtr();
    Vec3i * triangles = (Vec3i*)scene->getTriangleBuffer().getPtr();

    // Nodes histogram.
    int numberOfNodes = getNumberOfNodes();
    QVector<int> nodeHistogram(numberOfNodes);
    memset(nodeHistogram.data(), 0, sizeof(int) * numberOfNodes);
    nodeHistogram[0]++;

    // Triangle histogram.
    int numberOfTriangles = scene->getNumberOfTriangles();
    int numberOfReferences = getNumberOfReferences();
    QVector<int> triangleHistogram(numberOfTriangles);
    memset(triangleHistogram.data(), 0, sizeof(int) * numberOfTriangles);

    // Check triangle indices.
    for (int i = 0; i < numberOfReferences; ++i) {
        if (triangleIndices[i] < 0 || triangleIndices[i] >= numberOfTriangles)
            logger(LOG_WARN) << "WARN <HipBVH> Invalid triangle indices!\n";
        triangleHistogram[triangleIndices[i]]++;
    }

    for (int i = 0; i < numberOfTriangles; ++i) {
        if (triangleHistogram[i] < 1) {
            logger(LOG_WARN) << "WARN <HipBVH> Invalid triangle indices!\n";
            valid = false;
        }
    }

    // Reset triangle histogram.
    memset(triangleHistogram.data(), 0, sizeof(int) * numberOfTriangles);

    // Stack.
    QStack<int> stack;
    stack.push_back(0);

    // Traverse BVH.
    while (!stack.empty()) {

        // Pop.
        int nodeIndex = stack.back();
        stack.pop_back();
        HipBVHNode & node = nodes[nodeIndex];

        // Interior.
        if (!node.isLeaf()) {

            // Node size.
            int size = 0;

            for (int i = 0; i < node.getNumberOfChildren(); ++i) {

                // Child index.
                int childIndex = node.getChildIndex(i);

                // Child node.
                HipBVHNode & child = nodes[childIndex];

                // Parent index.
                if (child.getParentIndex() != nodeIndex) {
                    logger(LOG_WARN) << "WARN <HipBVH> Invalid parent index!\n";
                    valid = false;
                }

                // Add child size.
                size += child.getSize();

                // Update histogram.
                nodeHistogram[childIndex]++;

                // Push.
                stack.push_back(childIndex);

            }

            // Check sizes.
            if (node.getSize() != size) {
                logger(LOG_WARN) << "WARN <HipBVH> Invalid node size!\n";
                valid = false;
            }

        }

        // Leaf.
        else {

            // Check Bounds.
            if (node.getBegin() >= node.getEnd()) {
                logger(LOG_WARN) << "WARN <HipBVH> Invalid leaf bounds [" << node.getBegin() << "," << node.getEnd() << "]!\n";
                valid = false;
            }

            // Box.
            AABB box = node.getBoundingBox();

            // Update histogram.
            for (int i = node.getBegin(); i < node.getEnd(); ++i) {
                int triangleIndex = triangleIndices[i];
                triangleHistogram[triangleIndex]++;
                Vec3i triangle = triangles[triangleIndex];
                AABB triangleBox;
                triangleBox.grow(vertices[triangle.x]);
                triangleBox.grow(vertices[triangle.y]);
                triangleBox.grow(vertices[triangle.z]);
                if (!overlap(box, triangleBox)) {
                    logger(LOG_WARN) << "WARN <HipBVH> Triangle is not in a leaf bounding box!\n";
                    logger(LOG_WARN) << "WARN <HipBVH> Triangle box min = [" <<
                        triangleBox.mn.x << ", " << triangleBox.mn.y << ", " << triangleBox.mn.z << "], max = [" <<
                        triangleBox.mx.x << ", " << triangleBox.mx.y << ", " << triangleBox.mx.z << "].\n";
                    logger(LOG_WARN) << "WARN <HipBVH> Node box min = [" <<
                        box.mn.x << ", " << box.mn.y << ", " << box.mn.z << "], max = [" <<
                        box.mx.x << ", " << box.mx.y << ", " << box.mx.z << "].\n";
                    valid = false;
                }
            }

        }

    }

    // Check node histogram.
    for (int i = 0; i < numberOfNodes; ++i) {
        if (nodeHistogram[i] != 1) {
            logger(LOG_WARN) << "WARN <HipBVH> Not all nodes are referenced!\n";
            valid = false;
        }
    }

    // Check triangle histogram.
    for (int i = 0; i < numberOfTriangles; ++i) {
        if (triangleHistogram[i] == 0) {
            logger(LOG_WARN) << "WARN <HipBVH> Not all triangles are referenced!\n";
            valid = false;
        }
    }

    return valid;

}

HipBVH::HipBVH(Scene * scene) :
    scene(scene),
    ct(CT),
    ci(CI),
    numberOfLeafNodes(0),
    numberOfInteriorNodes(0),
    lastNodeSizeThreshold(-1),
    layout(BIN)
{

    // Compile.
    compiler.setSourceFile("../src/hippie/rt/bvh/HipBVHKernels.cu");
    compiler.compile();

    // Layout.
    QString layoutStr;
    Environment::getInstance()->getStringValue("Bvh.layout", layoutStr);
    if (layoutStr == "bin") layout = BIN;
    else if (layoutStr == "quad") layout = QUAD;
    else layout = OCT;

    // Cost constants.
    float _ct, _ci;
    Environment::getInstance()->getFloatValue("Bvh.ct", _ct);
    Environment::getInstance()->getFloatValue("Bvh.ci", _ci);
    if (_ct > 0.0f) ct = _ct;
    if (_ci > 0.0f) ci = _ci;

}

HipBVH::~HipBVH() {
}

HipBVH::Layout HipBVH::getLayout() {
    return layout;
}

float HipBVH::getCost() {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computeCost" + getLayoutString());

    // Reset cost.
    *(float*)module->getGlobal("cost").getMutablePtr() = 0.0f;

    // Set params.
    kernel.setParams(
        getNumberOfNodes(),
        scene->getSceneBox().area(),
        getCt(),
        getCi(),
        getNodes()
    );

    // Launch.
    kernel.launch(getNumberOfNodes(), Vec2i(REDUCTION_BLOCK_THREADS, 1));

    // Cost
    return *(float*)module->getGlobal("cost").getPtr();

}

float HipBVH::getCt() {
    return ct;
}

float HipBVH::getCi() {
    return ci;
}

float HipBVH::getAvgLeafSize() {

    // Check spatial splits.
    if (spatialSplits())
        logger(LOG_WARN) << "WARN <HipBVH> Spatial splits detected. Average leaf size might no be correct.\n";

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computeSumOfLeafSizes" + getLayoutString());

    // Reset size.
    *(int*)module->getGlobal("sumOfLeafSizes").getMutablePtr() = 0;

    // Set params.
    kernel.setParams(
        getNumberOfNodes(),
        getNumberOfInteriorNodes(),
        getNodes()
    );

    // Launch.
    kernel.launch(getNumberOfLeafNodes(), Vec2i(REDUCTION_BLOCK_THREADS, 1));

    // Avg. leaf size.
    return *(int*)module->getGlobal("sumOfLeafSizes").getPtr() / float(getNumberOfLeafNodes());

}

const QVector<float> & HipBVH::getLeafSizeHistogram() {

    // Check spatial splits.
    if (spatialSplits())
        logger(LOG_WARN) << "WARN <HipBVH> Spatial splits detected. Leaf histogram might no be correct.\n";

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computeLeafSizeHistogram" + getLayoutString());

    // Reset histogram.
    int * histogram = (int*)module->getGlobal("leafSizeHistogram").getMutablePtr();
    for (int i = 0; i <= MAX_LEAF_SIZE; ++i)
        histogram[i] = 0;

    // Set params.
    kernel.setParams(
        getNumberOfNodes(),
        getNumberOfInteriorNodes(),
        getNodes()
    );

    // Launch.
    kernel.launch(getNumberOfLeafNodes());

    // Update histogram.
    leafSizeHistogram.clear();
    histogram = (int*)module->getGlobal("leafSizeHistogram").getPtr();
    for (int i = 0; i <= MAX_LEAF_SIZE; ++i)
        leafSizeHistogram.push_back(histogram[i] / float(getNumberOfLeafNodes()));

    // Return histogram.
    return leafSizeHistogram;

}

int HipBVH::getNumberOfNodes() {
    return getNumberOfLeafNodes() + getNumberOfInteriorNodes();
}

int HipBVH::getNumberOfLeafNodes() {
    return numberOfLeafNodes;
}

int HipBVH::getNumberOfInteriorNodes() {
    return numberOfInteriorNodes;
}

int HipBVH::getNumberOfReferences() {
    return getTriangleIndices().getSize() / int(sizeof(int));
}

Buffer & HipBVH::getNodes() {
    return nodes;
}

Buffer & HipBVH::getWoopTriangles() {
    return woopTriangles;
}

Buffer & HipBVH::getTriangleIndices() {
    return triangleIndices;
}

float HipBVH::update() {
    float time = 0.0f;
    time += woopifyTriangles();
    time += refit();
    return time;
}

void HipBVH::colorizeScene(int nodeSizeThreshold) {
    nodeSizeThreshold = qMin(nodeSizeThreshold, getNumberOfReferences());
    if (lastNodeSizeThreshold != nodeSizeThreshold) {
        if (spatialSplits())
            logger(LOG_WARN) << "WARN <HipBVH> Spatial splits detected. Colorization might not be correct.\n";
        Buffer nodeColors;
        nodeColors.resizeDiscard(sizeof(unsigned int) * getNumberOfNodes());
        generateColors(nodeColors);
        colorizeTriangles(nodeColors, nodeSizeThreshold);
        lastNodeSizeThreshold = nodeSizeThreshold;
    }
}

bool HipBVH::validate() {
    if (layout == BIN) return validate<HipBVHNodeBin>();
    else if (layout == QUAD) return validate<HipBVHNodeQuad>();
    else return validate<HipBVHNodeOct>();
}

bool HipBVH::spatialSplits() {
    Q_ASSERT(scene != nullptr);
    Q_ASSERT(scene->getNumberOfTriangles() <= getNumberOfReferences());
    return scene->getNumberOfTriangles() != getNumberOfReferences();
}
