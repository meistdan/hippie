/**
 * \file	LBVHBuilder.cpp
 * \author	Daniel Meister
 * \date	2015/11/27
 * \brief	LBVHBuilder class source file.
 */

#include "LBVHBuilder.h"
#include "LBVHBuilderKernels.h"
#include "Presplitter.h"
#include "radix_sort/RadixSort.h"

void LBVHBuilder::allocate(int numberOfReferences) {
    mortonCodes[0].resizeDiscard(sizeof(unsigned long long) * numberOfReferences);
    mortonCodes[1].resizeDiscard(sizeof(unsigned long long) * numberOfReferences);
    nodeLeftIndices.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
    nodeRightIndices.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
    nodeBoxesMin.resizeDiscard(sizeof(Vec4f) * (2 * numberOfReferences - 1));
    nodeBoxesMax.resizeDiscard(sizeof(Vec4f) * (2 * numberOfReferences - 1));
    nodeParentIndices.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
    triangleIndices.resizeDiscard(sizeof(int) * numberOfReferences);
    referenceIndices[0].resizeDiscard(sizeof(int) * numberOfReferences);
    referenceIndices[1].resizeDiscard(sizeof(int) * numberOfReferences);
}

float LBVHBuilder::setupReferences(HipBVH & bvh, Scene * scene, int & numberOfReferences) {

    // Presplitting.
    if (presplitting) {

        // Presplitter (SBVH or Karras).
        float time = Presplitter().presplit(scene, referenceBoxesMin, referenceBoxesMax, bvh.getTriangleIndices(), numberOfReferences);

        // Resize.
        bvh.resize(numberOfReferences);

        // Time.
        return time;

    }

    // Triangles as they are.
    else {

        // Number of references.
        numberOfReferences = scene->getNumberOfTriangles();

        // Resize BVH.
        bvh.resize(numberOfReferences);

        // Allocate reference boxes.
        referenceBoxesMin.resizeDiscard(sizeof(Vec4f) * numberOfReferences);
        referenceBoxesMax.resizeDiscard(sizeof(Vec4f) * numberOfReferences);

        // Kernel.
        HipModule * module = compiler.compile();
        HipKernel kernel = module->getKernel("setupBoxes");

        // Set params.
        kernel.setParams(
            numberOfReferences,
            bvh.getTriangleIndices(),
            scene->getTriangleBuffer(),
            scene->getVertexBuffer(),
            referenceBoxesMin,
            referenceBoxesMax
        );

        // Launch.
        return kernel.launchTimed(numberOfReferences);

    }

}

float LBVHBuilder::computeMortonCodes(Scene * scene, int numberOfReferences) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computeMortonCodes60");

    // Scene box.
    *(AABB*)module->getGlobal("sceneBox").getMutablePtr() = scene->getSceneBox();
    //*(AABB*)module->getGlobal("sceneBox").getMutablePtr() = scene->getCentroidBox();

    // Set params.
    kernel.setParams(
        numberOfReferences,
        mortonCodeBits,
        mortonCodes[0],
        referenceIndices[0],
        referenceBoxesMin,
        referenceBoxesMax
    );

    // Launch.
    return kernel.launchTimed(numberOfReferences);

}

float LBVHBuilder::sortReferences(int numberOfReferences) {
    bool sortSwap = false;
    float time = RadixSort().sort(mortonCodes[0], mortonCodes[1], referenceIndices[0], referenceIndices[1],
        spine, sortSwap, numberOfReferences, 0, mortonCodeBits);
    if (sortSwap) {
        mortonCodes[0] = mortonCodes[1];
        referenceIndices[0] = referenceIndices[1];
    }
    return time;
}

float LBVHBuilder::setupLeaves(HipBVH & bvh, int numberOfReferences) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("setupLeaves");

    // Set params.
    kernel.setParams(
        numberOfReferences,
        referenceIndices[0],
        referenceIndices[1],
        bvh.getTriangleIndices(),
        triangleIndices,
        nodeLeftIndices,
        nodeRightIndices,
        nodeBoxesMin,
        nodeBoxesMax,
        referenceBoxesMin,
        referenceBoxesMax
    );

    // Launch.
    return kernel.launchTimed(numberOfReferences);

}

float LBVHBuilder::construct(int numberOfReferences) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("construct60");

    // Set params.
    kernel.setParams(
        numberOfReferences,
        nodeParentIndices,
        nodeLeftIndices,
        nodeRightIndices,
        mortonCodes[0]
    );

    // Launch.
    return kernel.launchTimed(numberOfReferences);

}

float LBVHBuilder::refit(HipBVH & bvh, int numberOfReferences) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("refit");

    // Clear termination counters.
    bvh.termCounters.clear();

    // Set params.
    kernel.setParams(
        2 * numberOfReferences - 1,
        bvh.termCounters,
        nodeParentIndices,
        nodeLeftIndices,
        nodeRightIndices,
        nodeBoxesMin,
        nodeBoxesMax
    );

    // Launch.
    return kernel.launchTimed(numberOfReferences);

}

float LBVHBuilder::build(HipBVH & bvh, Scene * scene) {

    // Number of references.
    int numberOfReferences = 0;

    // Setup reference boxes.
    float setupBoxesTime = setupReferences(bvh, scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <LBVHBuilder> Reference boxes setup in " << setupBoxesTime << "s.\n";

    // Allocate.
    allocate(numberOfReferences);

    // Morton codes.
    float mortonCodesTime = computeMortonCodes(scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <LBVHBuilder> Morton codes (" << mortonCodeBits << " bits) computed in " << mortonCodesTime << "s.\n";

    // Sort.
    float sortTime = sortReferences(numberOfReferences);
    logger(LOG_INFO) << "INFO <LBVHBuilder> References sorted in " << sortTime << "s.\n";

    // Setup leaves.
    float setupLeavesTime = setupLeaves(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <LBVHBuilder> Leaves setup in " << setupLeavesTime << "s.\n";

    // Construction.
    float constructTime = construct(numberOfReferences);
    logger(LOG_INFO) << "INFO <LBVHBuilder> Topology constructed in " << constructTime << "s.\n";

    // Refit.
    float refitTime = refit(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <LBVHBuilder> Bounding boxes refitted in " << refitTime << "s.\n";

    // Collapse.
    float collapseTime;
    if (adaptiveLeafSize)
        collapseTime = collapser.collapseAdaptive(numberOfReferences, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    else
        collapseTime = collapser.collapse(numberOfReferences, maxLeafSize, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    logger(LOG_INFO) << "INFO <LBVHBuilder> BVH collapsed and converted in " << collapseTime << "s.\n";

    // Woopify triangles.
    float woopTime = bvh.woopifyTriangles();
    logger(LOG_INFO) << "INFO <LBVHBuilder> Triangles woopified in " << woopTime << "s.\n";

    return setupBoxesTime + mortonCodesTime + sortTime + setupLeavesTime + constructTime + refitTime + collapseTime + woopTime;

}

LBVHBuilder::LBVHBuilder() : mortonCodeBits(60) {
    compiler.setSourceFile("../src/hippie/rt/bvh/LBVHBuilderKernels.cu");
    compiler.compile();
    int _mortonCodeBits;
    Environment::getInstance()->getIntValue("Bvh.lbvhMortonCodeBits", _mortonCodeBits);
    setMortonCodeBits(_mortonCodeBits);
}

LBVHBuilder::~LBVHBuilder() {
}

HipBVH * LBVHBuilder::build(Scene * scene) {
    float time = 0.0f;
    return build(scene, time);
}

HipBVH * LBVHBuilder::build(Scene * scene, float & time) {
    HipBVH * bvh = new HipBVH(scene);
    time = rebuild(*bvh);
    return bvh;
}

float LBVHBuilder::rebuild(HipBVH & bvh) {
    Scene * scene = bvh.scene;
    float time = build(bvh, scene);
    logger(LOG_INFO) << "INFO <LBVHBuilder> Building BVH from " << scene->getNumberOfTriangles() << " triangles...\n";
    logger(LOG_INFO) << "INFO <LBVHBuilder> " << (scene->getNumberOfTriangles() * 1.0e-3f / time) << " MTriangles/s\n";
    logger(LOG_INFO) << "INFO <LBVHBuilder> BVH built in " << time << " seconds.\n";
    //bvh.validate();
    return time;
}

int LBVHBuilder::getMortonCodeBits() {
    return mortonCodeBits;
}

void LBVHBuilder::setMortonCodeBits(int mortonCodeBits) {
    if (mortonCodeBits < 6 || mortonCodeBits > 60) logger(LOG_WARN) << "WARN <LBVHBuilder> Morton code bits must be in range [6,60].\n";
    else this->mortonCodeBits = mortonCodeBits;
}

void LBVHBuilder::clear() {
    mortonCodes[0].free();
    mortonCodes[1].free();
    nodeLeftIndices.free();
    nodeRightIndices.free();
    nodeBoxesMin.free();
    nodeBoxesMax.free();
    nodeParentIndices.free();
    triangleIndices.free();
    collapser.clear();
    referenceIndices[0].clear();
    referenceIndices[1].clear();
    referenceBoxesMin.clear();
    referenceBoxesMax.clear();
}
