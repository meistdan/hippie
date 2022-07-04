/**
 * \file	PLOCBuilder.cpp
 * \author	Daniel Meister
 * \date	2015/10/29
 * \brief	PLOCBuilder class source file.
 */

#include <QElapsedTimer>
#include "PLOCBuilder.h"
#include "Presplitter.h"

void PLOCBuilder::allocate(int numberOfReferences) {
    LBVHBuilder::allocate(numberOfReferences);
    blockOffsets.resizeDiscard(sizeof(int) * divCeil(numberOfReferences, PLOC_SCAN_BLOCK_THREADS));
    nodeIndices[0].resizeDiscard(sizeof(int) * numberOfReferences);
    nodeIndices[1].resizeDiscard(sizeof(int) * numberOfReferences);
    neighbourDistances.resizeDiscard(sizeof(float) * (2 * numberOfReferences - 1));
    neighbourIndices.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
}

float PLOCBuilder::clustering(HipBVH & bvh, int numberOfReferences) {

    // Kernel time.
    float generateNeighboursTime = 0.0f;
    float mergeTime = 0.0f;
    float localPrefixScanTime = 0.0f;
    float globalPrefixScanTime = 0.0f;
    float compactTime = 0.0f;

    // Kernels.
    HipModule * module = plocCompiler.compile();
    HipKernel generateNeighboursCachedKernel = module->getKernel("generateNeighboursCached");
    HipKernel generateNeighboursKernel = module->getKernel("generateNeighbours");
    HipKernel mergeKernel = module->getKernel("merge");
    HipKernel localPrefixScanKernel = module->getKernel("localPrefixScan");
    HipKernel globalPrefixScanKernel = module->getKernel("globalPrefixScan");
    HipKernel compactKernel = module->getKernel("compact");

    // Number of clusters.
    int numberOfClusters = numberOfReferences;

    // Swap flag.
    bool swapBuffers = sortSwap;

    // Step counter.
    int steps = 0;

    // Radius.
    int r = radius;

    // Main loop.
    while (numberOfClusters > 1) {

        // Increment step counter.
        ++steps;

        // Adaptive radius.
        if (adaptive) r = qMin(r + 1, maxRadius);

        // Generate neighbours.
        if (r <= PLOC_GEN_BLOCK_THREADS / 2) {
            generateNeighboursCachedKernel.setParams(
                numberOfClusters,
                r,
                neighbourDistances,
                neighbourIndices,
                nodeBoxesMin,
                nodeBoxesMax,
                referenceIndices[swapBuffers]
            );
            generateNeighboursTime += generateNeighboursCachedKernel.launchTimed(numberOfClusters, Vec2i(PLOC_GEN_BLOCK_THREADS, 1));
        }
        else {
            generateNeighboursKernel.setParams(
                numberOfClusters,
                r,
                neighbourDistances,
                neighbourIndices,
                nodeBoxesMin,
                nodeBoxesMax,
                referenceIndices[swapBuffers]
            );
            generateNeighboursTime += generateNeighboursKernel.launchTimed(numberOfClusters, Vec2i(PLOC_GEN_BLOCK_THREADS, 1));
        }

        // Clear prefix scan offset.
        module->getGlobal("prefixScanOffset").clear();

        // Node offset.
        int nodeOffset = numberOfClusters - 2;

        // Merge.
        mergeKernel.setParams(
            numberOfClusters,
            nodeOffset,
            neighbourIndices,
            referenceIndices[swapBuffers],
            referenceIndices[!swapBuffers],
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            nodeBoxesMin,
            nodeBoxesMax
        );
        mergeTime += mergeKernel.launchTimed(numberOfClusters);

        // New number of clusters.
        int newNumberOfClusters = numberOfClusters - *(int*)module->getGlobal("prefixScanOffset").getPtr();

        // Swap buffers.
        swapBuffers = !swapBuffers;

        // Local prefix scan.
        int numberOfBlocks = divCeil(numberOfClusters, PLOC_SCAN_BLOCK_THREADS);
        localPrefixScanKernel.setParams(
            numberOfClusters,
            referenceIndices[swapBuffers],
            mortonCodes[0],
            blockOffsets
        );
        localPrefixScanTime += localPrefixScanKernel.launchTimed(numberOfBlocks * PLOC_SCAN_BLOCK_THREADS, Vec2i(PLOC_SCAN_BLOCK_THREADS, 1));

        // Global prefix scan.
        globalPrefixScanKernel.setParams(
            numberOfBlocks,
            blockOffsets
        );
        globalPrefixScanTime += globalPrefixScanKernel.launchTimed(PLOC_SCAN_BLOCK_THREADS, Vec2i(PLOC_SCAN_BLOCK_THREADS, 1));
     
        // Compact.
        compactKernel.setParams(
            numberOfClusters,
            referenceIndices[swapBuffers],
            referenceIndices[!swapBuffers],
            blockOffsets,
            mortonCodes[0]
        );
        compactTime += compactKernel.launchTimed(numberOfClusters, Vec2i(PLOC_SCAN_BLOCK_THREADS, 1));

        // Update number of clusters.
        numberOfClusters = newNumberOfClusters;

        // Swap buffers.
        swapBuffers = !swapBuffers;

    }

    // Log.
    logger(LOG_INFO) << "INFO <PLOCBuilder> BVH topology constructed in " << steps << " steps.\n";
    logger(LOG_INFO) << "INFO <PLOCBuilder> Neighbours generated in " << generateNeighboursTime << "s.\n";
    logger(LOG_INFO) << "INFO <PLOCBuilder> Clusters merged in " << mergeTime << "s.\n";
    logger(LOG_INFO) << "INFO <PLOCBuilder> Local prefix scan computed in " << localPrefixScanTime << "s.\n";
    logger(LOG_INFO) << "INFO <PLOCBuilder> Global prefix scan computed in " << globalPrefixScanTime << "s.\n";
    logger(LOG_INFO) << "INFO <PLOCBuilder> Node indices compacted in " << compactTime << "s.\n";

    // Kernels time.
    return generateNeighboursTime + mergeTime + localPrefixScanTime + globalPrefixScanTime + compactTime;

}

float PLOCBuilder::build(HipBVH & bvh, Scene * scene) {

    // Number of references.
    int numberOfReferences = 0;

    // Setup reference boxes.
    float setupBoxesTime = setupReferences(bvh, scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <PLOCBuilder> Reference boxes setup in " << setupBoxesTime << "s.\n";

    // Allocate.
    allocate(numberOfReferences);

    // Morton codes.
    float mortonCodesTime = computeMortonCodes(scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <PLOCBuilder> Morton codes (" << mortonCodeBits << " bits) computed in " << mortonCodesTime << "s.\n";

    // Sort.
    float sortTime = sortReferences(numberOfReferences);
    logger(LOG_INFO) << "INFO <PLOCBuilder> References sorted in " << sortTime << "s.\n";

    // Setup leaves.
    float setupLeavesTime = setupLeaves(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <PLOCBuilder> Leaves setup in " << setupLeavesTime << "s.\n";

    // Node indices.
    referenceIndices[0] = referenceIndices[1];

    // Clustering.
    float clusteringTime = clustering(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <PLOCBuilder> Topology constructed in " << clusteringTime << "s.\n";

    // Collapse.
    float collapseTime;
    if (adaptiveLeafSize)
        collapseTime = collapser.collapseAdaptive(numberOfReferences, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    else
        collapseTime = collapser.collapse(numberOfReferences, maxLeafSize, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    logger(LOG_INFO) << "INFO <PLOCBuilder> BVH collapsed and converted in " << collapseTime << "s.\n";

    // Woopify triangles.
    float woopTime = bvh.woopifyTriangles();
    logger(LOG_INFO) << "INFO <PLOCBuilder> Triangles woopified in " << woopTime << "s.\n";

    return setupBoxesTime + mortonCodesTime + setupLeavesTime + sortTime + clusteringTime + collapseTime + woopTime;

}

PLOCBuilder::PLOCBuilder() : radius(1), maxRadius(PLOC_GEN_BLOCK_THREADS) {
    plocCompiler.setSourceFile("../src/hippie/rt/bvh/PLOCBuilderKernels.cu");
    plocCompiler.compile();
    int _mortonCodeBits;
    Environment::getInstance()->getIntValue("Bvh.plocMortonCodeBits", _mortonCodeBits);
    setMortonCodeBits(_mortonCodeBits);
    int _maxRadius;
    Environment::getInstance()->getIntValue("Bvh.plocMaxRadius", _maxRadius);
    setMaxRadius(_maxRadius);
    int _radius;
    Environment::getInstance()->getIntValue("Bvh.plocRadius", _radius);
    setRadius(_radius);
    Environment::getInstance()->getBoolValue("Bvh.plocAdaptive", adaptive);
}

PLOCBuilder::~PLOCBuilder() {
}

HipBVH * PLOCBuilder::build(Scene * scene) {
    float time = 0.0f;
    return build(scene, time);
}

HipBVH * PLOCBuilder::build(Scene * scene, float & time) {
    HipBVH * bvh = new HipBVH(scene);
    time = rebuild(*bvh);
    return bvh;
}

float PLOCBuilder::rebuild(HipBVH & bvh) {
    Scene * scene = bvh.scene;
    float time = build(bvh, scene);
    logger(LOG_INFO) << "INFO <PLOCBuilder> Building BVH from " << scene->getNumberOfTriangles() << " triangles...\n";
    logger(LOG_INFO) << "INFO <PLOCBuilder> " << (scene->getNumberOfTriangles() * 1.0e-3f / time) << " MTriangles/s\n";
    logger(LOG_INFO) << "INFO <PLOCBuilder> BVH built in " << time << " seconds.\n";
    //bvh.validate();
    return time;
}

bool PLOCBuilder::isAdaptive() {
    return adaptive;
}

void PLOCBuilder::setAdaptive(bool adaptive) {
    this->adaptive = adaptive;
}

int PLOCBuilder::getRadius() {
    return radius;
}

void PLOCBuilder::setRadius(int radius) {
    if (radius < 1 || radius > maxRadius) logger(LOG_WARN) << "WARN <PLOCBuilder> Radius must be in range [1," << maxRadius << "].\n";
    else this->radius = radius;
}

int PLOCBuilder::getMaxRadius() {
    return maxRadius;
}

void PLOCBuilder::setMaxRadius(int maxRadius) {
    if (maxRadius < 1) logger(LOG_WARN) << "WARN <PLOCBuilder> Max. radius must be positive.\n";
    else this->maxRadius = maxRadius;
}

void PLOCBuilder::clear() {
    blockOffsets.free();
    nodeIndices[0].free();
    nodeIndices[1].free();
    neighbourDistances.free();
    neighbourIndices.free();
    LBVHBuilder::clear();
}
