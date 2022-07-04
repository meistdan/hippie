/**
 * \file	ATRBuilder.cpp
 * \author	Daniel Meister
 * \date	2016/02/11
 * \brief	ATRBuilder class source file.
 */

#include "ATRBuilder.h"
#include "ATRBuilderKernels.h"

void ATRBuilder::allocate(int numberOfReferences) {
    LBVHBuilder::allocate(numberOfReferences);
    if (treeletSize > ATR_MAX_TREELET_SIZE) {
        int numberOfElements = sumArithmeticSequence(treeletSize - 1, 1, treeletSize - 1);
        int numberOfWarps = ((numberOfReferences + WARP_THREADS - 1) / WARP_THREADS);
        distanceMatrices.resizeDiscard(sizeof(float) * numberOfElements * numberOfWarps);
    }
    costs.resizeDiscard(sizeof(float) * (2 * numberOfReferences - 1));
    surfaceAreas.resizeDiscard(sizeof(float) * (2 * numberOfReferences - 1));
    subtreeReferences.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
}

void ATRBuilder::generateSchedule() {

    int numberOfElements = sumArithmeticSequence(treeletSize - 1, 1, treeletSize - 1);
    int elementsPerWarp = 2 * WARP_THREADS;
    int numberOfIterations = (numberOfElements + elementsPerWarp - 1) / elementsPerWarp;
    int scheduleSize = numberOfIterations * WARP_THREADS;

    schedule.resizeDiscard(scheduleSize * sizeof(int));
    schedule.clear();

    int count = 0;
    int * s = (int*)schedule.getMutablePtr();
    float multiplier = 0.0f;
    for (int i = 0; i < treeletSize; ++i) {
        for (int j = 0; j < i; ++j) {
            int index = count + static_cast<int>(multiplier) * WARP_THREADS;
            s[index] = (s[index] << 16) | (i << 8) | j;
            ++count;
            if (count == WARP_THREADS) {
                multiplier += 0.5f;
                count = 0;
            }
        }
    }

    // If multiplier was not integer, shift the last 'warpSize' elements over to the left.
    if (static_cast<int>(multiplier + 0.5f) > static_cast<int>(multiplier)) {
        for (int i = count; i < 32; ++i) {
            int index = i + static_cast<int>(multiplier) * WARP_THREADS;
            s[index] = (s[index] << 16);
        }
    }

}

float ATRBuilder::computeSurfaceAreas(int numberOfReferences) {

    // Kernel.
    HipModule * module = atrCompiler.compile();
    HipKernel kernel = module->getKernel("computeSurfaceAreas");

    // Set params.
    kernel.setParams(
        2 * numberOfReferences - 1,
        surfaceAreas,
        nodeBoxesMin,
        nodeBoxesMax
    );

    // Launch.
    float time = kernel.launchTimed(2 * numberOfReferences - 1);

    // Kernel time.
    return time;

}

float ATRBuilder::optimize(HipBVH & bvh, int numberOfReferences) {

    // Elapsed time.
    float time = 0.0f;

    // Generate schedule.
    generateSchedule();

    // Kernel.
    HipModule * module = atrCompiler.compile();
    HipKernel kernel = treeletSize > ATR_MAX_TREELET_SIZE ? module->getKernel("optimize") : module->getKernel("optimizeSmall");

    // Block and grid.
    int blockThreads = 128;
    int blocks = divCeil(numberOfReferences, blockThreads);
    int threads = blocks * blockThreads;

    // Shared memory.
    int treeletMemorySize = ((2 * treeletSize - 1) * sizeof(int) + treeletSize * sizeof(float));
    int sharedMemorySize = treeletMemorySize * (blockThreads / 32);
    int distanceMatrixSize = sumArithmeticSequence(treeletSize - 1, 1, treeletSize - 1);
    int distancesMatrixMemorySize = distanceMatrixSize * sizeof(float);
    if (treeletSize <= ATR_MAX_TREELET_SIZE)
        sharedMemorySize += distancesMatrixMemorySize * (blockThreads / 32);
    kernel.setSharedMemorySize(sharedMemorySize);

    // Schedule size.
    int scheduleSize = int(schedule.getSize()) / sizeof(int);

    // Optimization loop.
    int gamma = treeletSize;
    for (int i = 0; i < iterations; ++i) {

        // Clear counters.
        bvh.termCounters.clear();

        // Set params.
        if (treeletSize > ATR_MAX_TREELET_SIZE) {
            kernel.setParams(
                numberOfReferences,
                treeletSize,
                scheduleSize,
                distanceMatrixSize,
                gamma,
                bvh.getCi(),
                bvh.getCt(),
                costs,
                surfaceAreas,
                distanceMatrices,
                nodeLeftIndices,
                nodeRightIndices,
                nodeBoxesMin,
                nodeBoxesMax,
                bvh.termCounters,
                nodeParentIndices,
                subtreeReferences,
                schedule
            );
        }
        else {
            kernel.setParams(
                numberOfReferences,
                treeletSize,
                scheduleSize,
                distanceMatrixSize,
                gamma,
                bvh.getCi(),
                bvh.getCt(),
                costs,
                surfaceAreas,
                nodeLeftIndices,
                nodeRightIndices,
                nodeBoxesMin,
                nodeBoxesMax,
                bvh.termCounters,
                nodeParentIndices,
                subtreeReferences,
                schedule
            );
        }

        // Launch.
        time += kernel.launchTimed(threads, Vec2i(blockThreads, 1));

        // Inc. gamma.
        gamma *= 2;

        // Cost won't improve => Break.
        if (gamma > numberOfReferences) {
            logger(LOG_INFO) << "INFO <ATRBuilder> Gamma is greater than the number of triangles => Skip the rest of iterations (" << i << ").\n";
            break;
        }

    }

    // Kernel time.
    return time;

}

float ATRBuilder::build(HipBVH & bvh, Scene * scene) {

    // Number of references.
    int numberOfReferences = 0;

    // Setup reference boxes.
    float setupBoxesTime = setupReferences(bvh, scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <ATRBuilder> Reference boxes setup in " << setupBoxesTime << "s.\n";

    // Allocate.
    allocate(numberOfReferences);

    // Morton codes.
    float computeMortonCodesTime = computeMortonCodes(scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <ATRBuilder> Morton codes (" << mortonCodeBits << " bits) computed in " << computeMortonCodesTime << "s.\n";

    // Sort.
    float sortTime = sortReferences(numberOfReferences);
    logger(LOG_INFO) << "INFO <ATRBuilder> References sorted in " << sortTime << "s.\n";

    // Setup leaves.
    float setupLeavesTime = setupLeaves(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <ATRBuilder> Leaves setup in " << setupLeavesTime << "s.\n";

    // Construction.
    float constructTime = construct(numberOfReferences);
    logger(LOG_INFO) << "INFO <ATRBuilder> Topology constructed in " << constructTime << "s.\n";

    // Refit.
    float refitTime = refit(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <ATRBuilder> Bounding boxes refitted in " << refitTime << "s.\n";

    // Compute surface areas.
    float areasTime = computeSurfaceAreas(numberOfReferences);
    logger(LOG_INFO) << "INFO <ATRBuilder> Surface areas computed in " << areasTime << "s.\n";

    // Optimize.
    float optimizeTime = optimize(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <ATRBuilder> BVH optimized in " << optimizeTime << "s.\n";

    // Collapse.
    float collapseTime;
    if (adaptiveLeafSize)
        collapseTime = collapser.collapseAdaptive(numberOfReferences, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    else
        collapseTime = collapser.collapse(numberOfReferences, maxLeafSize, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    logger(LOG_INFO) << "INFO <ATRBuilder> BVH collapsed and converted in " << collapseTime << "s.\n";

    // Woopify triangles.
    float woopTime = bvh.woopifyTriangles();
    logger(LOG_INFO) << "INFO <ATRBuilder> Triangles woopified in " << woopTime << "s.\n";

    return setupBoxesTime + computeMortonCodesTime + sortTime + setupLeavesTime + constructTime + refitTime + areasTime + optimizeTime + collapseTime + woopTime;

}

ATRBuilder::ATRBuilder() : iterations(ATR_ITERATIONS), treeletSize(ATR_TREELET_SIZE) {
    atrCompiler.setSourceFile("../src/hippie/rt/bvh/ATRBuilderKernels.cu");
    atrCompiler.compile();
    int _mortonCodeBits;
    Environment::getInstance()->getIntValue("Bvh.atrMortonCodeBits", _mortonCodeBits);
    setMortonCodeBits(_mortonCodeBits);
    int _iterations;
    Environment::getInstance()->getIntValue("Bvh.atrIterations", _iterations);
    setIterations(_iterations);
}

ATRBuilder::~ATRBuilder() {
}

HipBVH * ATRBuilder::build(Scene * scene) {
    float time = 0.0f;
    return build(scene, time);
}

HipBVH * ATRBuilder::build(Scene * scene, float & time) {
    HipBVH * bvh = new HipBVH(scene);
    time = rebuild(*bvh);
    return bvh;
}

float ATRBuilder::rebuild(HipBVH & bvh) {
    Scene * scene = bvh.scene;
    float time = build(bvh, scene);
    logger(LOG_INFO) << "INFO <ATRBuilder> Building BVH from " << scene->getNumberOfTriangles() << " triangles...\n";
    logger(LOG_INFO) << "INFO <ATRBuilder> " << (scene->getNumberOfTriangles() * 1.0e-3f / time) << " MTriangles/s\n";
    logger(LOG_INFO) << "INFO <ATRBuilder> BVH built in " << time << " seconds.\n";
    //bvh.validate();
    return time;
}

int ATRBuilder::getTreeletSize() {
    return treeletSize;
}

void ATRBuilder::setTreeletSize(int treeletSize) {
    if (treeletSize >= 4 && treeletSize <= 30) this->treeletSize = treeletSize;
    else logger(LOG_WARN) << "WARN <ATRBuilder> Treelet size must be in range of [4,30].\n";
}

int ATRBuilder::getIterations() {
    return iterations;
}

void ATRBuilder::setIterations(int iterations) {
    if (iterations >= 0 && iterations <= 25) this->iterations = iterations;
    else logger(LOG_WARN) << "WARN <ATRBuilder> Iterations must be in range of [0,25].\n";
}

void ATRBuilder::clear() {
    LBVHBuilder::clear();
    schedule.free();
    distanceMatrices.free();
    costs.free();
    surfaceAreas.free();
    subtreeReferences.free();
}
