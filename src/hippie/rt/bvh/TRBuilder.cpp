/**
 * \file	TRBuilder.cpp
 * \author	Daniel Meister
 * \date	2016/03/14
 * \brief	TRBuilder class source file.
 */

#include "TRBuilder.h"
#include "TRBuilderKernels.h"

void TRBuilder::allocate(int numberOfReferences) {
    LBVHBuilder::allocate(numberOfReferences);
    costs.resizeDiscard(sizeof(float) * (2 * numberOfReferences - 1));
    surfaceAreas.resizeDiscard(sizeof(float) * (2 * numberOfReferences - 1));
    int numberOfWarps = ((numberOfReferences + WARP_THREADS - 1) / WARP_THREADS);
    int subsetSize = (1 << treeletSize) * numberOfWarps;
    subsetAreas.resizeDiscard(sizeof(int) * subsetSize);
    subsetBoxesMin.resizeDiscard(4 * sizeof(float) * subsetSize);
    subsetBoxesMax.resizeDiscard(4 * sizeof(float) * subsetSize);
    subtreeReferences.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
    stackNode.resizeDiscard(sizeof(int) * (treeletSize - 1) * numberOfWarps);
    stackMask.resizeDiscard(sizeof(char) * (treeletSize - 1) * numberOfWarps);
    stackSize.resizeDiscard(sizeof(int) * numberOfWarps);
    currentInternalNode.resizeDiscard(sizeof(int) * numberOfWarps);
}

int TRBuilder::populationCount(int x) {
    int count = 0;
    while (x > 0) {
        ++count;
        x &= x - 1;
    }
    return count;
}

void TRBuilder::processSubset(int subset, int superset, QVector<QSet<int>> & dependencies, QVector<QSet<int>> & subsetsBySize) {

    int subsetSize = populationCount(subset);
    if (subsetSize == 1)
        return;

    if (subsetSize <= treeletSize - 2) {
        dependencies[subset].insert(superset);
        subsetsBySize[subsetSize].insert(subset);
    }

    // Handle dependencies.
    if (subsetSize > 2) {
        // Find each partition of the subset.
        int delta = (subset - 1) & subset;
        int partition = (-delta) & subset;
        while (partition != 0) {
            int partitionComplement = partition ^ subset;
            processSubset(partition, subset, dependencies, subsetsBySize);
            processSubset(partitionComplement, subset, dependencies, subsetsBySize);
            partition = (partition - delta) & subset;
        }
    }

}

void TRBuilder::generateSchedule() {

    int numberOfSubsets = (1 << treeletSize) - 1;

    QVector<QVector<int>> hostSchedule;

    // Group of subsets that can be composed with each subset.
    QVector<QSet<int>> dependencies(numberOfSubsets + 1);

    // Subsets grouped by size.
    QVector<QSet<int>> subsetsPerSize(treeletSize - 1);

    // Round that each subset should be processed in.
    std::vector<int> subsetRounds(numberOfSubsets + 1, -1);

    // Recursively process subsets, creating a list of dependencies that will be used to assemble the schedule.
    processSubset(numberOfSubsets, 0, dependencies, subsetsPerSize);

    // Subset size.
    for (int i = treeletSize - 2; i >= 0; --i) {
        // Subset.
        for (auto & subset : subsetsPerSize[i]) {
            unsigned int minimumRound = 0;

            // Check dependencies.
            for (auto & dependency : dependencies[subset]) {
                unsigned int dependencyRound = subsetRounds[dependency];
                minimumRound = qMax(minimumRound, dependencyRound + 1);
            }

            // Insert at a round.
            int round = minimumRound;
            if (hostSchedule.size() <= round)
                hostSchedule.push_back(QVector<int>());

            while (hostSchedule[round].size() == WARP_THREADS) {
                ++round;
                if (hostSchedule.size() <= round)
                    hostSchedule.push_back(QVector<int>());
            }
            hostSchedule[round].push_back(subset);
            subsetRounds[subset] = round;
        }
    }

    for (int i = 0; i < hostSchedule.size(); ++i) {
        for (int j = hostSchedule[i].size(); j < WARP_THREADS; ++j) {
            hostSchedule[i].push_back(0);
        }
    }

    std::reverse(hostSchedule.begin(), hostSchedule.end());

    schedule.resizeDiscard(sizeof(int) * WARP_THREADS * hostSchedule.size());
    int * s = (int*)schedule.getMutablePtr();
    for (int i = 0; i < hostSchedule.size(); ++i)
        memcpy(s + i * WARP_THREADS, hostSchedule[i].data(), sizeof(int) * WARP_THREADS);

}

float TRBuilder::computeSurfaceAreas(int numberOfReferences) {

    // Kernel.
    HipModule * module = trCompiler.compile();
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

float TRBuilder::optimize(HipBVH & bvh, int numberOfReferences) {

    // Elapsed time.
    float time = 0.0f;

    // Generate schedule.
    generateSchedule();

    // Kernel.
    HipModule * module = trCompiler.compile();
    HipKernel kernel = module->getKernel("optimize");

    // Block and grid.
    int blockThreads = 128;
    int blocks = divCeil(numberOfReferences, blockThreads);
    int threads = blocks * blockThreads;

    // Shared memory.
    int treeletMemorySize = ((2 * treeletSize - 1) * sizeof(int) + treeletSize * sizeof(float));
    int costAndMaskSize = int(1 << treeletSize) * sizeof(float) + int(1 << treeletSize) * sizeof(char);
    int sharedMemorySize = (treeletMemorySize + costAndMaskSize) * (blockThreads / 32);
    kernel.setSharedMemorySize(sharedMemorySize);

    // Number of rounds.
    int numberOfRounds = int(schedule.getSize()) / (sizeof(int) * WARP_THREADS);

    // Optimization loop.
    int gamma = treeletSize;
    for (int i = 0; i < iterations; ++i) {

        // Clear counters.
        bvh.termCounters.clear();

        // Set params.
        kernel.setParams(
            numberOfReferences,
            numberOfRounds,
            treeletSize,
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
            subsetAreas,
            subsetBoxesMin,
            subsetBoxesMax,
            schedule,
            currentInternalNode,
            stackNode,
            stackSize,
            stackMask
        );

        // Launch.
        time += kernel.launchTimed(threads, Vec2i(blockThreads, 1));

        // Inc. gamma.
        gamma *= 2;

        // Cost won't improve => Break.
        if (gamma > numberOfReferences) {
            logger(LOG_INFO) << "INFO <TRBuilder> Gamma is greater than the number of references => Skip the rest of iterations (" << i << ").\n";
            break;
        }

    }

    // Kernel time.
    return time;

}

float TRBuilder::build(HipBVH & bvh, Scene * scene) {

    // Number of references.
    int numberOfReferences = 0;

    // Setup reference boxes.
    float setupBoxesTime = setupReferences(bvh, scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <TRBVHBuilder> Rference boxes setup in " << setupBoxesTime << "s.\n";

    // Allocate.
    allocate(numberOfReferences);

    // Morton codes.
    float computeMortonCodesTime = computeMortonCodes(scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <TRBuilder> Morton codes (" << mortonCodeBits << " bits) computed in " << computeMortonCodesTime << "s.\n";

    // Sort.
    float sortTime = sortReferences(numberOfReferences);
    logger(LOG_INFO) << "INFO <TRBuilder> References sorted in " << sortTime << "s.\n";

    // Setup leaves.
    float setupLeavesTime = setupLeaves(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <TRBuilder> Leaves setup in " << setupLeavesTime << "s.\n";

    // Construction.
    float constructTime = construct(numberOfReferences);
    logger(LOG_INFO) << "INFO <TRBuilder> Topology constructed in " << constructTime << "s.\n";

    // Refit.
    float refitTime = refit(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <TRBuilder> Bounding boxes refitted in " << refitTime << "s.\n";

    // Compute surface areas.
    float areasTime = computeSurfaceAreas(numberOfReferences);
    logger(LOG_INFO) << "INFO <TRBuilder> Surface areas compute in " << areasTime << "s.\n";

    // Optimize.
    float optimizeTime = optimize(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <TRBuilder> BVH optimized in " << optimizeTime << "s.\n";

    // Collapse.
    float collapseTime;
    if (adaptiveLeafSize)
        collapseTime = collapser.collapseAdaptive(numberOfReferences, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    else
        collapseTime = collapser.collapse(numberOfReferences, maxLeafSize, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    logger(LOG_INFO) << "INFO <TRBuilder> BVH collapsed and converted in " << collapseTime << "s.\n";

    // Woopify triangles.
    float woopTime = bvh.woopifyTriangles();
    logger(LOG_INFO) << "INFO <TRBuilder> Triangles woopified in " << woopTime << "s.\n";

    return setupBoxesTime + computeMortonCodesTime + sortTime + setupLeavesTime + constructTime + refitTime + areasTime + optimizeTime + collapseTime + woopTime;

}

TRBuilder::TRBuilder() : iterations(TR_ITERATIONS), treeletSize(TR_TREELET_SIZE) {
    trCompiler.setSourceFile("../src/hippie/rt/bvh/TRBuilderKernels.cu");
    trCompiler.compile();
    int _mortonCodeBits;
    Environment::getInstance()->getIntValue("Bvh.trMortonCodeBits", _mortonCodeBits);
    setMortonCodeBits(_mortonCodeBits);
    int _iterations;
    Environment::getInstance()->getIntValue("Bvh.trIterations", _iterations);
    setIterations(_iterations);
}

TRBuilder::~TRBuilder() {
}

HipBVH * TRBuilder::build(Scene * scene) {
    float time = 0.0f;
    return build(scene, time);
}

HipBVH * TRBuilder::build(Scene * scene, float & time) {
    HipBVH * bvh = new HipBVH(scene);
    time = rebuild(*bvh);
    return bvh;
}

float TRBuilder::rebuild(HipBVH & bvh) {
    Scene * scene = bvh.scene;
    float time = build(bvh, scene);
    logger(LOG_INFO) << "INFO <TRBuilder> Building BVH from " << scene->getNumberOfTriangles() << " triangles...\n";
    logger(LOG_INFO) << "INFO <TRBuilder> " << (scene->getNumberOfTriangles() * 1.0e-3f / time) << " MTriangles/s\n";
    logger(LOG_INFO) << "INFO <TRBuilder> BVH built in " << time << " seconds.\n";
    //bvh.validate();
    return time;
}

int TRBuilder::getTreeletSize() {
    return treeletSize;
}

void TRBuilder::setTreeletSize(int treeletSize) {
    if (treeletSize >= 7 && treeletSize <= 7) this->treeletSize = treeletSize;
    else logger(LOG_WARN) << "WARN <TRBuilder> Treelet size must be in range of [7,7].\n";
}

int TRBuilder::getIterations() {
    return iterations;
}

void TRBuilder::setIterations(int iterations) {
    if (iterations >= 0 && iterations <= 25) this->iterations = iterations;
    else logger(LOG_WARN) << "WARN <TRBuilder> Iterations must be in range of [0,25].\n";
}

void TRBuilder::clear() {
    LBVHBuilder::clear();
    schedule.free();
    costs.free();
    surfaceAreas.free();
    subsetAreas.free();
    subsetBoxesMin.free();
    subsetBoxesMax.free();
    subtreeReferences.free();
    stackNode.free();
    stackMask.free();
    stackSize.free();
    currentInternalNode.free();
}
