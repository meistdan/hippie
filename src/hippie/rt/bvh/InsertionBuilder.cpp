/**
 * \file	InsertionBuilder.cpp
 * \author	Daniel Meister
 * \date	2017/01/30
 * \brief	InsertionBuilder class source file.
 */

#include <QDir>
#include <QElapsedTimer>
#include <QQueue>
#include "environment/AppEnvironment.h"
#include "InsertionBuilder.h"
#include "rt/bvh/SBVHBuilder.h"

void InsertionBuilder::allocate(int numberOfReferences) {
    LBVHBuilder::allocate(numberOfReferences);
    locks.resizeDiscard(sizeof(unsigned long long) * (2 * numberOfReferences - 1));
    areaReductions.resizeDiscard(sizeof(float) * (2 * numberOfReferences - 1));
    outNodeIndices.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
}

float InsertionBuilder::optimizeInsertion(HipBVH & bvh, int numberOfReferences, float initTime) {

    // Kernel times.
    float findBestNodeTime = 0.0f;
    float findBestNodeTimeT = 0.0f;
    float lockNodesTime = 0.0f;
    float checkLocksTime = 0.0f;
    float reinsertTime = 0.0f;
    float refitTime = 0.0f;
    float computeCostTime = 0.0f;

    // Kernels.
    HipModule * insertionModule = insertionCompiler.compile();
    HipKernel reinsertKernel = insertionModule->getKernel("reinsert");
    HipKernel computeCostKernel = insertionModule->getKernel("computeCost");
    HipModule * module = compiler.compile();
    HipKernel refitKernel = module->getKernel("refit");

    // Scene box.
    Vec3f sceneBoxMin = Vec3f(*(Vec4f*)nodeBoxesMin.getPtr());
    Vec3f sceneBoxMax = Vec3f(*(Vec4f*)nodeBoxesMax.getPtr());
    AABB sceneBox = AABB(sceneBoxMin, sceneBoxMax);
    float sceneBoxArea = sceneBox.area();

    // Step counter.
    int steps = 0;

    // Mod.
    int modCur = mod;

    // Number of nodes.
    int numberOfNodes = 2 * numberOfReferences - 1;

    // Number of inserted nodes.
    int insertedNodesTotal = 0;
    int insertedNodes;
    int foundNodes;

    // Costs.
    float cost = MAX_FLOAT;
    float bestCost = MAX_FLOAT;
    float prevCost;

    // Best BVH so far.
    bestNodeParentIndices = nodeParentIndices;
    bestNodeLeftIndices = nodeLeftIndices;
    bestNodeRightIndices = nodeRightIndices;

#if INSERTION_LOG
    // Stats logger.
    QString output;
    Environment::getInstance()->getStringValue("Benchmark.output", output);
    QString logDir = "benchmark/" + output;
    if (!QDir(logDir).exists())
        QDir().mkpath(logDir);
    Logger stats(logDir + "/sts_" + output + ".log");

    // Clear cost.
    insertionModule->getGlobal("cost").clear();

    // Compute intial cost.
    computeCostKernel.setParams(
        numberOfNodes,
        numberOfReferences,
        sceneBoxArea,
        bvh.getCt(),
        bvh.getCi(),
        nodeBoxesMin,
        nodeBoxesMax
    );
    computeCostTime += computeCostKernel.launchTimed(numberOfNodes, Vec2i(REDUCTION_BLOCK_THREADS, 1));

    // Cost.
    cost = *(float*)insertionModule->getGlobal("cost").getPtr();
    stats << initTime << " " << cost << " 0 0\n";
#endif

    while (true) {

        // Kernels.
        HipKernel findBestNodeKernel = insertionModule->getKernel("findBestNode");
        HipKernel lockNodesKernel = insertionModule->getKernel("lockNodes");
        HipKernel checkLocksKernel = insertionModule->getKernel("checkLocks");
        
        // Previous cost.
        prevCost = cost;

        // Find the best node.
        findBestNodeKernel.setParams(
            numberOfNodes,
            numberOfReferences,
            modCur,
            steps % modCur,
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            outNodeIndices,
            areaReductions,
            nodeBoxesMin,
            nodeBoxesMax
        );
        findBestNodeTimeT = findBestNodeKernel.launchTimed(divCeil(numberOfNodes, modCur));
        findBestNodeTime += findBestNodeTimeT;

        // Clear locks.
        locks.clear();

        // Lock nodes on paths.
        lockNodesKernel.setParams(
            numberOfNodes,
            numberOfReferences,
            modCur,
            steps % modCur,
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            outNodeIndices,
            areaReductions,
            locks
        );
        lockNodesTime += lockNodesKernel.launchTimed(divCeil(numberOfNodes, modCur));

        // Clear number of inserted nodes.
        insertionModule->getGlobal("insertedNodes").clear();
        insertionModule->getGlobal("foundNodes").clear();

        // Check locks.
        checkLocksKernel.setParams(
            numberOfNodes,
            numberOfReferences,
            modCur,
            steps % modCur,
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            outNodeIndices,
            areaReductions,
            locks
        );
        checkLocksTime += checkLocksKernel.launchTimed(divCeil(numberOfNodes, modCur));

        // Remove and insert nodes.
        reinsertKernel.setParams(
            numberOfNodes,
            modCur,
            steps % modCur,
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            outNodeIndices,
            areaReductions
        );
        reinsertTime += reinsertKernel.launchTimed(divCeil(numberOfNodes, modCur));

        // Check number of inserted nodes.
        foundNodes = *(int*)insertionModule->getGlobal("foundNodes").getPtr();
        insertedNodes = *(int*)insertionModule->getGlobal("insertedNodes").getPtr();
        insertedNodesTotal += insertedNodes;

        // Clear termination counters.
        bvh.termCounters.clear();

        // Refit.
        refitKernel.setParams(
            numberOfNodes,
            bvh.termCounters,
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            nodeBoxesMin,
            nodeBoxesMax
        );
        refitTime += refitKernel.launchTimed(numberOfReferences);

        // Clear cost.
        insertionModule->getGlobal("cost").clear();

        // Compute cost.
        computeCostKernel.setParams(
            numberOfNodes,
            numberOfReferences,
            sceneBoxArea,
            bvh.getCt(),
            bvh.getCi(),
            nodeBoxesMin,
            nodeBoxesMax
        );
        computeCostTime += computeCostKernel.launchTimed(numberOfNodes, Vec2i(REDUCTION_BLOCK_THREADS, 1));

        // Cost.
        cost = *(float*)insertionModule->getGlobal("cost").getPtr();

        // Update the best BVH.
        if (bestCost > cost) {
            bestCost = cost;
            bestNodeParentIndices = nodeParentIndices;
            bestNodeLeftIndices = nodeLeftIndices;
            bestNodeRightIndices = nodeRightIndices;
        }

        // Increment step counter.
        ++steps;

        // Log.
        logger(LOG_INFO) << "INFO <InsertionBuilder> Inserted nodes " << insertedNodes << " / " << foundNodes << " (" << float(insertedNodes) / float(foundNodes) << "), time " << findBestNodeTimeT << "s, cost " << cost << ".\n";
        qInfo() << "INFO <InsertionBuilder> Inserted nodes " << insertedNodes << "/" << foundNodes << "(" << float(insertedNodes) / float(foundNodes) << "), time" << findBestNodeTimeT << "s, cost" << cost << ".";

#if INSERTION_LOG
        // Log convergence.
        float time = initTime + findBestNodeTime + lockNodesTime + checkLocksTime + reinsertTime + refitTime + computeCostTime;
        stats << time << " " << cost << " " << foundNodes << " " << insertedNodes << " " << "\n";
#endif

        // Break conditions.
        const float BETA = 0.01f;
        if ((fabs(prevCost - cost) <= BETA || insertedNodes == 0) && modCur == 1) {
            break;
        }

        // Decrease mod. parameter.
        if (fabs(prevCost - cost) <= BETA) {
            modCur = qMax(1, modCur - 1);
        }

    }

    // Take the best BVH so far.
    nodeParentIndices = bestNodeParentIndices;
    nodeLeftIndices = bestNodeLeftIndices;
    nodeRightIndices = bestNodeRightIndices;

    // Refit best BVH.
    refitKernel.setParams(
        numberOfNodes,
        bvh.termCounters,
        nodeParentIndices,
        nodeLeftIndices,
        nodeRightIndices,
        nodeBoxesMin,
        nodeBoxesMax
    );
    refitTime += refitKernel.launchTimed(numberOfReferences);

    // Log.
    logger(LOG_INFO) << "INFO <InsertionBuilder> SBVH " << sbvh << ".\n";
    logger(LOG_INFO) << "INFO <InsertionBuilder> Mod " << mod << ".\n";
    logger(LOG_INFO) << "INFO <InsertionBuilder> BVH optimized in " << steps << " steps and "
        << insertedNodesTotal << " nodes were reinserted.\n";
    logger(LOG_INFO) << "INFO <InsertionBuilder> Nodes found in " << findBestNodeTime << "s.\n";
    logger(LOG_INFO) << "INFO <InsertionBuilder> Nodes locked in " << lockNodesTime << "s.\n";
    logger(LOG_INFO) << "INFO <InsertionBuilder> Locks checked in " << checkLocksTime << "s.\n";
    logger(LOG_INFO) << "INFO <InsertionBuilder> Nodes removed and inserted in " << reinsertTime << "s.\n";
    logger(LOG_INFO) << "INFO <InsertionBuilder> Bounding boxes refitted in " << refitTime << "s.\n";
    logger(LOG_INFO) << "INFO <InsertionBuilder> Cost computed in " << computeCostTime << "s.\n";

#if INSERTION_LOG
    // Phases logger.
    Logger phases(logDir + "/phs_" + output + ".log");
    // Log phases times.
    phases << findBestNodeTime << " " << lockNodesTime << " " << checkLocksTime << " " << reinsertTime << " " << refitTime << " " << computeCostTime << "\n";
#endif

    return findBestNodeTime + lockNodesTime + checkLocksTime + reinsertTime + refitTime + computeCostTime;

}

float InsertionBuilder::buildSBVH(HipBVH& bvh, Scene* scene, int & numberOfReferences) {

    // Queue entry.
    struct QueueEntry {
        const BVH::Node * node;
        int index;
        int parent;
        QueueEntry(const BVH::Node * node = nullptr, int index = 0, int parent = -1) : node(node), index(index), parent(parent) {}
    };

    // SBVH builder.
    SBVHBuilder builder;
    builder.setAdaptiveLeafSize(false);
    builder.setMaxLeafSize(1);
    BVH* sbvh = builder.buildSBVH(scene);

    // Allocate.
    numberOfReferences = sbvh->getTriangleIndices().size();
    referenceBoxesMin.resizeDiscard(sizeof(Vec4f)* numberOfReferences);
    referenceBoxesMax.resizeDiscard(sizeof(Vec4f)* numberOfReferences);
    bvh.resize(numberOfReferences);
    allocate(numberOfReferences);

    // A single triangle per leaf required.
    if (numberOfReferences != sbvh->getNumberOfLeafNodes()) {
        logger << "ERROR <InsertionBuilder> BVH must contain exactly one reference per leaf.\n";
        exit(EXIT_FAILURE);
    }

    // CPU pointers.
    int * nodeLeftIndicesPtr = (int*)nodeLeftIndices.getMutablePtr();
    int * nodeRightIndicesPtr = (int*)nodeRightIndices.getMutablePtr();
    int * nodeParentIndicesPtr = (int*)nodeParentIndices.getMutablePtr();
    Vec4f * nodeBoxesMinPtr = (Vec4f*)nodeBoxesMin.getMutablePtr();
    Vec4f * nodeBoxesMaxPtr = (Vec4f*)nodeBoxesMax.getMutablePtr();
    Vec4f * referenceBoxesMinPtr = (Vec4f*)referenceBoxesMin.getMutablePtr();
    Vec4f * referenceBoxesMaxPtr = (Vec4f*)referenceBoxesMax.getMutablePtr();
    int* triangleIndicesPtr = (int*)triangleIndices.getMutablePtr();

    // Convert to the GPU representation.
    int nextLeafIdx = numberOfReferences - 1;
    int nextInteriorIdx = 0;
    QQueue<QueueEntry> queue;
    queue.push_back(QueueEntry(sbvh->getRoot(), nextInteriorIdx++));
    while (!queue.empty()) {

        // Pop node.
        QueueEntry entry = queue.head();
        queue.pop_front();

        // Box and parent index.
        nodeParentIndicesPtr[entry.index] = entry.parent;
        nodeBoxesMinPtr[entry.index] = Vec4f(entry.node->box.mn, 0.0f);
        nodeBoxesMaxPtr[entry.index] = Vec4f(entry.node->box.mx, 0.0f);

        // Leaf.
        if (entry.node->isLeaf()) {
            Q_ASSERT(entry.node->begin + 1 == entry.node->end);
            int referenceIndex = entry.index - numberOfReferences + 1;
            referenceBoxesMinPtr[referenceIndex] = nodeBoxesMinPtr[entry.index];
            referenceBoxesMaxPtr[referenceIndex] = nodeBoxesMaxPtr[entry.index];
            triangleIndicesPtr[referenceIndex] = sbvh->getTriangleIndices()[entry.node->begin];
            nodeLeftIndicesPtr[entry.index] = referenceIndex;
            nodeRightIndicesPtr[entry.index] = referenceIndex + 1;
        }

        // Interior.
        else {
            const BVH::InteriorNode* interior = dynamic_cast<const BVH::InteriorNode*>(entry.node);
            BVH::Node* c0 = interior->children[0];
            if (c0->isLeaf()) queue.push_back(QueueEntry(c0, nextLeafIdx++, entry.index));
            else queue.push_back(QueueEntry(c0, nextInteriorIdx++, entry.index));
            nodeLeftIndicesPtr[entry.index] = queue.last().index;
            BVH::Node* c1 = interior->children[1];
            if (c1->isLeaf()) queue.push_back(QueueEntry(c1, nextLeafIdx++, entry.index));
            else queue.push_back(QueueEntry(c1, nextInteriorIdx++, entry.index));
            nodeRightIndicesPtr[entry.index] = queue.last().index;
        }

    }
    
    Q_ASSERT(nextInteriorIdx == numberOfReferences - 1);
    Q_ASSERT(nextLeafIdx == 2 * numberOfReferences - 1);

    // Delete BVH.
    delete sbvh;

    return 0.0f;

}

float InsertionBuilder::buildLBVH(HipBVH& bvh, Scene* scene, int & numberOfReferences) {

    // Setup reference boxes.
    float setupBoxesTime = setupReferences(bvh, scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <InsertionBuilder> Reference boxes setup in " << setupBoxesTime << "s.\n";

    // Allocate.
    allocate(numberOfReferences);

    // Morton codes.
    float mortonCodesTime = computeMortonCodes(scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <InsertionBuilder> Morton codes (" << mortonCodeBits << " bits) computed in " << mortonCodesTime << "s.\n";

    // Sort.
    float sortTime = sortReferences(numberOfReferences);
    logger(LOG_INFO) << "INFO <InsertionBuilder> Triangles sorted in " << sortTime << "s.\n";

    // Setup leaves.
    float setupLeavesTime = setupLeaves(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <InsertionBuilder> Leaves setup in " << setupLeavesTime << "s.\n";

    // Construction.
    float constructTime = construct(numberOfReferences);
    logger(LOG_INFO) << "INFO <InsertionBuilder> Topology constructed in " << constructTime << "s.\n";

    // Refit.
    float refitTime = refit(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <InsertionBuilder> Bounding boxes refitted in " << refitTime << "s.\n";

    return setupBoxesTime + mortonCodesTime + sortTime + setupLeavesTime + constructTime + refitTime;
}

float InsertionBuilder::build(HipBVH & bvh, Scene * scene) {

    // LBVH or SBVH
    int numberOfReferences = 0;
    float buildTime = 0.0f;
    if (sbvh) buildTime = buildSBVH(bvh, scene, numberOfReferences);
    else buildTime = buildLBVH(bvh, scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <InsertionBuilder> Initial BVH constructed in " << buildTime << "s.\n";

    // Optimize by insertion.
    float optimizeInsertionTime = optimizeInsertion(bvh, numberOfReferences, buildTime);
    logger(LOG_INFO) << "INFO <InsertionBuilder> BVH optimized by insertion in " << optimizeInsertionTime << "s.\n";

    // Collapse.
    float collapseTime;
    if (adaptiveLeafSize)
        collapseTime = collapser.collapseAdaptive(numberOfReferences, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    else
        collapseTime = collapser.collapse(numberOfReferences, maxLeafSize, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    logger(LOG_INFO) << "INFO <InsertionBuilder> BVH collapsed and converted in " << collapseTime << "s.\n";

    // Woopify triangles.
    float woopTime = bvh.woopifyTriangles();
    logger(LOG_INFO) << "INFO <InsertionBuilder> Triangles woopified in " << woopTime << "s.\n";

    return buildTime + optimizeInsertionTime + collapseTime + woopTime;

}

InsertionBuilder::InsertionBuilder() : mod(8) {
    insertionCompiler.setSourceFile("../src/hippie/rt/bvh/InsertionBuilderKernels.cu");
    insertionCompiler.compile();
    int _mortonCodeBits;
    Environment::getInstance()->getIntValue("Bvh.insertionMortonCodeBits", _mortonCodeBits);
    setMortonCodeBits(_mortonCodeBits);
    int _mod;
    Environment::getInstance()->getIntValue("Bvh.insertionMod", _mod);
    setMod(_mod);
    Environment::getInstance()->getBoolValue("Bvh.insertionSbvh", sbvh);
}

InsertionBuilder::~InsertionBuilder() {
}

HipBVH * InsertionBuilder::build(Scene * scene) {
    float time = 0.0f;
    return build(scene, time);
}

HipBVH * InsertionBuilder::build(Scene * scene, float & time) {
    HipBVH * bvh = new HipBVH(scene);
    time = rebuild(*bvh);
    return bvh;
}

float InsertionBuilder::rebuild(HipBVH & bvh) {
    Scene * scene = bvh.scene;
    float time = build(bvh, scene);
    logger(LOG_INFO) << "INFO <InsertionBuilder> Building BVH from " << scene->getNumberOfTriangles() << " triangles...\n";
    logger(LOG_INFO) << "INFO <InsertionBuilder> " << (scene->getNumberOfTriangles() * 1.0e-3f / time) << " MTriangles/s\n";
    logger(LOG_INFO) << "INFO <InsertionBuilder> BVH built in " << time << " seconds.\n";
    //bvh.validate();
    //clear();
    return time;
}

bool InsertionBuilder::isSbvh() {
    return sbvh;
}

void InsertionBuilder::setSbvh(bool sbvh) {
    this->sbvh = sbvh;
}

int InsertionBuilder::getMod() {
    return mod;
}

void InsertionBuilder::setMod(int mod) {
    if (mod < 1 || mod > 64) logger(LOG_WARN) << "WARN <InsertionBuilder> Mod must be in range [1,64].\n";
    else this->mod = mod;
}

int InsertionBuilder::getMortonCodeBits() {
    return mortonCodeBits;
}

void InsertionBuilder::setMortonCodeBits(int mortonCodeBits) {
    if (mortonCodeBits < 6 || mortonCodeBits > 60) logger(LOG_WARN) << "WARN <InsertionBuilder> Morton code bits must be in range [6,60].\n";
    else this->mortonCodeBits = mortonCodeBits;
}

void InsertionBuilder::clear() {
    LBVHBuilder::clear();
    collapser.clear();
    locks.free();
    areaReductions.free();
    bestNodeParentIndices.free();
    bestNodeLeftIndices.free();
    bestNodeRightIndices.free();
}
