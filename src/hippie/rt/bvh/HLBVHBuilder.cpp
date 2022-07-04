/**
 * \file	HLBVHBuilder.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	HLBVHBuilder class source file.
 */

#include "HLBVHBuilder.h"
#include "util/Logger.h"

HLBVHQueue::HLBVHQueue(void) {
}

HLBVHQueue::~HLBVHQueue(void) {
}

void HLBVHQueue::init(int size, int binSize) {
    TaskQueue<HLBVHTask>::init(size);
    bins[0].resizeDiscard(3 * size * binSize * sizeof(Vec4f));
    bins[1].resizeDiscard(3 * size * binSize * sizeof(Vec4f));
    newTaskIndices.resizeDiscard(sizeof(int) * size);
    splitIndices.resizeDiscard(sizeof(int) * size);
    *(int*)this->size[0].getMutablePtr() = 1;
    *(HLBVHTask*)queue[0].getMutablePtr() = HLBVHTask(0, -1);
}

void HLBVHQueue::clear(void) {
    TaskQueue<HLBVHTask>::clear();
    for (int i = 0; i < 2; ++i)
        bins[i].free();
    newTaskIndices.free();
    splitIndices.free();
}

Buffer & HLBVHQueue::getBinsBuffer(int i) {
    Q_ASSERT(i >= 0 && i < 2);
    return bins[i];
}

Buffer & HLBVHQueue::getNewTaskIndicesBuffer(void) {
    return newTaskIndices;
}

Buffer & HLBVHQueue::getSplitIndicesBuffer(void) {
    return splitIndices;
}

void HLBVHBuilder::allocate(int numberOfReferences) {
    LBVHBuilder::allocate(numberOfReferences);
    nodeParentIndices.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
    nodeLeftIndices.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
    nodeRightIndices.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
    nodeStates.resizeDiscard(sizeof(int) * (2 * numberOfReferences - 1));
    nodeOffsets.resizeDiscard(sizeof(int) * (numberOfReferences - 1));
    leafClusterIndices.resizeDiscard(sizeof(int) * numberOfReferences);
    clusterTaskIndices.resizeDiscard(sizeof(int) * numberOfReferences);
    clusterNodeIndices.resizeDiscard(sizeof(int) * numberOfReferences);
    clusterBinIndices.resizeDiscard(sizeof(Vec4i) * numberOfReferences);
}

float HLBVHBuilder::computeNodeStates(HipBVH & bvh, int numberOfReferences) {

    // Kernel.
    HipModule * module = hlbvhCompiler.compile();
    HipKernel kernel = module->getKernel("computeNodeStates");

    // Clear termination counters.
    bvh.termCounters.clear();

    // Set params.
    kernel.setParams(
        numberOfReferences,
        mortonCodeBits,
        mortonCodeSAHBits,
        bvh.termCounters,
        LBVHBuilder::nodeParentIndices,
        LBVHBuilder::nodeLeftIndices,
        LBVHBuilder::nodeRightIndices,
        nodeStates,
        mortonCodes[0]
    );

    // Launch.
    return kernel.launchTimed(numberOfReferences);

}

float HLBVHBuilder::computeLeafClusterIndices(int numberOfReferences) {

    // Kernel.
    HipModule * module = hlbvhCompiler.compile();
    HipKernel kernel = module->getKernel("computeLeafClusterIndices");

    // Set params.
    kernel.setParams(
        numberOfReferences,
        leafClusterIndices,
        LBVHBuilder::nodeParentIndices,
        nodeStates
    );

    // Launch.
    return kernel.launchTimed(numberOfReferences);

}

float HLBVHBuilder::invalidateIntermediateClusters(HipBVH & bvh, int numberOfReferences) {

    // Kernel.
    HipModule * module = hlbvhCompiler.compile();
    HipKernel kernel = module->getKernel("invalidateIntermediateClusters");

    // Clear termination counters.
    bvh.termCounters.clear();

    // Set params.
    kernel.setParams(
        numberOfReferences,
        bvh.termCounters,
        leafClusterIndices,
        LBVHBuilder::nodeParentIndices,
        nodeStates
    );

    // Launch.
    return kernel.launchTimed(numberOfReferences);

}

float HLBVHBuilder::computeNodeOffsets(int numberOfReferences) {

    // Kernel.
    HipModule * module = hlbvhCompiler.compile();
    HipKernel kernel = module->getKernel("computeNodeOffsets");

    // Clear prefix scan offset.
    module->getGlobal("prefixScanOffset").clear();

    // Set params.
    kernel.setParams(
        numberOfReferences,
        nodeOffsets,
        nodeStates
    );

    // Launch.
    return kernel.launchTimed(numberOfReferences - 1);

}

float HLBVHBuilder::compact(int numberOfReferences) {

    // Kernel.
    HipModule * module = hlbvhCompiler.compile();
    HipKernel kernel = module->getKernel("compact");

    // Set params.
    kernel.setParams(
        2 * numberOfReferences - 1,
        nodeOffsets,
        nodeStates,
        LBVHBuilder::nodeParentIndices,
        LBVHBuilder::nodeLeftIndices,
        LBVHBuilder::nodeRightIndices,
        nodeParentIndices,
        nodeLeftIndices,
        nodeRightIndices
    );

    // Launch.
    return kernel.launchTimed(2 * numberOfReferences - 1);

}

float HLBVHBuilder::computeClusters(int numberOfReferences) {

    // Kernel.
    HipModule * module = hlbvhCompiler.compile();
    HipKernel kernel = module->getKernel("computeClusters");

    // Clear prefix scan offset.
    module->getGlobal("prefixScanOffset").clear();

    // Set params.
    kernel.setParams(
        2 * numberOfReferences - 1,
        mortonCodeBits,
        mortonCodeSAHBits,
        nodeStates,
        nodeOffsets,
        LBVHBuilder::nodeLeftIndices,
        clusterNodeIndices,
        clusterBinIndices,
        mortonCodes[0]
    );

    // Launch.
    float time = kernel.launchTimed(2 * numberOfReferences - 1);

    // Number of clusters.
    numberOfClusters = *(int*)module->getGlobal("prefixScanOffset").getPtr();

    // Kernel time.
    return time;

}

float HLBVHBuilder::refit(HipBVH & bvh, int numberOfReferences) {

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

float HLBVHBuilder::split() {

    // Kernel.
    HipModule * module = hlbvhCompiler.compile();
    HipKernel resetKernel = module->getKernel("resetBins");
    HipKernel binKernel = module->getKernel("binClusters");
    HipKernel splitKernel = module->getKernel("split");
    HipKernel distributeKernel = module->getKernel("distributeClusters");

    float resetTime = 0.0f;
    float binTime = 0.0f;
    float splitTime = 0.0f;
    float distTime = 0.0f;

    // Init. task indices.
    clusterTaskIndices.clear();

    // Node counter.
    *(int*)module->getGlobal("prefixScanOffset").getMutablePtr() = 1;

    // Number of bins.
    int numberOfBins = 1 << (mortonCodeSAHBits / 3);

    // Init queue.
    queue.init(numberOfClusters, numberOfBins);

    while (queue.getInSize() > 0) {

        // Reset bins.
        int numberOfAllBins = 3 * queue.getInSize() * numberOfBins;
        resetKernel.setParams(numberOfAllBins, queue.getBinsBuffer(0), queue.getBinsBuffer(1));
        resetTime += resetKernel.launchTimed(numberOfAllBins);

        // Binning.
        binKernel.setParams(
            queue.getInSize(),
            numberOfBins,
            numberOfClusters,
            clusterTaskIndices,
            clusterNodeIndices,
            clusterBinIndices,
            nodeBoxesMin,
            nodeBoxesMax,
            queue.getBinsBuffer(0),
            queue.getBinsBuffer(1)
        );
        binTime += binKernel.launchTimed(numberOfClusters);

        // Splitting.
        splitKernel.setParams(
            numberOfBins,
            queue.getInSize(),
            queue.getOutSizeBuffer(),
            queue.getNewTaskIndicesBuffer(),
            queue.getSplitIndicesBuffer(),
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            nodeBoxesMin,
            nodeBoxesMax,
            queue.getBinsBuffer(0),
            queue.getBinsBuffer(1),
            queue.getInBuffer(),
            queue.getOutBuffer()
        );
        splitTime += splitKernel.launchTimed(queue.getInSize());

        // Distribute clusters.
        distributeKernel.setParams(
            queue.getInSize(),
            numberOfBins,
            numberOfClusters,
            queue.getNewTaskIndicesBuffer(),
            queue.getSplitIndicesBuffer(),
            nodeParentIndices,
            nodeLeftIndices,
            nodeRightIndices,
            clusterTaskIndices,
            clusterNodeIndices,
            clusterBinIndices,
            queue.getBinsBuffer(0),
            queue.getBinsBuffer(1),
            queue.getInBuffer()
        );
        distTime += distributeKernel.launchTimed(numberOfClusters);

        // Ping - pong queues.
        queue.swap();

        // Reset counters.
        queue.resetOutSize();

    }

    return resetTime + binTime + splitTime + distTime;

}

float HLBVHBuilder::build(HipBVH & bvh, Scene * scene) {

    // Number of references.
    int numberOfReferences = 0;

    // Setup reference boxes.
    float setupBoxesTime = setupReferences(bvh, scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Reference boxes setup in " << setupBoxesTime << "s.\n";

    // Allocate.
    allocate(numberOfReferences);

    // Morton codes.
    float mortonCodesTime = computeMortonCodes(scene, numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Morton codes (" << mortonCodeBits << " bits) computed in " << mortonCodesTime << "s.\n";

    // Sort.
    float sortTime = sortReferences(numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> References sorted in " << sortTime << "s.\n";

    // Setup leaves.
    float setupLeavesTime = setupLeaves(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Leaves setup in " << setupLeavesTime << "s.\n";

    // Construction.
    float constructTime = construct(numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Topology constructed in " << constructTime << "s.\n";


    // Compute node states.
    float statesTime = computeNodeStates(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Node states computed in " << statesTime << "s\n";

    // Compute cluster indices.
    float clusterIndicesTime = computeLeafClusterIndices(numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Cluster indices computed in " << clusterIndicesTime << "s.\n";

    // Invalidate cluster intermediate clusters.
    float invalidateTime = invalidateIntermediateClusters(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Intermediate clusters invalidated in " << invalidateTime << "s.\n";

    // Compute node offsets.
    float offsetsTime = computeNodeOffsets(numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Node offsets computed in " << offsetsTime << "s.\n";

    // Compact.
    float compactTime = compact(numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Node compacted in " << compactTime << "s.\n";

    // Compute clusters.
    float clusterTime = computeClusters(numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Cluster computed in " << clusterTime << "s.\n";


    // Refit lower levels.
    float refitTime = refit(bvh, numberOfReferences);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Bounding boxes refitted in " << refitTime << "s.\n";

    // Build upper levels.
    float splitTime = split();
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Upper level constructed in " << splitTime << "s.\n";

    // Collapse.
    float collapseTime;
    if (adaptiveLeafSize)
        collapseTime = collapser.collapseAdaptive(numberOfReferences, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    else
        collapseTime = collapser.collapse(numberOfReferences, maxLeafSize, nodeParentIndices,
            nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> BVH collapsed and converted in " << collapseTime << "s.\n";

    // Woopify triangles.
    float woopTime = bvh.woopifyTriangles();
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Triangles woopified in " << woopTime << "s.\n";

    return setupBoxesTime + mortonCodesTime + sortTime + setupLeavesTime + constructTime + statesTime + clusterIndicesTime +
        invalidateTime + offsetsTime + compactTime + refitTime + splitTime + collapseTime + woopTime;

}

HLBVHBuilder::HLBVHBuilder() : mortonCodeSAHBits(15) {
    hlbvhCompiler.setSourceFile("../src/hippie/rt/bvh/HLBVHBuilderKernels.cu");
    hlbvhCompiler.compile();
    //hlbvhCompiler.addOption("-G");
    int _mortonCodeBits;
    Environment::getInstance()->getIntValue("Bvh.hlbvhMortonCodeBits", _mortonCodeBits);
    setMortonCodeBits(_mortonCodeBits);
    int _mortonCodeSAHBits;
    Environment::getInstance()->getIntValue("Bvh.hlbvhMortonCodeSAHBits", _mortonCodeSAHBits);
    setMortonCodeSAHBits(_mortonCodeSAHBits);
}

HLBVHBuilder::~HLBVHBuilder() {
}

int HLBVHBuilder::getMortonCodeSAHBits() {
    return mortonCodeSAHBits;
}

void HLBVHBuilder::setMortonCodeSAHBits(int mortonCodeSAHBits) {
    if (mortonCodeSAHBits < 6 || mortonCodeSAHBits > 15) logger(LOG_WARN) << "WARN <HLBVHBuilder> Morton code SAH bits must be in range [6,15].\n";
    else if (mortonCodeBits < mortonCodeSAHBits) logger(LOG_WARN) << "WARN <HLBVHBuilder> Morton code SAH bits must be less than Morton bits.\n";
    else this->mortonCodeSAHBits = mortonCodeSAHBits;
}

HipBVH * HLBVHBuilder::build(Scene * scene) {
    float time;
    return build(scene, time);
}

HipBVH * HLBVHBuilder::build(Scene * scene, float & time) {
    HipBVH * bvh = new HipBVH(scene);
    time = rebuild(*bvh);
    return bvh;
}

float HLBVHBuilder::rebuild(HipBVH & bvh) {
    Scene * scene = bvh.scene;
    float time = build(bvh, scene);
    logger(LOG_INFO) << "INFO <HLBVHBuilder> Building BVH from " << scene->getNumberOfTriangles() << " triangles...\n";
    logger(LOG_INFO) << "INFO <HLBVHBuilder> " << (scene->getNumberOfTriangles() * 1.0e-3f / time) << " MTriangles/s\n";
    logger(LOG_INFO) << "INFO <HLBVHBuilder> BVH built in " << time << " seconds.\n";
    //bvh.validate();
    return time;
}

void HLBVHBuilder::clear() {
    LBVHBuilder::clear();
    leafClusterIndices.free();
    nodeStates.free();
    nodeOffsets.free();
    nodeParentIndices.free();
    nodeLeftIndices.free();
    nodeRightIndices.free();
    clusterTaskIndices.free();
    clusterNodeIndices.free();
    clusterBinIndices.free();
    queue.clear();
}
