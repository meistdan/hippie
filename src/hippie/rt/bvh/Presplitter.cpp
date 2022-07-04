/**
* \file	    Presplitter.h
* \author	Daniel Meister
* \date	    2019/05/02
* \brief	Presplitter class source file.
*/

#include "Presplitter.h"
#include "rt/bvh/SBVHBuilder.h"

#define PRESPLITTER_SBVH 1

SplitQueue::SplitQueue() {
}

SplitQueue::~SplitQueue() {
}

void SplitQueue::init(int maxSize, int size) {
    TaskQueue<SplitTask>::init(maxSize);
    for (int i = 0; i < 4; ++i)
        boxes[i].resizeDiscard(maxSize * sizeof(Vec4f));
    *(int*)this->size[0].getMutablePtr() = size;
}

void SplitQueue::clear() {
    for (int i = 0; i < 4; ++i) 
        boxes[i].free();
    TaskQueue<SplitTask>::clear();
}

Buffer & SplitQueue::getInBoxBuffer(int i) {
    Q_ASSERT(i >= 0 && i < 2);
    return swapBuffers ? boxes[1 + 2 * i] : boxes[0 + 2 * i];
}

Buffer & SplitQueue::getOutBoxBuffer(int i) {
    Q_ASSERT(i >= 0 && i < 2);
    return swapBuffers ? boxes[0 + 2 * i] : boxes[1 + 2 * i];
}

float Presplitter::computePriorities(Scene * scene) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("computePriorities");

    // Resize buffer.
    priorities.resizeDiscard(sizeof(float) * scene->getNumberOfTriangles());

    // Scene box.
    *(AABB*)module->getGlobal("sceneBox").getMutablePtr() = scene->getSceneBox();

    // Set params.
    kernel.setParams(
        scene->getNumberOfTriangles(),
        scene->getTriangleBuffer(),
        scene->getVertexBuffer(),
        priorities
    );

    // Launch.
    float time = kernel.launchTimed(scene->getNumberOfTriangles());

    // Kernel time.
    return time;

}

float Presplitter::sumPriorities(int numberOfTriangles, int & S, float D) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("sumPriorities");

    // Reset sum.
    *(float*)module->getGlobal("S").getMutablePtr() = 0;

    // Set params.
    kernel.setParams(
        numberOfTriangles,
        D,
        priorities
    );

    // Launch.
    float time = kernel.launchTimed(numberOfTriangles, Vec2i(REDUCTION_BLOCK_THREADS, 1));

    // Return sum.
    S = *(float*)module->getGlobal("S").getPtr();

    // Kernel time.
    return time;

}

float Presplitter::sumPrioritiesRound(int numberOfTriangles, int & S, float D) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("sumPrioritiesRound");

    // Reset sum.
    *(int*)module->getGlobal("S").getMutablePtr() = 0;

    // Set params.
    kernel.setParams(
        numberOfTriangles,
        D,
        priorities
    );

    // Launch.
    float time = kernel.launchTimed(numberOfTriangles);

    // Return sum.
    S = *(int*)module->getGlobal("S").getPtr();

    // Kernel time.
    return time;

}

float Presplitter::computeD(int numberOfTriangles, int Smax, int & S, float & D) {

    // Time
    float time = 0.0f;

    // Bounds.
    time += sumPriorities(numberOfTriangles, S, 1.0f); float Dmin = Smax / S;
    time += sumPrioritiesRound(numberOfTriangles, S, Dmin);  float Dmax = Dmin * Smax / S;

    // Binary search.
    const int ROUNDS = 6;
    for (int i = 0; i < ROUNDS; ++i) {

        // Average bounds.
        float Davg = 0.5f * (Dmin + Dmax);

        // Total number of splits.
        time += sumPrioritiesRound(numberOfTriangles, S, Davg);

        // Update bounds.
        if (S > Smax) Dmax = Davg;
        else Dmin = Davg;

    }
    
    // Final D and S.
    D = Dmin;
    time += sumPrioritiesRound(numberOfTriangles, S, D);

    // Kernel time.
    return time;
}

float Presplitter::initSplitTasks(Scene * scene, int numberOfReferences, float D) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("initSplitTasks");

    // Init queue.
    queue.init(numberOfReferences, scene->getNumberOfTriangles());

    // Set params.
    kernel.setParams(
        scene->getNumberOfTriangles(),
        D,
        priorities,
        scene->getTriangleBuffer(),
        scene->getVertexBuffer(),
        queue.getInBoxBuffer(0),
        queue.getInBoxBuffer(1),
        queue.getInBuffer()
    );

    // Launch.
    float time = kernel.launchTimed(scene->getNumberOfTriangles());

    // Kernel time.
    return time;

}

float Presplitter::split(Scene * scene, int numberOfReferences, Buffer & referenceBoxesMin, Buffer & referenceBoxesMax, Buffer & triangleIndices) {

    // Time
    float time = 0.0f;

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("split");

    // Reference counter.
    *(int*)module->getGlobal("prefixScanOffset").getMutablePtr() = 0;

    // Resize
    referenceBoxesMin.resizeDiscard(sizeof(Vec4f) * numberOfReferences);
    referenceBoxesMax.resizeDiscard(sizeof(Vec4f) * numberOfReferences);
    triangleIndices.resizeDiscard(sizeof(int) * numberOfReferences);

    while (queue.getInSize() > 0) {

        // Split.
        kernel.setParams(
            queue.getInSize(),
            queue.getOutSizeBuffer(),
            triangleIndices,
            scene->getTriangleBuffer(),
            scene->getVertexBuffer(),
            referenceBoxesMin,
            referenceBoxesMax,
            queue.getInBoxBuffer(0),
            queue.getInBoxBuffer(1),
            queue.getOutBoxBuffer(0),
            queue.getOutBoxBuffer(1),
            queue.getInBuffer(),
            queue.getOutBuffer()
        );
        time += kernel.launchTimed(queue.getInSize());

        // Ping - pong queues.
        queue.swap();

        // Reset counters.
        queue.resetOutSize();

    }

    // Reference counter.
    int refCounter = *(int*)module->getGlobal("prefixScanOffset").getPtr();

    // Kernel time.
    return time;

}

Presplitter::Presplitter() : beta(0.5f) {
    compiler.setSourceFile("../src/hippie/rt/bvh/PresplitterKernels.cu");
    float _beta;
    Environment::getInstance()->getFloatValue("Bvh.presplitterBeta", _beta);
    setBeta(_beta);
}

Presplitter::~Presplitter() {
}

float Presplitter::presplit(Scene * scene, Buffer & referenceBoxesMin, Buffer & referenceBoxesMax, Buffer & triangleIndices, int & numberOfReferences) {
#if PRESPLITTER_SBVH
    // SBVH builder for presplitting.
    SBVHBuilder builder;
    builder.setAdaptiveLeafSize(false);
    builder.setMaxLeafSize(1);
    BVH * bvh = builder.buildSBVH(scene);

    // Resize result buffers.
    numberOfReferences = bvh->getTriangleIndices().size();
    referenceBoxesMin.resizeDiscard(sizeof(Vec4f) * numberOfReferences);
    referenceBoxesMax.resizeDiscard(sizeof(Vec4f) * numberOfReferences);
    triangleIndices.resizeDiscard(sizeof(int) * numberOfReferences);
    Vec4f * referenceBoxesMinPtr = (Vec4f*)referenceBoxesMin.getMutablePtr();
    Vec4f * referenceBoxesMaxPtr = (Vec4f*)referenceBoxesMax.getMutablePtr();
    int * triangleIndicesPtr = (int*)triangleIndices.getMutablePtr();

    if (numberOfReferences != bvh->getNumberOfLeafNodes()) {
        logger(LOG_ERROR) << "ERROR <Presplitter> BVH must contain exactly one reference per leaf.\n";
        exit(EXIT_FAILURE);
    }

    // Find leaf boxes.
    int index = 0;
    QStack<const BVH::Node*> stack;
    stack.push(bvh->getRoot());
    while (!stack.empty()) {

        // Pop node.
        const BVH::Node * node = stack.top();
        stack.pop();

        // Leaf.
        if (node->isLeaf()) {
            Q_ASSERT(node->begin + 1 == node->end);
            referenceBoxesMinPtr[index] = Vec4f(node->box.mn, 0.0f);
            referenceBoxesMaxPtr[index] = Vec4f(node->box.mx, 0.0f);
            triangleIndicesPtr[index] = bvh->getTriangleIndices()[node->begin];
            ++index;
        }

        // Interior.
        else {
            const BVH::InteriorNode * interior = dynamic_cast<const BVH::InteriorNode*>(node);
            stack.push(interior->children[1]);
            stack.push(interior->children[0]);
        }

    }

    // Delete BVH.
    delete bvh;

    return 0.0f;
#else
    // Compute priorities.
    float computePrioritiesTime = computePriorities(scene);
    logger(LOG_INFO) << "INFO <Presplitter> Priorities computed in " << computePrioritiesTime << "s.\n";

    // Compute D.
    int S;
    float D;
    float Smax = getBeta() * scene->getNumberOfTriangles();
    float computeDTime = computeD(scene->getNumberOfTriangles(), Smax, S, D);
    logger(LOG_INFO) << "INFO <Presplitter> D computed in " << computeDTime << "s.\n";

    // Init split tasks.
    numberOfReferences = S + scene->getNumberOfTriangles();
    float initSplitTasksTime = initSplitTasks(scene, numberOfReferences, D);
    logger(LOG_INFO) << "INFO <Presplitter> Split tasks initialized in " << initSplitTasksTime << "s.\n";

    // Split.
    float splitTime = split(scene, numberOfReferences, referenceBoxesMin, referenceBoxesMax, triangleIndices);
    logger(LOG_INFO) << "INFO <Presplitter> Triangles splitted in " << splitTime << "s.\n";

    return computePrioritiesTime + computeDTime + initSplitTasksTime + splitTime;
#endif
}

float Presplitter::getBeta() {
    return beta;
}

void Presplitter::setBeta(float beta) {
    if (beta >= 0.0f) this->beta = beta;
    else logger(LOG_WARN) << "WARN <Presplitter> Beta must be non-negative.\n";
}

void Presplitter::clear() {
    priorities.clear();
    queue.clear();
}
