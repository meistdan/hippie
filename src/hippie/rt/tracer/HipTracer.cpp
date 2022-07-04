/**
  * \file	HipTracer.cpp
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	HipTracer class source file.
  */

#include "HipTracer.h"
#include "util/Logger.h"

#include "rt/bvh/HipBVHNode.h"

float HipTracer::trace(RayBuffer & rays, const QString & kernelName) {

    int numRays = rays.getSize();
    if (!numRays) return 0.0f;

    // Check BVH consistency.
    if (!bvh) {
        logger(LOG_ERROR) << "ERROR <HipTracer> No BVH!\n";
        exit(EXIT_FAILURE);
    }

    // Compile kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel(kernelName);

    // Set parameters.
    kernel.setParams(
        numRays,                                    // numRays
        (rays.getClosestHit()) ? 0 : 1,				// anyHit
        rays.getRayBuffer().getHipPtr(),            // rays
        rays.getResultBuffer().getMutableHipPtr(),  // results
        bvh->getNodes().getHipPtr(),                // nodes
        bvh->getWoopTriangles().getHipPtr(),        // tris
        rays.getStatBuffer(),						// ray stats
        rays.getIndexBuffer(),						// ray indices
        bvh->getTriangleIndices().getHipPtr()       // triIndices
    );

    // Reset global ray counter.
    *(int*)module->getGlobal("g_warpCounter").getMutablePtr() = 0;

    // Determine block and grid sizes.
    Q_ASSERT(kernelConfig.desiredWarps != 0);
    Vec2i blockSize(kernelConfig.blockWidth, kernelConfig.blockHeight);
    int blockWarps = (blockSize.x * blockSize.y + 31) / 32;
    int numBlocks = (kernelConfig.desiredWarps + blockWarps - 1) / blockWarps;

    // Launch.
    return kernel.launchTimed(numBlocks * blockSize.x * blockSize.y, blockSize);

}

HipTracer::HipTracer() : bvh(nullptr) {}

HipTracer::~HipTracer() {
}

HipBVH * HipTracer::getBVH() {
    return bvh;
}

void HipTracer::setBVH(HipBVH * bvh) {

    // Store BVH.
    this->bvh = bvh;

    // Kernel name.
    QString kernelFilename;
    if (bvh->getLayout() == HipBVH::Layout::BIN) kernelFilename = "HipTracerBinKernels.cu";
    else if (bvh->getLayout() == HipBVH::Layout::QUAD) kernelFilename = "HipTracerQuadKernels.cu";
    else kernelFilename = "HipTracerOctKernels.cu";

    // Compile kernel.
    compiler.setSourceFile("../src/hippie/rt/tracer/" + kernelFilename);
    //compiler.clearDefines();
    HipModule * module = compiler.compile();

    // Initialize config with default values.
    {
        KernelConfig & c = *(KernelConfig*)module->getGlobal("g_config").getMutablePtr();
        c.blockWidth = 0;
        c.blockHeight = 0;
        c.desiredWarps = 0;
    }

    // Query config.
    module->getKernel("queryConfig").launch(1, Vec2i(1, 1));
    kernelConfig = *(const KernelConfig*)module->getGlobal("g_config").getPtr();

}

float HipTracer::trace(RayBuffer & rays) {
    return trace(rays, "trace");
}

float HipTracer::traceStats(RayBuffer & rays) {
    return trace(rays, "traceStats");
}

float HipTracer::traceSort(RayBuffer & rays) {
    float sortTime;
    float traceTime;
    return traceSort(rays, sortTime, traceTime);
}

float HipTracer::traceStatsSort(RayBuffer & rays) {
    float sortTime;
    float traceTime;
    return traceStatsSort(rays, sortTime, traceTime);
}

float HipTracer::traceSort(RayBuffer & rays, float & sortTime, float & traceTime) {
#if REAL_RAY_LENGTH
    trace(rays);
#endif
    sortTime = rays.mortonSort();
    traceTime = trace(rays, "traceSort");
    return sortTime + traceTime;
}

float HipTracer::traceStatsSort(RayBuffer & rays, float & sortTime, float & traceTime) {
#if REAL_RAY_LENGTH
    trace(rays);
#endif
    sortTime = rays.mortonSort();
    traceTime = trace(rays, "traceStatsSort");
    return sortTime + traceTime;
}
