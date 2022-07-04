/**
  * \file	HipTracer.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	HipTracer class header file.
  */

#ifndef _HIP_TRACER_H_
#define _HIP_TRACER_H_

#include "HipTracerKernels.h"
#include "gpu/HipCompiler.h"
#include "rt/bvh/HipBVH.h"
#include "rt/ray/RayBuffer.h"

class HipTracer {

private:

    HipCompiler compiler;
    KernelConfig kernelConfig;
    HipBVH * bvh;

    HipTracer(const HipTracer&); // forbidden
    HipTracer & operator=(const HipTracer&); // forbidden

    float trace(RayBuffer & rays, const QString & kernelName);

public:

    HipTracer(void);
    ~HipTracer(void);

    HipBVH * getBVH(void);
    void setBVH(HipBVH * bvh);

    float trace(RayBuffer & rays);
    float traceStats(RayBuffer & rays);
    float traceSort(RayBuffer & rays);
    float traceStatsSort(RayBuffer & rays);
    float traceSort(RayBuffer & rays, float & sortTime, float & traceTime);
    float traceStatsSort(RayBuffer & rays, float & sortTime, float & traceTime);

};

#endif /* _HIP_TRACER_H_ */
