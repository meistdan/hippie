/**
 * \file	Renderer.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Renderer class header file.
 */

#ifndef _RENDERER_H_
#define _RENDERER_H_

#include "rt/bvh/HLBVHBuilder.h"
#include "rt/ray/RayGen.h"
#include "rt/scene/Camera.h"
#include "rt/tracer/HipTracer.h"

#define RENDERER_MAX_KEY_VALUE 2.0f
#define RENDERER_MAX_WHITE_POINT 2.0f
#define RENDERER_MAX_RADIUS 100.0f
#define RENDERER_MAX_SAMPLES 1024
#define RENDERER_MAX_BATCH_SIZE (8 * 1024 * 1024)
#define RENDERER_MAX_RECURSION_DEPTH 8

#define SORT_LOG 1

class PathQueue {

private:

    RayBuffer rays[2];
    bool swapBuffers;

public:

    PathQueue(void);
    ~PathQueue(void);

    void swap(void);
    void init(int size);

    RayBuffer & getInputRays(void);
    RayBuffer & getOutputRays(void);

};

class Renderer {

public:

    enum RayType {
        PRIMARY_RAYS,
        WHITTED_RAYS,
        SHADOW_RAYS,
        AO_RAYS,
        PATH_RAYS,
        PSEUDOCOLOR_RAYS,
        THERMAL_RAYS,
        MAX_RAYS
    };

private:

    HipCompiler compiler;
    HipTracer tracer;
    RayGen raygen;

    RayType rayType;
    float keyValue;
    float whitePoint;
    float aoRadius;
    float shadowRadius;
    int numberOfPrimarySamples;
    int numberOfAOSamples;
    int numberOfShadowSamples;
    int recursionDepth;
    int nodeSizeThreshold;
    int thermalThreshold;
    int numberOfHits;

    bool sortShadowRays;
    bool sortAORays;
    bool sortPathRays;

    int avgRayCounts[RENDERER_MAX_RECURSION_DEPTH];
    float sortTimes[RENDERER_MAX_RECURSION_DEPTH];
    float traceSortTimes[RENDERER_MAX_RECURSION_DEPTH];
    float traceTimes[RENDERER_MAX_RECURSION_DEPTH];

    int pass;
    int bounce;
    int frameIndex;

    unsigned long long numberOfPrimaryRays;
    unsigned long long numberOfShadowRays;
    unsigned long long numberOfAORays;
    unsigned long long numberOfPathRays;

    float primaryTraceTime;
    float shadowTraceTime;
    float aoTraceTime;
    float pathTraceTime;

    PathQueue pathQueue;
    RayBuffer primaryRays;
    RayBuffer aoRays;
    RayBuffer shadowRays;
    Buffer auxPixels;
    Buffer decreases;
    Buffer seeds;

    RayBuffer::MortonCodeMethod stringToMortonCodeMethod(const QString & mortonCodeMethod);
    RayType stringToRayType(const QString & rayType);

    float computeRayHits(RayBuffer & rays);
    float initDecreases(int numberOfPixels);
    float interpolateColors(int numberOfPixels, Buffer & pixels, Buffer & framePixels);

    float primaryPass(Scene & scene, Camera & camera, Buffer & pixels);
    float shadowPass(Scene & scene, RayBuffer & inRays, Buffer & inPixels, Buffer & outPixels, bool replace);
    float aoPass(Scene & scene, RayBuffer & inRays, Buffer & inPixels, Buffer & outPixels, bool replace);
    float pathPass(Scene & scene, Buffer & pixels, RayBuffer & inRays, RayBuffer & outRays);

    float renderPrimary(Scene & scene, Camera & camera, Buffer & pixels);
    float renderShadow(Scene & scene, Camera & camera, Buffer & pixels);
    float renderAO(Scene & scene, Camera & camera, Buffer & pixels);
    float renderPath(Scene & scene, Camera & camera, Buffer & pixels);
    float renderPseudocolor(Scene & scene, Camera & camera, Buffer & pixels);
    float renderThermal(Camera & camera, Buffer & pixels);

    float reconstructSmooth(Scene & scene, RayBuffer & rays, Buffer & pixels);
    float reconstructPseudocolor(Scene & scene, Buffer & pixels);
    float reconstructThermal(Buffer & pixels);
    float reconstructShadow(RayBuffer & inRays, Buffer & inPixels, Buffer & outPixels, int batchBegin, int batchEnd, bool replace);
    float reconstructAO(RayBuffer & inRays, Buffer & inPixels, Buffer & outPixels, int batchBegin, int batchEnd, bool replace);

    float tracePrimaryRays(Camera & camera);
    float traceShadowRays(Scene & scene, RayBuffer & inRays, int batchBegin, int batchEnd);
    float traceAORays(Scene & scene, RayBuffer & inRays, int batchBegin, int batchEnd);
    float tracePathRays(Scene & scene, RayBuffer & inRays, RayBuffer & outRays);

public:

    Renderer(void);
    ~Renderer(void);

    RayType getRayType(void);
    void setRayType(RayType rayType);
    void setKeyValue(float keyValue);
    float getKeyValue(void);
    void setWhitePoint(float whitePoint);
    float getWhitePoint(void);
    float getAORadius(void);
    void setAORadius(float aoRadius);
    float getShadowRadius(void);
    void setShadowRadius(float shadowRadius);
    int getNumberOfPrimarySamples(void);
    void setNumberOfPrimarySamples(int numberOfPrimarySamples);
    int getNumberOfAOSamples(void);
    void setNumberOfAOSamples(int numberOfAOSamples);
    int getNumberOfShadowSamples(void);
    void setNumberOfShadowSamples(int numberOfShadowSamples);
    int getRecursionDepth(void);
    void setRecursionDepth(int recursionDepth);
    int getNodeSizeThreshold(void);
    void setNodeSizeThreshold(int nodeSizeThreshold);
    int getThermalThreshold(void);
    void setThermalThreshold(int thermalThreshold);
    bool getRussianRoulette(void);
    void setRussianRoulette(bool russianRoulette);

    bool getSortShadowRays(void);
    void setSortShadowRays(bool sortShadowRays);
    bool getSortAORays(void);
    void setSortAORays(bool sortAORays);
    bool getSortPathRays(void);
    void setSortPathRays(bool sortPathRays);

    float getShadowRayLength(void);
    void setShadowRayLength(float shadowRayLength);
    float getAORayLength(void);
    void setAORayLength(float aoRayLength);
    float getPathRayLength(void);
    void setPathRayLength(float pathRayLength);

    int getShadowMortonCodeBits(void);
    void setShadowMortonCodeBits(int shadowMortonCodeBits);
    int getAOMortonCodeBits(void);
    void setAOMortonCodeBits(int aoMortonCodeBits);
    int getPathMortonCodeBits(void);
    void setPathMortonCodeBits(int pathMortonCodeBits);

    RayBuffer::MortonCodeMethod getShadowMortonCodeMethod(void);
    void setShadowMortonCodeMethod(RayBuffer::MortonCodeMethod shadowMortonCodeMethod);
    RayBuffer::MortonCodeMethod getAOMortonCodeMethod(void);
    void setAOMortonCodeMethod(RayBuffer::MortonCodeMethod aoMortonCodeMethod);
    RayBuffer::MortonCodeMethod getPathMortonCodeMethod(void);
    void setPathMortonCodeMethod(RayBuffer::MortonCodeMethod pathMortonCodeMethod);

    float render(Scene & scene, HipBVH & bvh, Camera & camera, Buffer & pixels, Buffer & framePixels);

    void resetFrameIndex(void);

    unsigned long long getNumberOfPrimaryRays(void);
    unsigned long long getNumberOfShadowRays(void);
    unsigned long long getNumberOfAORays(void);
    unsigned long long getNumberOfPathRays(void);
    unsigned long long getNumberOfRays(void);

    float getPrimaryTraceTime(void);
    float getShadowTraceTime(void);
    float getAOTraceTime(void);
    float getPathTraceTime(void);
    float getTraceTime(void);

    float getPrimaryTracePerformance(void);
    float getShadowTracePerformance(void);
    float getAOTracePerformance(void);
    float getPathTracePerformance(void);
    float getTracePerformance(void);

};

#endif /* _RENDERER_H_ */
