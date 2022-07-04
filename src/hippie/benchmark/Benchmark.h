/**
 * \file	Bnechmark.h
 * \author	Daniel Meister
 * \date	2014/08/19
 * \brief	Benchmark class header file.
 */

#ifndef _BENCHMARK_H_
#define _BENCHMARK_H_

#include "environment/AppEnvironment.h"
#include "rt/bvh/LBVHBuilder.h"
#include "rt/bvh/PLOCBuilder.h"
#include "rt/bvh/ATRBuilder.h"
#include "rt/bvh/TRBuilder.h"
#include "rt/bvh/SBVHBuilder.h"
#include "rt/bvh/InsertionBuilder.h"
#include "rt/renderer/Renderer.h"
#include "rt/scene/Interpolator.h"
#include "rt/scene/SceneLoader.h"
#include "util/ImageExporter.h"
#include "util/Logger.h"

#define BENCHMARK_CYCLES 1
#define BENCHMARK_PICTURE_PREFIX "image"

class Benchmark {

public:

    enum BVHUpdateMethod {
        REBUILD,
        REFIT
    };

private:

    Logger out;
    Buffer pixels;
    Buffer framePixels;

    ImageExporter exporter;
    Renderer renderer;

    BVHBuilder * builder;
    LBVHBuilder lbvhBuilder;
    HLBVHBuilder hlbvhBuilder;
    PLOCBuilder plocBuilder;
    ATRBuilder atrBuilder;
    TRBuilder trBuilder;
    SBVHBuilder sbvhBuilder;
    InsertionBuilder insertionBuilder;

    Interpolator interpolator;

    Scene * scene;
    HipBVH * bvh;
    BVHUpdateMethod updateMethod;

    SceneLoader sceneLoader;
    Camera camera;

    QString output;
    QString root;

    bool images;

    void testStatic(void);
    void testDynamic(void);

    QString rayTypeToString(Renderer::RayType type);
    QString updateMethodToString(BVHUpdateMethod updateMethod);

    void init(void);

public:

    Benchmark(void);
    ~Benchmark(void);

    void run(void);

};

#endif /* _BENCHMARK_H_ */
