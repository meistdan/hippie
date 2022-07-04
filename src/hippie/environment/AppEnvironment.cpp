/**
 * \file	AppEnvironment.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	AppEnvironment class source file.
 */

#include "AppEnvironment.h"

void AppEnvironment::registerOptions() {

    registerOption("Application.mode", "interactive", OPT_STRING);
    registerOption("Application.stats", "1", OPT_BOOL);

    registerOption("Benchmark.output", "default", OPT_STRING);
    registerOption("Benchmark.images", "0", OPT_BOOL);

    registerOption("Resolution.width", "1024", OPT_INT);
    registerOption("Resolution.height", "768", OPT_INT);

    registerOption("Scene.filename", OPT_STRING);
    registerOption("Scene.filefilter", OPT_STRING);
    registerOption("Scene.environment", OPT_STRING);
    registerOption("Scene.light", "5.0 -2.0 1.0", OPT_VECTOR);

    registerOption("Renderer.numberOfPrimarySamples", "1", OPT_INT);
    registerOption("Renderer.numberOfAOSamples", "8", OPT_INT);
    registerOption("Renderer.numberOfShadowSamples", "2", OPT_INT);
    registerOption("Renderer.aoRadius", "0.01", OPT_FLOAT);
    registerOption("Renderer.shadowRadius", "0.001", OPT_FLOAT);
    registerOption("Renderer.rayType", "primary", OPT_STRING);
    registerOption("Renderer.nodeSizeThreshold", "5000", OPT_INT);
    registerOption("Renderer.thermalThreshold", "250", OPT_INT);
    registerOption("Renderer.recursionDepth", "1", OPT_INT);
    registerOption("Renderer.keyValue", "0.6", OPT_FLOAT);
    registerOption("Renderer.whitePoint", "2.0", OPT_FLOAT);
    registerOption("Renderer.russianRoulette", "0", OPT_BOOL);

    registerOption("Renderer.sortShadowRays", "0", OPT_BOOL);
    registerOption("Renderer.sortAORays", "0", OPT_BOOL);
    registerOption("Renderer.sortPathRays", "0", OPT_BOOL);
    registerOption("Renderer.shadowRayLength", "0.0", OPT_FLOAT);
    registerOption("Renderer.aoRayLength", "0.25", OPT_FLOAT);
    registerOption("Renderer.pathRayLength", "0.25", OPT_FLOAT);
    registerOption("Renderer.shadowMortonCodeBits", "32", OPT_INT);
    registerOption("Renderer.aoMortonCodeBits", "32", OPT_INT);
    registerOption("Renderer.pathMortonCodeBits", "32", OPT_INT);
    registerOption("Renderer.shadowMortonCodeMethod", "aila", OPT_STRING);
    registerOption("Renderer.aoMortonCodeMethod", "aila", OPT_STRING);
    registerOption("Renderer.pathMortonCodeMethod", "aila", OPT_STRING);

    registerOption("Bvh.layout", "bin", OPT_STRING);
    registerOption("Bvh.ct", "3.0", OPT_FLOAT);
    registerOption("Bvh.ci", "2.0", OPT_FLOAT);
    registerOption("Bvh.maxLeafSize", "8", OPT_INT);
    registerOption("Bvh.adaptiveLeafSize", "1", OPT_BOOL);
    registerOption("Bvh.presplitting", "0", OPT_BOOL);
    registerOption("Bvh.lbvhMortonCodeBits", "60", OPT_INT);
    registerOption("Bvh.hlbvhMortonCodeBits", "60", OPT_INT);
    registerOption("Bvh.hlbvhMortonCodeSAHBits", "15", OPT_INT);
    registerOption("Bvh.trIterations", "2", OPT_INT);
    registerOption("Bvh.trMortonCodeBits", "60", OPT_INT);
    registerOption("Bvh.atrIterations", "2", OPT_INT);
    registerOption("Bvh.atrMortonCodeBits", "60", OPT_INT);
    registerOption("Bvh.insertionMortonCodeBits", "60", OPT_INT);
    registerOption("Bvh.insertionMod", "1", OPT_INT);
    registerOption("Bvh.insertionSbvh", "0", OPT_BOOL);
    registerOption("Bvh.plocRadius", "1", OPT_INT);
    registerOption("Bvh.plocMaxRadius", "512", OPT_INT);
    registerOption("Bvh.plocMortonCodeBits", "60", OPT_INT);
    registerOption("Bvh.plocAdaptive", "0", OPT_BOOL);
    registerOption("Bvh.sbvhAlpha", "0.00001", OPT_FLOAT);
    registerOption("Bvh.presplitterBeta", "0.1", OPT_FLOAT);
    registerOption("Bvh.method", "lbvh", OPT_STRING);
    registerOption("Bvh.update", "refit", OPT_STRING);

    registerOption("Camera.position", "0.0 0.0 0.0", OPT_VECTOR);
    registerOption("Camera.direction", "0.0 0.0 -1.0", OPT_VECTOR);
    registerOption("Camera.wheelAngle", "0.0", OPT_FLOAT);
    registerOption("Camera.nearPlane", "0.001", OPT_FLOAT);
    registerOption("Camera.farPlane", "3.0", OPT_FLOAT);
    registerOption("Camera.fieldOfView", "45.0", OPT_FLOAT);
    registerOption("Camera.step", "0.025", OPT_FLOAT);

    registerOption("Animation.frameRate", "30.0", OPT_FLOAT);
    registerOption("Animation.loop", "1", OPT_BOOL);
    registerOption("Animation.pause", "0", OPT_BOOL);

    registerOption("Screenshots.directory", "screenshots", OPT_STRING);

}

AppEnvironment::AppEnvironment() : Environment() {
    registerOptions();
}
