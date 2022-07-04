/**
 * \file	Interpolator.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Interpolator class source file.
 */

#include "Interpolator.h"
#include "InterpolatorKernels.h"
#include "util/Logger.h"
#include <QtMath>

Interpolator::Interpolator() {
    compiler.setSourceFile("../src/hippie/rt/scene/InterpolatorKernels.cu");
}


Interpolator::~Interpolator() {
}

float Interpolator::update(DynamicScene & scene) {
    return updateAdaptive(scene, -1.0f);
}

float Interpolator::updateAdaptive(DynamicScene & scene, float time) {

    // Update scene time.
    float frameRate = scene.getFrameRate();
    float sceneTime = 0.0f;
    if (time < 0.0f) sceneTime = scene.time + 1.0f / scene.getFrameRate();
    else sceneTime = scene.time + qCeil(time * frameRate) / frameRate;
    if (scene.isLooped()) {
        while (sceneTime < 0.0f) sceneTime += 1.0f;
        while (sceneTime >= 1.0f) sceneTime -= 1.0f;
    }
    else {
        if (sceneTime < 0.0f) sceneTime = 0.0f;
        if (sceneTime >= 1.0f) sceneTime = 1.0f - INTERPOLATOR_EPSILON;
    }
    scene.time = sceneTime;
    float alpha = sceneTime;
    float frameIndex = alpha * (scene.getNumberOfFrames() - 1);

    // Bounding frame indices.
    int frameIndex0 = int(frameIndex);
    int frameIndex1 = (frameIndex0 + 1) % scene.getNumberOfFrames();

    // Interpolation parameter t.
    float t = frameIndex - float(frameIndex0);

    // Interpolate scene box.
    AABB sceneBox0 = scene.staticSceneBox;
    AABB sceneBox1 = scene.staticSceneBox;
    sceneBox0.grow(scene.frames[frameIndex0].sceneBox);
    sceneBox1.grow(scene.frames[frameIndex1].sceneBox);
    scene.sceneBox.mn = (1 - t) * sceneBox0.mn + t * sceneBox1.mn;
    scene.sceneBox.mx = (1 - t) * sceneBox0.mx + t * sceneBox1.mx;

    // Interpolate centroid box.
    AABB centroidBox0 = scene.staticCentroidBox;
    AABB centroidBox1 = scene.staticCentroidBox;
    centroidBox0.grow(scene.frames[frameIndex0].centroidBox);
    centroidBox1.grow(scene.frames[frameIndex1].centroidBox);
    scene.centroidBox.mn = (1 - t) * centroidBox0.mn + t * centroidBox1.mn;
    scene.centroidBox.mx = (1 - t) * centroidBox0.mx + t * centroidBox1.mx;

    // Time.
    float kernelsTime = 0.0f;

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel vertexKernel = module->getKernel("interpolateVertices");
    HipKernel normalKernel = module->getKernel("interpolateNormals");

    // Interpolate vertices.
    vertexKernel.setParams(
        scene.getNumberOfStaticVertices(),
        scene.getNumberOfDynamicVertices(),
        t,
        scene.frames[frameIndex0].vertices,
        scene.frames[frameIndex1].vertices,
        scene.vertices
    );
    kernelsTime += vertexKernel.launchTimed(scene.getNumberOfDynamicVertices());

    // Interpolate normals.
    normalKernel.setParams(
        scene.getNumberOfStaticVertices(),
        scene.getNumberOfDynamicVertices(),
        t,
        scene.frames[frameIndex0].normals,
        scene.frames[frameIndex1].normals,
        scene.normals
    );
    kernelsTime += normalKernel.launchTimed(scene.getNumberOfDynamicVertices());

    return kernelsTime;

}
