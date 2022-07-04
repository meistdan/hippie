/**
  * \file	RayGen.cpp
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	RayGen class source file.
  */

#include "RayGen.h"
#include "RayGenKernels.h"
#include "rt/renderer/Renderer.h"
#include "util/Logger.h"

float RayGen::initSeeds(int numberOfPixels, int frameIndex) {

    // Resize seeds.
    seeds.resizeDiscard(sizeof(unsigned int) * numberOfPixels);

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("initSeeds");

    // Set params.
    kernel.setParams(numberOfPixels, frameIndex, seeds);

    // Launch.
    return kernel.launchTimed(numberOfPixels);

}

RayGen::RayGen() {
    compiler.setSourceFile("../src/hippie/rt/ray/RayGenKernels.cu");
    Environment::getInstance()->getBoolValue("Renderer.russianRoulette", russianRoulette);
}

float RayGen::primary(RayBuffer & orays, Camera & camera, int sampleIndex) {

    // Closest hit.
    pixelTable.setSize(camera.getSize());
    orays.resize(camera.getSize().x * camera.getSize().y);
    orays.setClosestHit(true);
    orays.getSlotToIndexBuffer() = pixelTable.getIndexToPixel();
    orays.getIndexToSlotBuffer() = pixelTable.getPixelToIndex();

    // Compile kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("generatePrimaryRays");
    Mat4f screenToWorld = inverse(camera.getProjectionViewMatrix());

    // Set parameters.
    kernel.setParams(
        sampleIndex,
        camera.getPosition(),
        screenToWorld,
        camera.getSize(),
        camera.getFar(),
        orays.getSlotToIndexBuffer(),
        orays.getRayBuffer()
    );

    // Launch.
    return kernel.launchTimed(camera.getSize().x * camera.getSize().y);

}

float RayGen::shadow(RayBuffer & orays, RayBuffer & irays, int batchBegin, int batchEnd, int numberOfSamples, const Vec3f & light, float lightRadius) {

    // Closest hit.
    orays.resize((batchEnd - batchBegin) * numberOfSamples);
    orays.setClosestHit(false);

    // Compile kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("generateShadowRays");

    // Set parameters.
    kernel.setParams(
        batchBegin,
        batchEnd - batchBegin,
        numberOfSamples,
        seeds,
        lightRadius,
        light,
        irays.getRayBuffer(),
        orays.getRayBuffer(),
        irays.getResultBuffer(),
        orays.getSlotToIndexBuffer(),
        orays.getIndexToSlotBuffer()
    );

    // Launch.
    return kernel.launchTimed(batchEnd - batchBegin);

}

float RayGen::ao(RayBuffer & orays, RayBuffer & irays, Scene & scene, int batchBegin, int batchEnd, int numberOfSamples, float maxDist) {

    // Closest hit.
    orays.resize((batchEnd - batchBegin) * numberOfSamples);
    orays.setClosestHit(false);

    // Compile kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("generateAORays");

    // Set parameters.
    kernel.setParams(
        batchBegin,
        batchEnd - batchBegin,
        numberOfSamples,
        seeds,
        maxDist,
        irays.getRayBuffer(),
        orays.getRayBuffer(),
        irays.getResultBuffer(),
        orays.getSlotToIndexBuffer(),
        orays.getIndexToSlotBuffer(),
        scene.getTriangleBuffer(),
        scene.getVertexBuffer()
    );

    // Launch.
    return kernel.launchTimed(batchEnd - batchBegin);

}

float RayGen::path(RayBuffer & orays, RayBuffer & irays, Buffer & decreases, Scene & scene) {

    // Closest hit.
    orays.resize(irays.getSize());
    orays.setClosestHit(true);

    // Compile kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("generatePathRays");

    // Set scene data.
    memcpy(module->getGlobal("materials").getMutablePtr(), scene.getMaterialBuffer().getPtr(),
        sizeof(Material) * qMin(scene.getNumberOfMaterials(), MAX_MATERIALS));
    if (scene.hasTextures()) {
        TextureAtlas & textureAtlas = scene.getTextureAtlas();
        TextureItem * diffuseTexturesItems = (TextureItem*)module->getGlobal("diffuseTextureItems").getMutablePtr();
        for (int i = 0; i < qMin(textureAtlas.getNumberOfTextures(), MAX_MATERIALS); ++i)
            diffuseTexturesItems[i] = textureAtlas.getTextureItems()[i];
    }

    // Number of output rays buffer.
    Buffer numberOfRaysBuffer;
    numberOfRaysBuffer.resizeDiscard(sizeof(int));
    numberOfRaysBuffer.clear();

    // Set parameters.
    kernel.setParams(
        russianRoulette,
        irays.getSize(),
        (unsigned long long)scene.getTextureAtlas().getTextureObject(),
        seeds,
        numberOfRaysBuffer,
        irays.getSlotToIndexBuffer(),
        orays.getSlotToIndexBuffer(),
        scene.getMatIndexBuffer(),
        scene.getTriangleBuffer(),
        scene.getNormalBuffer(),
        scene.getTexCoordBuffer(),
        irays.getRayBuffer(),
        orays.getRayBuffer(),
        irays.getResultBuffer(),
        decreases
    );

    // Launch.
    float time = kernel.launchTimed(irays.getSize());

    // Resize ouput rays.
    int numberOfRays = *(int*)numberOfRaysBuffer.getPtr();
    orays.resize(numberOfRays);

    return time;

}

bool RayGen::getRussianRoulette() {
    return russianRoulette;
}

void RayGen::setRussianRoulette(bool russianRoulette) {
    this->russianRoulette = russianRoulette;
}
