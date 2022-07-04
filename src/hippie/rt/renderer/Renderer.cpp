/**
 * \file	Renderer.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Renderer class source file.
 */

#include <QDir>
#include "RendererKernels.h"
#include "Renderer.h"
#include "util/Logger.h"
#include "environment/AppEnvironment.h"

PathQueue::PathQueue() : swapBuffers(false) {
}

PathQueue::~PathQueue() {
}

void PathQueue::swap() {
    swapBuffers = !swapBuffers;
}

void PathQueue::init(int size) {
    rays[0].resize(size);
    rays[1].resize(size);
}

RayBuffer & PathQueue::getInputRays() {
    return swapBuffers ? rays[1] : rays[0];
}

RayBuffer & PathQueue::getOutputRays() {
    return swapBuffers ? rays[0] : rays[1];
}

RayBuffer::MortonCodeMethod Renderer::stringToMortonCodeMethod(const QString & mortonCodeMethod) {
    if (mortonCodeMethod == "aila")
        return RayBuffer::AILA;
    else if (mortonCodeMethod == "paraboloid")
        return RayBuffer::PARABOLOID;
    else if (mortonCodeMethod == "octahedron")
        return RayBuffer::OCTAHEDRON;
    else if (mortonCodeMethod == "origin")
        return RayBuffer::ORIGIN;
    else if (mortonCodeMethod == "costa")
        return RayBuffer::COSTA;
    else if (mortonCodeMethod == "reise")
        return RayBuffer::REIS;
    else
        return RayBuffer::TWO_POINT;
}

Renderer::RayType Renderer::stringToRayType(const QString & rayType) {
    if (rayType == "primary")
        return PRIMARY_RAYS;
    else if (rayType == "shadow")
        return SHADOW_RAYS;
    else if (rayType == "ao")
        return AO_RAYS;
    else if (rayType == "path")
        return PATH_RAYS;
    else if (rayType == "pseudocolor")
        return PSEUDOCOLOR_RAYS;
    else
        return THERMAL_RAYS;
}

float Renderer::computeRayHits(RayBuffer & rays) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("countRayHits");

    // Reset hit counter
    module->getGlobal("rayHits").clear();

    // Set params.
    kernel.setParams(rays.getSize(), rays.getResultBuffer());

    // Launch.
    float time = kernel.launchTimed(rays.getSize(), Vec2i(HITS_BLOCK_THREADS, 1));

    // Ray hits
    numberOfHits = *(int*)module->getGlobal("rayHits").getPtr();

    return time;
}

float Renderer::initDecreases(int numberOfPixels) {

    // Resize decreases.
    decreases.resizeDiscard(numberOfPixels * sizeof(Vec3f));

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("initDecreases");

    // Set params.
    kernel.setParams(numberOfPixels, decreases);

    // Launch.
    return kernel.launchTimed(numberOfPixels);

}

float Renderer::interpolateColors(int numberOfPixels, Buffer & pixels, Buffer & framePixels) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("interpolateColors");

    // Set params.
    kernel.setParams(numberOfPixels, frameIndex, keyValue, whitePoint, framePixels, pixels);

    // Launch.
    return kernel.launchTimed(numberOfPixels);

}

float Renderer::primaryPass(Scene & scene, Camera & camera, Buffer & pixels) {
    float traceTime = tracePrimaryRays(camera);
    float reconstructTime = initDecreases(camera.getSize().x * camera.getSize().y);
    reconstructTime += reconstructSmooth(scene, primaryRays, pixels);
    numberOfPrimaryRays += camera.getSize().x * camera.getSize().y;
    primaryTraceTime += traceTime;
    return traceTime + reconstructTime;
}

float Renderer::shadowPass(Scene & scene, RayBuffer & inRays, Buffer & inPixels, Buffer & outPixels, bool replace) {
    float traceTime = 0.0f;
    float reconstructTime = 0.0f;
    int batchDiff = RENDERER_MAX_BATCH_SIZE / numberOfShadowSamples;
    int batchIndex = 0;
    int batchBegin;
    int batchEnd;
    do {
        batchBegin = batchIndex;
        batchEnd = qMin(batchBegin + batchDiff, inRays.getSize());
        traceTime += traceShadowRays(scene, inRays, batchBegin, batchEnd);
        reconstructTime += reconstructShadow(inRays, inPixels, outPixels, batchBegin, batchEnd, replace);
        batchIndex = batchEnd;
    } while (batchIndex != inRays.getSize());
    float rayHitsTime = computeRayHits(inRays);
    numberOfShadowRays += numberOfShadowSamples * numberOfHits;
    shadowTraceTime += traceTime;
    return traceTime + reconstructTime + rayHitsTime;
}

float Renderer::aoPass(Scene & scene, RayBuffer & inRays, Buffer & inPixels, Buffer & outPixels, bool replace) {
    float traceTime = 0.0f;
    float reconstructTime = 0.0f;
    int batchDiff = RENDERER_MAX_BATCH_SIZE / numberOfAOSamples;
    int batchIndex = 0;
    int batchBegin;
    int batchEnd;
    do {
        batchBegin = batchIndex;
        batchEnd = qMin(batchBegin + batchDiff, inRays.getSize());
        traceTime += traceAORays(scene, inRays, batchBegin, batchEnd);
        reconstructTime += reconstructAO(inRays, inPixels, outPixels, batchBegin, batchEnd, replace);
        batchIndex = batchEnd;
    } while (batchIndex != inRays.getSize());
    float rayHitsTime = computeRayHits(inRays);
    numberOfAORays += numberOfAOSamples * numberOfHits;
    aoTraceTime += traceTime;
    return traceTime + reconstructTime + rayHitsTime;
}

float Renderer::pathPass(Scene & scene, Buffer & pixels, RayBuffer & inRays, RayBuffer & outRays) {
    float traceTime = tracePathRays(scene, inRays, outRays);
    float reconstructTime = reconstructSmooth(scene, outRays, pixels);
    pathTraceTime += traceTime;
    numberOfPathRays += outRays.getSize();
    return traceTime + reconstructTime;
}

float Renderer::renderPrimary(Scene & scene, Camera & camera, Buffer & pixels) {
    return primaryPass(scene, camera, pixels);
}

float Renderer::renderShadow(Scene & scene, Camera & camera, Buffer & pixels) {
    auxPixels.clear();
    float time = primaryPass(scene, camera, auxPixels);
    time += shadowPass(scene, primaryRays, auxPixels, pixels, false);
    return time;
}

float Renderer::renderAO(Scene & scene, Camera & camera, Buffer & pixels) {
    auxPixels.clear();
    float time = primaryPass(scene, camera, auxPixels);
    time += aoPass(scene, primaryRays, auxPixels, pixels, false);
    return time;
}

float Renderer::renderPath(Scene & scene, Camera & camera, Buffer & pixels) {
    auxPixels.clear();
    float time = primaryPass(scene, camera, auxPixels);
    time += shadowPass(scene, primaryRays, auxPixels, pixels, false);
    pathQueue.init(primaryRays.getSize());
    for (bounce = 0; bounce < recursionDepth; ++bounce) {
        auxPixels.clear();
        if (bounce > 0)
            time += pathPass(
                scene,
                auxPixels,
                pathQueue.getInputRays(),
                pathQueue.getOutputRays()
            );
        else
            time += pathPass(
                scene,
                auxPixels,
                primaryRays,
                pathQueue.getOutputRays()
            );
        if (pathQueue.getOutputRays().getSize() == 0) break;
        time += shadowPass(scene, pathQueue.getOutputRays(), auxPixels, pixels, false);
        pathQueue.swap();
    }
    return time;
}

float Renderer::renderPseudocolor(Scene & scene, Camera & camera, Buffer & pixels) {
    float time = tracePrimaryRays(camera);
    time += reconstructPseudocolor(scene, pixels);
    return time;
}

float Renderer::renderThermal(Camera & camera, Buffer & pixels) {
    float time = tracePrimaryRays(camera);
    time += reconstructThermal(pixels);
    return time;
}

float Renderer::reconstructSmooth(Scene & scene, RayBuffer & rays, Buffer & pixels) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("reconstructSmooth");

    // Set scene data.
    memcpy(module->getGlobal("materials").getMutablePtr(), scene.getMaterialBuffer().getPtr(),
        sizeof(Material) * qMin(scene.getNumberOfMaterials(), MAX_MATERIALS));
    if (scene.hasTextures()) {
        TextureAtlas & textureAtlas = scene.getTextureAtlas();
        TextureItem * diffuseTexturesItems = (TextureItem*)module->getGlobal("diffuseTextureItems").getMutablePtr();
        for (int i = 0; i < qMin(textureAtlas.getNumberOfTextures(), MAX_MATERIALS); ++i)
            diffuseTexturesItems[i] = textureAtlas.getTextureItems()[i];
    }

    // Set params.	
    kernel.setParams(
        rays.getSize(),
        numberOfPrimarySamples,
        (unsigned long long)scene.getTextureAtlas().getTextureObject(),
        (unsigned long long)scene.getEnvironmentMap().getTextureObject(),
        scene.getMatIndexBuffer(),
        scene.getTriangleBuffer(),
        scene.getNormalBuffer(),
        scene.getTexCoordBuffer(),
        rays.getRayBuffer(),
        rays.getResultBuffer(),
        scene.getLight(),
        rays.getSlotToIndexBuffer(),
        pixels,
        decreases
    );

    // Launch.
    return kernel.launchTimed(rays.getSize());

}

float Renderer::reconstructPseudocolor(Scene & scene, Buffer & pixels) {

    // Colorize scene.
    tracer.getBVH()->colorizeScene(nodeSizeThreshold);

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("reconstructPseudocolor");

    // Set params.
    kernel.setParams(
        primaryRays.getSize(),
        numberOfPrimarySamples,
        scene.getMatIndexBuffer(),
        scene.getTriangleBuffer(),
        scene.getNormalBuffer(),
        scene.getPseudocolorBuffer(),
        primaryRays.getRayBuffer(),
        primaryRays.getResultBuffer(),
        scene.getLight(),
        primaryRays.getSlotToIndexBuffer(),
        pixels
    );

    // Launch.
    return kernel.launchTimed(primaryRays.getSize());

}

float Renderer::reconstructThermal(Buffer & pixels) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("reconstructThermal");

    // Set params.
    kernel.setParams(
        primaryRays.getSize(),
        numberOfPrimarySamples,
        thermalThreshold,
        primaryRays.getSlotToIndexBuffer(),
        primaryRays.getStatBuffer(),
        pixels
    );

    // Launch.
    return kernel.launchTimed(primaryRays.getSize());

}

float Renderer::reconstructShadow(RayBuffer & inRays, Buffer & inPixels, Buffer & outPixels, int batchBegin, int batchEnd, bool replace) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("reconstructShadow");

    // Set params.
    kernel.setParams(
        batchBegin,
        batchEnd - batchBegin,
        numberOfShadowSamples,
        replace,
        shadowRays.getResultBuffer(),
        inRays.getSlotToIndexBuffer(),
        shadowRays.getIndexToSlotBuffer(),
        inPixels,
        outPixels
    );

    // Launch.
    return kernel.launchTimed(batchEnd - batchBegin);

}

float Renderer::reconstructAO(RayBuffer & inRays, Buffer & inPixels, Buffer & outPixels, int batchBegin, int batchEnd, bool replace) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("reconstructAO");

    // Set params.
    kernel.setParams(
        batchBegin,
        batchEnd - batchBegin,
        numberOfAOSamples,
        replace,
        aoRays.getResultBuffer(),
        inRays.getSlotToIndexBuffer(),
        aoRays.getIndexToSlotBuffer(),
        inPixels,
        outPixels
    );

    // Launch.
    return kernel.launchTimed(batchEnd - batchBegin);

}

float Renderer::tracePrimaryRays(Camera & camera) {
    float time = raygen.primary(primaryRays, camera, pass + numberOfPrimarySamples * (frameIndex - 1));
    if (rayType == THERMAL_RAYS) {
        time += tracer.traceStats(primaryRays);
    }
    else {
        time += tracer.trace(primaryRays);
    }
    return time;
}

float Renderer::traceShadowRays(Scene & scene, RayBuffer & inRays, int batchBegin, int batchEnd) {
    float time = 0.0f;
    time += raygen.shadow(shadowRays, inRays, batchBegin, batchEnd, numberOfShadowSamples, scene.getLight(), shadowRadius);
    if (sortShadowRays) time += tracer.traceSort(shadowRays);
    else time += tracer.trace(shadowRays);
    return time;
}

float Renderer::traceAORays(Scene & scene, RayBuffer & inRays, int batchBegin, int batchEnd) {
    float time = raygen.ao(aoRays, inRays, scene, batchBegin, batchEnd, numberOfAOSamples, aoRadius);
    if (sortAORays) time += tracer.traceSort(aoRays);
    else time += tracer.trace(aoRays);
    return time;
}

float Renderer::tracePathRays(Scene & scene, RayBuffer & inRays, RayBuffer & outRays) {
    float time = raygen.path(outRays, inRays, decreases, scene);
    if (sortPathRays) {
#if SORT_LOG
        float sortTime;
        float traceSortTime;
        float traceTime = tracer.trace(outRays);
        time += tracer.traceSort(outRays, sortTime, traceSortTime);
        avgRayCounts[bounce] += outRays.getSize() / getNumberOfPrimarySamples();
        sortTimes[bounce] += sortTime;
        traceSortTimes[bounce] += traceSortTime;
        traceTimes[bounce] += tracer.trace(outRays);
#else
        time += tracer.traceSort(outRays);
#endif
    }
    else time += tracer.trace(outRays);
    return time;
}

Renderer::Renderer() :
    rayType(PRIMARY_RAYS),
    keyValue(1.0f),
    whitePoint(1.0f),
    aoRadius(1.0f),
    shadowRadius(1.0f),
    numberOfPrimarySamples(1),
    numberOfAOSamples(4),
    numberOfShadowSamples(4),
    recursionDepth(3),
    frameIndex(1),
    nodeSizeThreshold(5000),
    thermalThreshold(200)
{
    compiler.setSourceFile("../src/hippie/rt/renderer/RendererKernels.cu");

    QString _rayType;
    Environment::getInstance()->getStringValue("Renderer.rayType", _rayType);
    setRayType(stringToRayType(_rayType));

    float _keyValue;
    Environment::getInstance()->getFloatValue("Renderer.keyValue", _keyValue);
    setKeyValue(_keyValue);
    float _whitePoint;
    Environment::getInstance()->getFloatValue("Renderer.whitePoint", _whitePoint);
    setWhitePoint(_whitePoint);
    int _numberOfAOSamples;
    Environment::getInstance()->getIntValue("Renderer.numberOfAOSamples", _numberOfAOSamples);
    setNumberOfAOSamples(_numberOfAOSamples);
    int _numberOfPrimarySamples;
    Environment::getInstance()->getIntValue("Renderer.numberOfPrimarySamples", _numberOfPrimarySamples);
    setNumberOfPrimarySamples(_numberOfPrimarySamples);
    int _numberOfShadowSamples;
    Environment::getInstance()->getIntValue("Renderer.numberOfShadowSamples", _numberOfShadowSamples);
    setNumberOfShadowSamples(_numberOfShadowSamples);
    float _aoRadius;
    Environment::getInstance()->getFloatValue("Renderer.aoRadius", _aoRadius);
    setAORadius(_aoRadius);
    float _shadowRadius;
    Environment::getInstance()->getFloatValue("Renderer.shadowRadius", _shadowRadius);
    setShadowRadius(_shadowRadius);
    int _nodeSizeThreshold;
    Environment::getInstance()->getIntValue("Renderer.nodeSizeThreshold", _nodeSizeThreshold);
    setNodeSizeThreshold(_nodeSizeThreshold);
    int _thermalThreshold;
    Environment::getInstance()->getIntValue("Renderer.thermalThreshold", _thermalThreshold);
    setThermalThreshold(_thermalThreshold);
    int _recursionDepth;
    Environment::getInstance()->getIntValue("Renderer.recursionDepth", _recursionDepth);
    setRecursionDepth(_recursionDepth);

    Environment::getInstance()->getBoolValue("Renderer.sortShadowRays", sortShadowRays);
    Environment::getInstance()->getBoolValue("Renderer.sortAORays", sortAORays);
    Environment::getInstance()->getBoolValue("Renderer.sortPathRays", sortPathRays);
    float _shadowRayLength;
    Environment::getInstance()->getFloatValue("Renderer.shadowRayLength", _shadowRayLength);
    setShadowRayLength(_shadowRayLength);
    float _aoRayLength;
    Environment::getInstance()->getFloatValue("Renderer.aoRayLength", _aoRayLength);
    setAORayLength(_aoRayLength);
    float _pathRayLength;
    Environment::getInstance()->getFloatValue("Renderer.pathRayLength", _pathRayLength);
    setPathRayLength(_pathRayLength);
    int _shadowMortonCodeBits;
    Environment::getInstance()->getIntValue("Renderer.shadowMortonCodeBits", _shadowMortonCodeBits);
    setShadowMortonCodeBits(_shadowMortonCodeBits);
    int _aoMortonCodeBits;
    Environment::getInstance()->getIntValue("Renderer.aoMortonCodeBits", _aoMortonCodeBits);
    setAOMortonCodeBits(_aoMortonCodeBits);
    int _pathMortonCodeBits;
    Environment::getInstance()->getIntValue("Renderer.pathMortonCodeBits", _pathMortonCodeBits);
    setPathMortonCodeBits(_pathMortonCodeBits);
    QString _shadowMortonCodeMethod;
    Environment::getInstance()->getStringValue("Renderer.shadowMortonCodeMethod", _shadowMortonCodeMethod);
    setShadowMortonCodeMethod(stringToMortonCodeMethod(_shadowMortonCodeMethod));
    QString _aoMortonCodeMethod;
    Environment::getInstance()->getStringValue("Renderer.aoMortonCodeMethod", _aoMortonCodeMethod);
    setAOMortonCodeMethod(stringToMortonCodeMethod(_aoMortonCodeMethod));
    QString _pathMortonCodeMethod;
    Environment::getInstance()->getStringValue("Renderer.pathMortonCodeMethod", _pathMortonCodeMethod);
    setPathMortonCodeMethod(stringToMortonCodeMethod(_pathMortonCodeMethod));
}

Renderer::~Renderer() {
}

Renderer::RayType Renderer::getRayType() {
    return rayType;
}

void Renderer::setKeyValue(float keyValue) {
    if (keyValue <= 0 || keyValue > RENDERER_MAX_KEY_VALUE) {
        logger(LOG_WARN) << "WARN <Renderer> KeyValue must be in range (0," << RENDERER_MAX_KEY_VALUE << "].\n";
    }
    else {
        this->keyValue = keyValue;
        resetFrameIndex();
    }
}

float Renderer::getKeyValue() {
    return keyValue;
}

void Renderer::setWhitePoint(float whitePoint) {
    if (whitePoint <= 0 || whitePoint > RENDERER_MAX_WHITE_POINT) {
        logger(LOG_WARN) << "WARN <Renderer> WhitePoint must be in range (0," << RENDERER_MAX_WHITE_POINT << "].\n";
    }
    else {
        this->whitePoint = whitePoint;
        resetFrameIndex();
    }
}

float Renderer::getWhitePoint() {
    return whitePoint;
}

void Renderer::setRayType(RayType rayType) {
    resetFrameIndex();
    this->rayType = rayType;
}

float Renderer::getAORadius() {
    return aoRadius;
}

void Renderer::setAORadius(float aoRadius) {
    if (aoRadius <= 0 || aoRadius > RENDERER_MAX_RADIUS) {
        logger(LOG_WARN) << "WARN <Renderer> AO radius must be in range (0," << RENDERER_MAX_RADIUS << "].\n";
    }
    else {
        this->aoRadius = aoRadius;
        resetFrameIndex();
    }
}

float Renderer::getShadowRadius() {
    return shadowRadius;
}
void Renderer::setShadowRadius(float shadowRadius) {
    if (shadowRadius <= 0 || shadowRadius > RENDERER_MAX_RADIUS) {
        logger(LOG_WARN) << "WARN <Renderer> Shadow radius must be in range (0," << RENDERER_MAX_RADIUS << "].\n";
    }
    else {
        this->shadowRadius = shadowRadius;
        resetFrameIndex();
    }
}

int Renderer::getNumberOfPrimarySamples(void) {
    return numberOfPrimarySamples;
}

void Renderer::setNumberOfPrimarySamples(int numberOfPrimarySamples) {
    if (numberOfPrimarySamples <= 0 || numberOfPrimarySamples > RENDERER_MAX_SAMPLES) {
        logger(LOG_WARN) << "WARN <Renderer> Number of primary samples must be in range (0," << RENDERER_MAX_SAMPLES << "].\n";
    }
    else {
        this->numberOfPrimarySamples = numberOfPrimarySamples;
        resetFrameIndex();
    }
}

int Renderer::getNumberOfAOSamples() {
    return numberOfAOSamples;
}

void Renderer::setNumberOfAOSamples(int numberOfAOSamples) {
    if (numberOfAOSamples <= 0 || numberOfAOSamples > RENDERER_MAX_SAMPLES) {
        logger(LOG_WARN) << "WARN <Renderer> Number of AO samples must be in range (0," << RENDERER_MAX_SAMPLES << "].\n";
    }
    else {
        this->numberOfAOSamples = numberOfAOSamples;
        resetFrameIndex();
    }
}

int Renderer::getNumberOfShadowSamples() {
    return numberOfShadowSamples;
}

void Renderer::setNumberOfShadowSamples(int numberOfShadowSamples) {
    if (numberOfShadowSamples <= 0 || numberOfShadowSamples > RENDERER_MAX_SAMPLES) {
        logger(LOG_WARN) << "WARN <Renderer> Number of shadow samples must be in range (0," << RENDERER_MAX_SAMPLES << "].\n";
    }
    else {
        this->numberOfShadowSamples = numberOfShadowSamples;
        resetFrameIndex();
    }
}

int Renderer::getRecursionDepth() {
    return recursionDepth;
}

void Renderer::setRecursionDepth(int recursionDepth) {
    if (recursionDepth < 0 || recursionDepth > RENDERER_MAX_RECURSION_DEPTH) {
        logger(LOG_WARN) << "WARN <Renderer> Recursion depth must be in range (0," << RENDERER_MAX_RECURSION_DEPTH << "].\n";
    }
    else {
        this->recursionDepth = recursionDepth;
        resetFrameIndex();
    }
}

int Renderer::getNodeSizeThreshold() {
    return nodeSizeThreshold;
}

void Renderer::setNodeSizeThreshold(int nodeSizeThreshold) {
    if (nodeSizeThreshold <= 0) {
        logger(LOG_WARN) << "WARN <Renderer> Node size threshold must be positive.\n";
    }
    else {
        this->nodeSizeThreshold = nodeSizeThreshold;
        resetFrameIndex();
    }
}

int Renderer::getThermalThreshold() {
    return thermalThreshold;
}

void Renderer::setThermalThreshold(int thermalThreshold) {
    if (thermalThreshold <= 0) {
        logger(LOG_WARN) << "WARN <Renderer> Thermal threshold must be positive.\n";
    }
    else {
        this->thermalThreshold = thermalThreshold;
        resetFrameIndex();
    }
}

bool Renderer::getRussianRoulette() {
    return raygen.getRussianRoulette();
}

void Renderer::setRussianRoulette(bool russianRoulette) {
    raygen.setRussianRoulette(russianRoulette);
}

bool Renderer::getSortShadowRays() {
    return sortShadowRays;
}

void Renderer::setSortShadowRays(bool sortShadowRays) {
    this->sortShadowRays = sortShadowRays;
}

bool Renderer::getSortAORays() {
    return sortAORays;
}

void Renderer::setSortAORays(bool sortAORays) {
    this->sortAORays = sortAORays;
}

bool Renderer::getSortPathRays() {
    return sortPathRays;
}

void Renderer::setSortPathRays(bool sortPathRays) {
    this->sortPathRays = sortPathRays;
}

float Renderer::getShadowRayLength(void) {
    return shadowRays.getRayLength();
}

void Renderer::setShadowRayLength(float shadowRayLength) {
    shadowRays.setRayLength(shadowRayLength);
}

float Renderer::getAORayLength(void) {
    return aoRays.getRayLength();
}

void Renderer::setAORayLength(float aoRayLength) {
    if (getAORadius() > aoRayLength) logger(LOG_WARN) << "WARN <Renderer> AO Ray length must be less or equal to ao radius.\n";
    else aoRays.setRayLength(aoRayLength);
}

float Renderer::getPathRayLength(void) {
    Q_ASSERT(pathQueue.getInputRays().getRayLength() == pathQueue.getOutputRays().getRayLength());
    return pathQueue.getInputRays().getRayLength();
}

void Renderer::setPathRayLength(float pathRayLength) {
    pathQueue.getInputRays().setRayLength(pathRayLength);
    pathQueue.getOutputRays().setRayLength(pathRayLength);
}

int Renderer::getShadowMortonCodeBits() {
    return shadowRays.getMortonCodeBits();
}

void Renderer::setShadowMortonCodeBits(int shadowMortonCodeBits) {
    shadowRays.setMortonCodeBits(shadowMortonCodeBits);
}

int Renderer::getAOMortonCodeBits() {
    return aoRays.getMortonCodeBits();
}
void Renderer::setAOMortonCodeBits(int aoMortonCodeBits) {
    aoRays.setMortonCodeBits(aoMortonCodeBits);
}

int Renderer::getPathMortonCodeBits() {
    Q_ASSERT(pathQueue.getInputRays().getMortonCodeBits() == pathQueue.getOutputRays().getMortonCodeBits());
    return pathQueue.getInputRays().getMortonCodeBits();
}

void Renderer::setPathMortonCodeBits(int pathMortonCodeBits) {
    pathQueue.getInputRays().setMortonCodeBits(pathMortonCodeBits);
    pathQueue.getOutputRays().setMortonCodeBits(pathMortonCodeBits);
}

RayBuffer::MortonCodeMethod Renderer::getShadowMortonCodeMethod() {
    return shadowRays.getMortonCodeMethod();
}

void Renderer::setShadowMortonCodeMethod(RayBuffer::MortonCodeMethod shadowMortonCodeMethod) {
    shadowRays.setMortonCodeMethod(shadowMortonCodeMethod);
}

RayBuffer::MortonCodeMethod Renderer::getAOMortonCodeMethod() {
    return aoRays.getMortonCodeMethod();
}

void Renderer::setAOMortonCodeMethod(RayBuffer::MortonCodeMethod aoMortonCodeMethod) {
    aoRays.setMortonCodeMethod(aoMortonCodeMethod);
}

RayBuffer::MortonCodeMethod Renderer::getPathMortonCodeMethod() {
    Q_ASSERT(pathQueue.getInputRays().getMortonCodeMethod() == pathQueue.getOutputRays().getMortonCodeMethod());
    return pathQueue.getInputRays().getMortonCodeMethod();
}

void Renderer::setPathMortonCodeMethod(RayBuffer::MortonCodeMethod pathMortonCodeMethod) {
    pathQueue.getInputRays().setMortonCodeMethod(pathMortonCodeMethod);
    pathQueue.getOutputRays().setMortonCodeMethod(pathMortonCodeMethod);
}

float Renderer::render(Scene & scene, HipBVH & bvh, Camera & camera, Buffer & pixels, Buffer & framePixels) {

    // Elapsed time.
    float time = 0.0f;

    // Set BVH.
    tracer.setBVH(&bvh);

    // Resize pixel buffers.
    pixels.resizeDiscard(camera.getSize().x * camera.getSize().y * sizeof(Vec4f));
    framePixels.resizeDiscard(pixels.getSize());
    auxPixels.resizeDiscard(pixels.getSize());

    // Clear pixel buffer.
    framePixels.clear();
    if (frameIndex == 1)
        pixels.clear();

    // Clear trace times.
    primaryTraceTime = 0.0f;
    shadowTraceTime = 0.0f;
    aoTraceTime = 0.0f;
    pathTraceTime = 0.0f;

    // Clear number of rays.
    numberOfPrimaryRays = 0;
    numberOfShadowRays = 0;
    numberOfAORays = 0;
    numberOfPathRays = 0;

    // Clear sort counts.
    for (int i = 0; i < RENDERER_MAX_RECURSION_DEPTH; ++i) {
        avgRayCounts[i] = 0;
        sortTimes[i] = 0.0f;
        traceSortTimes[i] = 0.0f;
        traceTimes[i] = 0.0f;
    }

    // Init seeds.
    if (rayType == PATH_RAYS || rayType == SHADOW_RAYS || rayType == AO_RAYS)
        time += raygen.initSeeds(camera.getSize().x * camera.getSize().y, frameIndex);

    // For-each primary ray.
    for (pass = 0; pass < numberOfPrimarySamples; ++pass) {
        if (rayType == PRIMARY_RAYS)
            time += renderPrimary(scene, camera, framePixels);
        else if (rayType == SHADOW_RAYS)
            time += renderShadow(scene, camera, framePixels);
        else if (rayType == AO_RAYS)
            time += renderAO(scene, camera, framePixels);
        else if (rayType == PATH_RAYS)
            time += renderPath(scene, camera, framePixels);
        else if (rayType == PSEUDOCOLOR_RAYS)
            time += renderPseudocolor(scene, camera, framePixels);
        else if (rayType == THERMAL_RAYS)
            time += renderThermal(camera, framePixels);
    }

    // Log sort counts.
    QString mode;
    Environment::getInstance()->getStringValue("Application.mode", mode);
    if (mode != "interactive") {
        QString output;
        Environment::getInstance()->getStringValue("Benchmark.output", output);
        QString logDir = "benchmark/" + output;
        if (!QDir(logDir).exists())
            QDir().mkpath(logDir);
        Logger sort(logDir + "/sort_" + output + ".log");
        for (int i = 0; i < RENDERER_MAX_RECURSION_DEPTH; ++i)
            sort << avgRayCounts[i] << " " << sortTimes[i] << " " << traceSortTimes[i] << " " << traceTimes[i] << "\n";
    }

    // Interpolate colors.
    time += interpolateColors(camera.getSize().x * camera.getSize().y, pixels, framePixels);

    // Inc. frame number.
    ++frameIndex;

    return time;

}

void Renderer::resetFrameIndex() {
    frameIndex = 1;
}

unsigned long long Renderer::getNumberOfPrimaryRays() {
    return numberOfPrimaryRays;
}

unsigned long long Renderer::getNumberOfShadowRays() {
    return numberOfShadowRays;
}

unsigned long long Renderer::getNumberOfAORays() {
    return numberOfAORays;
}

unsigned long long Renderer::getNumberOfPathRays() {
    return numberOfPathRays;
}

unsigned long long Renderer::getNumberOfRays() {
    return numberOfPrimaryRays + numberOfShadowRays + numberOfAORays + numberOfPathRays;
}

float Renderer::getPrimaryTraceTime() {
    return primaryTraceTime;
}

float Renderer::getShadowTraceTime() {
    return shadowTraceTime;
}

float Renderer::getAOTraceTime() {
    return aoTraceTime;
}

float Renderer::getPathTraceTime() {
    return pathTraceTime;
}

float Renderer::getTraceTime() {
    return primaryTraceTime + shadowTraceTime + aoTraceTime + pathTraceTime;
}

float Renderer::getPrimaryTracePerformance() {
    return primaryTraceTime == 0.0f ? 0.0f : numberOfPrimaryRays * 1.0e-6f / primaryTraceTime;
}

float Renderer::getShadowTracePerformance() {
    return shadowTraceTime == 0.0f ? 0.0f : numberOfShadowRays * 1.0e-6f / shadowTraceTime;
}

float Renderer::getAOTracePerformance() {
    return aoTraceTime == 0.0f ? 0.0f : numberOfAORays * 1.0e-6f / aoTraceTime;
}

float Renderer::getPathTracePerformance() {
    return pathTraceTime == 0.0f ? 0.0f : numberOfPathRays * 1.0e-6f / pathTraceTime;
}

float Renderer::getTracePerformance() {
    return getTraceTime() == 0.0f ? 0.0f : getNumberOfRays() * 1.0e-6f / getTraceTime();
}
