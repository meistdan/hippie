/**
 * \file	DynamicScene.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	DynamicScene class source file.
 */

#include "environment/AppEnvironment.h"
#include "gpu/HipModule.h"
#include "DynamicScene.h"

DynamicScene::DynamicScene() :
    numberOfStaticVertices(0),
    loop(true),
    frameRate(30.0f),
    time(0.0f),
    frameIndex(0)
{
    Environment::getInstance()->getBoolValue("Animation.loop", loop);
    float _frameRate;
    Environment::getInstance()->getFloatValue("Animation.frameRate", _frameRate);
    setFrameRate(_frameRate);
}

DynamicScene::~DynamicScene() {
}

float DynamicScene::getFrameRate() {
    return frameRate;
}

void DynamicScene::setFrameRate(float frameRate) {
    if (frameRate > 0.0f)
        this->frameRate = frameRate;
}

bool DynamicScene::isLooped() {
    return loop;
}

void DynamicScene::setLoop(bool loop) {
    this->loop = loop;
}

float DynamicScene::getTime() {
    return time;
}

void DynamicScene::setTime(float time) {
    if (time >= 0.0f && time < 1.0f)
        this->time = time;
}

void DynamicScene::resetTime() {
    time = 0.0f;
}

int DynamicScene::getNumberOfFrames() {
    return frames.size();
}

int DynamicScene::getNumberOfStaticVertices() {
    return numberOfStaticVertices;
}

int DynamicScene::getNumberOfDynamicVertices() {
    return numberOfVertices - numberOfStaticVertices;
}

void DynamicScene::increaseFrameRate() {
    frameRate = qMin(frameRate + DELTA_FRAME_RATE, MAX_FRAME_RATE);
}

void DynamicScene::decreaseFrameRate() {
    frameRate = qMax(frameRate - DELTA_FRAME_RATE, 1.0f);
}

int DynamicScene::getFrameIndex() {
    return frameIndex;
}

void DynamicScene::setFrameIndex(int index) {
    if (index >= 0 && index < frames.size()) {
        frameIndex = index;
        sceneBox = staticSceneBox;
        sceneBox.grow(frames[frameIndex].sceneBox);
        centroidBox = staticCentroidBox;
        centroidBox.grow(frames[frameIndex].centroidBox);
        int offset = numberOfStaticVertices * sizeof(Vec3f);
        vertices.setRange(offset, frames[index].vertices, 0, frames[index].vertices.getSize());
        normals.setRange(offset, frames[index].normals, 0, frames[index].normals.getSize());
        int reqBytes = 2 * sizeof(Vec3f) * getNumberOfDynamicVertices();
    }
}

void DynamicScene::setNextFrame() {
    if (loop) frameIndex = (frameIndex + 1) % getNumberOfFrames();
    else frameIndex = qMin(frameIndex + 1, getNumberOfFrames() - 1);
    setFrameIndex(frameIndex);
}

void DynamicScene::setPreviousFrame() {
    if (loop) frameIndex = frameIndex - 1 < 0 ? getNumberOfFrames() - 1 : frameIndex - 1;
    else frameIndex = qMax(0, frameIndex - 1);
    setFrameIndex(frameIndex);
}

bool DynamicScene::isDynamic() const {
    return true;
}
