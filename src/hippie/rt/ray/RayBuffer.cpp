/**
 * \file	RayBuffer.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	RayBuffer class source file.
 */

#include "RayBuffer.h"
#include "RayBufferKernels.h"
#include "gpu/HipCompiler.h"
#include "radix_sort/RadixSort.h"
#include "util/AABB.h"
#include "util/Logger.h"
#include "util/Random.h"

void RayBuffer::randomSort() {
    Ray* rays = (Ray*)getRayBuffer().getMutablePtr();
    int* indexToSlot = (int*)getIndexToSlotBuffer().getMutablePtr();
    int* slotToIndex = (int*)getSlotToIndexBuffer().getMutablePtr();
    Random random;
    for (int slot = 0; slot < getSize(); slot++) {
        int slot2 = random.randomInt(getSize() - slot - 1) + slot;
        int index = slotToIndex[slot];
        int index2 = slotToIndex[slot2];
        std::swap(rays[slot], rays[slot2]);
        std::swap(slotToIndex[slot], slotToIndex[slot2]);
        std::swap(indexToSlot[index], indexToSlot[index2]);
    }
}

float RayBuffer::computeMortonCodes(Buffer & mortonCodes) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel;

    QString bits = "64";
    switch (mortonCodeMethod) {
    case AILA:
        kernel = module->getKernel(QString("computeMortonCodesAila%1").arg(bits));
        break;
    case PARABOLOID:
        kernel = module->getKernel(QString("computeMortonCodesParaboloid%1").arg(bits));
        break;
    case OCTAHEDRON:
        kernel = module->getKernel(QString("computeMortonCodesOctahedron%1").arg(bits));
        break;
    case ORIGIN:
        kernel = module->getKernel(QString("computeMortonCodesOrigin%1").arg(bits));
        break;
    case COSTA:
        kernel = module->getKernel(QString("computeMortonCodesCosta%1").arg(bits));
        break;
    case REIS:
        kernel = module->getKernel(QString("computeMortonCodesReis%1").arg(bits));
        break;
    default: {
        kernel = module->getKernel(QString("computeMortonCodesTwoPoint%1").arg(bits));
    }
    }

    // Copy rays bounding box to constant memory.
    AABB box;
    box.grow(Vec3f(0.0f));
    box.grow(Vec3f(1.0f));
    *((AABB*)module->getGlobal("raysBoundingBox").getMutablePtr()) = box;

    // Set params.
    kernel.setParams(
        getSize(),
        mortonCodeBits,
        rayLength,
        getRayBuffer(),
        getResultBuffer(),
        mortonCodes,
        getIndexBuffer()
    );

    // Launch.
    float time = kernel.launchTimed(getSize());

    // Kernel time.
    return time;

}

float RayBuffer::reorderRays(Buffer & oldRayBuffer, Buffer & oldIndexToPixel) {

    // Kernel.
    HipModule * module = compiler.compile();
    HipKernel kernel = module->getKernel("reorderRays");

    // Set params.
    kernel.setParams(
        getSize(),
        getIndexBuffer(),
        oldRayBuffer,
        getRayBuffer(),
        oldIndexToPixel,
        getSlotToIndexBuffer(),
        getIndexToSlotBuffer()
    );

    // Launch.
    float time = kernel.launchTimed(getSize());

    // Kernel time.
    return time;

}

float RayBuffer::mortonSort() {

    // Allocate temporary buffers.
    indices[1].resizeDiscard(capacity * sizeof(int));

    // Compute Moron codes.
    mortonCodes[0].resizeDiscard(sizeof(unsigned long long) * capacity);
    mortonCodes[1].resizeDiscard(sizeof(unsigned long long) * capacity);
    float mortonCodesTime = computeMortonCodes(mortonCodes[0]);

    // Sort keys.
    float sortTime = 0.0f;
    bool sortSwap = false;
    RadixSort().sort(mortonCodes[0], mortonCodes[1], indices[0], indices[1], spine, sortSwap, getSize(), 0, mortonCodeBits);
    if (sortSwap) indices[0] = indices[1];

    // Total time.
    return mortonCodesTime + sortTime;

}

RayBuffer::RayBuffer() : size(0), capacity(0), closestHit(true), mortonCodeBits(30), rayLength(0.25f), mortonCodeMethod(AILA) {
    compiler.setSourceFile("../src/hippie/rt/ray/RayBufferKernels.cu");
}

int RayBuffer::getSize() const {
    return size;
}

void RayBuffer::resize(int size) {
    Q_ASSERT(size >= 0);
    if (capacity < size) {
        capacity = size;
        rays.resize(size * sizeof(Ray));
        results.resize(size * sizeof(RayResult));
        stats.resize(size * sizeof(Vec2i));
        indexToSlot.resize(size * sizeof(int));
        slotToIndex.resize(size * sizeof(int));
        indices[0].resize(size * sizeof(int));
    }
    this->size = size;
}

bool RayBuffer::getClosestHit() const {
    return closestHit;
}

void RayBuffer::setClosestHit(bool closestHit) {
    this->closestHit = closestHit;
}

int RayBuffer::getMortonCodeBits() {
    return mortonCodeBits;
}

void RayBuffer::setMortonCodeBits(int mortonCodeBits) {
    if (mortonCodeBits < 6 || mortonCodeBits > 64) logger(LOG_WARN) << "WARN <RayBuffer> Morton code bits must be in range [6,64].\n";
    else this->mortonCodeBits = mortonCodeBits;
}

float RayBuffer::getRayLength() {
    return rayLength;
}

void RayBuffer::setRayLength(float rayLength) {
    if (rayLength < 0.0f || rayLength > 1.0f) logger(LOG_WARN) << "WARN <RayBuffer> Ray length must be in range [0,1].\n";
    else this->rayLength = rayLength;
}

RayBuffer::MortonCodeMethod RayBuffer::getMortonCodeMethod() {
    return mortonCodeMethod;
}

void RayBuffer::setMortonCodeMethod(MortonCodeMethod mortonCodeMethod) {
    this->mortonCodeMethod = mortonCodeMethod;
}

Buffer & RayBuffer::getRayBuffer() {
    return rays;
}
Buffer & RayBuffer::getResultBuffer() {
    return results;
}

Buffer & RayBuffer::getStatBuffer() {
    return stats;
}

Buffer & RayBuffer::getIndexBuffer() {
    return indices[0];
}

Buffer & RayBuffer::getIndexToSlotBuffer() {
    return indexToSlot;
}

Buffer & RayBuffer::getSlotToIndexBuffer() {
    return slotToIndex;
}
