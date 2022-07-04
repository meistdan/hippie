/**
  * \file	RayBuffer.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	RayBuffer class header file.
  */

#ifndef _RAY_BUFFER_H_
#define _RAY_BUFFER_H_

#include "Ray.h"
#include "RayBufferKernels.h"
#include "gpu/Buffer.h"
#include "gpu/HipCompiler.h"

class RayBuffer {

public:

    enum MortonCodeMethod {
        TWO_POINT,
        AILA,
        PARABOLOID,
        OCTAHEDRON,
        ORIGIN,
        COSTA,
        REIS
    };

private:

    HipCompiler compiler;

    int capacity;
    int size;
    Buffer rays;
    Buffer results;
    Buffer indexToSlot;
    Buffer slotToIndex;
    Buffer stats;
    Buffer mortonCodes[2];
    Buffer indices[2];
    Buffer spine;
    bool closestHit;

    MortonCodeMethod mortonCodeMethod;
    int mortonCodeBits;
    float rayLength;

    void randomSort(void);

    float computeMortonCodes(Buffer & mortonCodes);
    float reorderRays(Buffer & oldRayBuffer, Buffer & oldIndexToPixel);
    float mortonSort(void);

public:

    RayBuffer(void);

    int getSize(void) const;
    void resize(int n);

    bool getClosestHit(void) const;
    void setClosestHit(bool closestHit);
    int getMortonCodeBits(void);
    void setMortonCodeBits(int mortonCodeBits);
    float getRayLength(void);
    void setRayLength(float rayLength);
    MortonCodeMethod getMortonCodeMethod(void);
    void setMortonCodeMethod(MortonCodeMethod mortonCodeMethod);

    Buffer & getRayBuffer(void);
    Buffer & getResultBuffer(void);
    Buffer & getStatBuffer(void);
    Buffer & getIndexBuffer(void);
    Buffer & getIndexToSlotBuffer(void);
    Buffer & getSlotToIndexBuffer(void);

    friend class HipTracer;

};

#endif /* _RAY_BUFFER_H_ */
