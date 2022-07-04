/**
  * \file	RayGen.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	RayGen class header file.
  */

#ifndef _RAY_GEN_H_
#define _RAY_GEN_H_

#include "gpu/HipCompiler.h"
#include "rt/ray/RayBuffer.h"
#include "rt/ray/PixelTable.h"
#include "rt/scene/Scene.h"
#include "rt/scene/Camera.h"
#include "util/AABB.h"


class RayGen {

private:

    HipCompiler compiler;
    PixelTable pixelTable;
    Buffer seeds;
    bool russianRoulette;

public:

    RayGen(void);

    float initSeeds(int numberOfPixels, int frameIndex = 1);

    float primary(RayBuffer & orays, Camera & camera, int sampleIndex);
    float shadow(RayBuffer & orays, RayBuffer & irays, int batchBegin, int batchEnd, int numberOfSamples, const Vec3f & light, float lightRadius);
    float ao(RayBuffer & orays, RayBuffer & irays, Scene & scene, int batchBegin, int batchEnd, int numberOfSamples, float maxDist);
    float path(RayBuffer & orays, RayBuffer & irays, Buffer & decreases, Scene & scene);

    bool getRussianRoulette(void);
    void setRussianRoulette(bool russianRoulette);

};

#endif /* _RAY_GEN_H_ */
