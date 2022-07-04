/**
  * \file	RayGenKernels.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	RayGen kernels header file.
  */

#ifndef _RAY_GEN_KERNELS_H_
#define _RAY_GEN_KERNELS_H_

#include "Globals.h"
#include "rt/scene/Material.h"
#include "Ray.h"

#define MAX_MATERIALS 512

#ifdef __KERNELCC__
extern "C" {

    CONSTANT int4 diffuseTextureItems[MAX_MATERIALS];
    CONSTANT unsigned char materials[MAX_MATERIALS * sizeof(Material)];

    GLOBAL void generatePrimaryRays(
        const int sampleIndex,
        Vec3f origin,
        Mat4f screenToWorld,
        Vec2i size,
        float maxDist,
        int * indexToPixel,
        Ray * rays
    );

    GLOBAL void generateShadowRays(
        const int batchBegin,
        const int batchSize,
        const int numberOfSamples,
        unsigned int * seeds,
        float lightRadius,
        Vec3f light,
        Ray * inputRays,
        Ray * outputRays,
        RayResult * inputResults,
        int * outputSlotToIndex,
        int * outputIndexToSlot
    );

    GLOBAL void generateAORays(
        const int batchBegin,
        const int batchSize,
        const int numberOfSamples,
        unsigned int * seeds,
        float maxDist,
        Ray * inputRays,
        Ray * outputRays,
        RayResult * inputResults,
        int * outputSlotToIndex,
        int * outputIndexToSlot,
        Vec3i * vertIndices,
        Vec3f * vertices
    );

    GLOBAL void generatePathRays(
        const bool russianRoulette,
        const int numberOfInputRays,
        unsigned long long diffuseTex2,
        unsigned int * seeds,
        int * numberOfOutputRaysLoc,
        int * inputIndexToPixel,
        int * outputIndexToPixel,
        int * matIndices,
        Vec3i * triangles,
        Vec3f * normals,
        Vec2f * texCoords,
        Ray * inputRays,
        Ray * outputRays,
        RayResult * inputResults,
        Vec3f * decreases
    );

    GLOBAL void initSeeds(
        const int numberOfPixels,
        const int frameIndex,
        unsigned int * seeds
    );

}
#endif

#endif /* _RAY_GEN_KERNELS_H_ */
