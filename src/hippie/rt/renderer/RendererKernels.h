/**
 * \file	RendererKenrels.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Renderer kernels header file.
 */

#ifndef _RENDERER_KERNELS_H_
#define _RENDERER_KERNELS_H_

#include "Globals.h"
#include "rt/scene/Material.h"
#include "rt/ray/Ray.h"

#define BACKGROUND_COLOR Vec3f(0.52f, 0.69f, 1.0f)
#define MAX_MATERIALS 512
#define HITS_BLOCK_THREADS 256

#ifdef __KERNELCC__
extern "C" {

    CONSTANT int4 diffuseTextureItems[MAX_MATERIALS];
    CONSTANT unsigned char materials[MAX_MATERIALS * sizeof(Material)];

    DEVICE int rayHits;

    GLOBAL void reconstructSmooth(
        const int numberOfRays,
        const int numberOfSamples,
        unsigned long long diffuseTex,
        unsigned long long environmentTex,
        int * matIndices,
        Vec3i * triangles,
        Vec3f * normals,
        Vec2f * texCoords,
        Ray * rays,
        RayResult *  results,
        Vec3f light,
        int * indexToPixel,
        Vec4f * pixels,
        Vec3f * decreases
    );

    GLOBAL void reconstructPseudocolor(
        const int numberOfPixels,
        const int numberOfSamples,
        int * matIndices,
        Vec3i * triangles,
        Vec3f * normals,
        Vec3f * pseudocolors,
        Ray * rays,
        RayResult *  results,
        Vec3f light,
        int * indexToPixel,
        Vec4f * pixels
    );

    GLOBAL void reconstructThermal(
        const int numberOfPixels,
        const int numberOfSamples,
        const int threshold,
        int * indexToPixel,
        Vec2i * stats,
        Vec4f * pixels
    );

    GLOBAL void reconstructShadow(
        const int batchBegin,
        const int batchSize,
        const int numberOfSamples,
        const bool replace,
        RayResult * outputResults,
        int * indexToPixel,
        int * indexToSlot,
        Vec4f * inPixels,
        Vec4f * outPixels
    );

    GLOBAL void reconstructAO(
        const int batchBegin,
        const int batchSize,
        const int numberOfSamples,
        const bool replace,
        RayResult * outputResults,
        int * indexToPixel,
        int * indexToSlot,
        Vec4f * inPixels,
        Vec4f * outPixels
    );

    GLOBAL void interpolateColors(
        const int numberOfPixels,
        const int frameIndex,
        const float keyValue,
        const float whitePoint,
        Vec4f * framePixels,
        Vec4f * pixels
    );

    GLOBAL void initDecreases(
        const int numberOfPixels,
        Vec3f * decreases
    );

    GLOBAL void countRayHits(
        const int numberOfRays,
        RayResult * rayResults
    );

}
#endif

#endif /* _RENDERER_KERNELS_H_ */
