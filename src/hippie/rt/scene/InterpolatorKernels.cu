/**
 * \file	InterpolatorKernels.cu
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Interpolator kernels source file.
 */

#include "rt/scene/InterpolatorKernels.h"

extern "C" GLOBAL void interpolateVertices(
    const int numberOfStaticVertices,
    const int numberOfDynamicVertices,
    const float t,
    Vec3f * vertices0,
    Vec3f * vertices1,
    Vec3f * vertices
) {

    // Vertex index.
    const int vertexIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Interpolate values.
    if (vertexIndex < numberOfDynamicVertices) {
        vertices[numberOfStaticVertices + vertexIndex] = mix(vertices0[vertexIndex], vertices1[vertexIndex], t);
    }

}

extern "C" GLOBAL void interpolateNormals(
    const int numberOfStaticNormals,
    const int numberOfDynamicNormals,
    const float t,
    Vec3f * normals0,
    Vec3f * normals1,
    Vec3f * normals
) {

    // Normal index.
    const int normalIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Interpolate values.
    if (normalIndex < numberOfDynamicNormals) {
        normals[numberOfStaticNormals + normalIndex] = normalize(mix(normals0[normalIndex], normals1[normalIndex], t));
    }

}
