/**
 * \file	InterpolatorKernels.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Interpolator kernels header file.
 */

#ifndef _INTERPOLATOR_KERNELS_H_
#define _INTERPOLATOR_KERNELS_H_

#include "Globals.h"

#ifdef __KERNELCC__
extern "C" {

    GLOBAL void interpolateVertices(
        const int numberOfStaticVertices,
        const int numberOfDynamicVertices,
        const float t,
        Vec3f * vertices0,
        Vec3f * vertices1,
        Vec3f * vertices
    );

    GLOBAL void interpolateNormals(
        const int numberOfStaticNormals,
        const int numberOfDynamicNormals,
        const float t,
        Vec3f * normals0,
        Vec3f * normals1,
        Vec3f * normals
    );

}
#endif

#endif /* _INTERPOLATOR_KERNELS_H_ */
