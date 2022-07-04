/**
 * \file	Interpolator.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Interpolator class header file.
 */

#ifndef _INTERPOLATOR_H_
#define _INTERPOLATOR_H_

#include "gpu/HipCompiler.h"
#include "DynamicScene.h"

#define INTERPOLATOR_EPSILON 1.0e-5f

class Interpolator {

private:

    HipCompiler compiler;

public:

    Interpolator(void);
    ~Interpolator(void);

    float update(DynamicScene & scene);
    float updateAdaptive(DynamicScene & scene, float time);

};

#endif /* _INTERPOLATOR_H_ */
