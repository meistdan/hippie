/**
 * \file	Material.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Material struct header file.
 */

#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include "Globals.h"
#include "util/Math.h"

struct Material {
    int texIndex;
    float shininess;
    Vec3f diffuse;
    Vec3f specular;
};

#endif /* _MATERIAL_H_ */
