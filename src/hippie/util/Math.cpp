/**
 * \file	Math.cpp
 * \author	Daniel Meister
 * \date	2021/12/09
 * \brief	Math class source file.
 */

#include "Math.h"

HOST_DEVICE Mat4f perspective(float fovy, float aspect, float near, float far) {
    float fy = rcp(tan(fovy * 0.5f));
    float fx = fy / aspect;
    float d = rcp(near - far);
    Mat4f r;
    r.setRow(0, Vec4f(fx, 0.0f, 0.0f, 0.0f));
    r.setRow(1, Vec4f(0.0f, fy, 0.0f, 0.0f));
    r.setRow(2, Vec4f(0.0f, 0.0f, (near + far) * d, 2.0f * near * far * d));
    r.setRow(3, Vec4f(0.0f, 0.0f, -1.0f, 0.0f));
    return r;
}

HOST_DEVICE Mat4f rotate(float angle, const Vec3f & axis) {
    Mat4f R;
    float cosa = cosf(angle);
    float sina = sinf(angle);
    R(0, 0) = cosa + sqr(axis.x) * (1.0f - cosa);			    R(0, 1) = axis.x * axis.y * (1.0f - cosa) - axis.z * sina;	R(0, 2) = axis.x * axis.z * (1.0f - cosa) + axis.y * sina;
    R(1, 0) = axis.x * axis.y * (1.0f - cosa) + axis.z * sina;	R(1, 1) = cosa + sqr(axis.y) * (1.0f - cosa);			    R(1, 2) = axis.y * axis.z * (1.0f - cosa) - axis.x * sina;
    R(2, 0) = axis.z * axis.x * (1.0f - cosa) - axis.y * sina;	R(2, 1) = axis.z * axis.y * (1.0f - cosa) + axis.x * sina;	R(2, 2) = cosa + sqr(axis.z) * (1.0f - cosa);
    return R;
}

HOST_DEVICE Mat4f translate(const Vec3f & xyz) {
    Mat4f R;
    R(0, 3) = xyz.x;
    R(1, 3) = xyz.y;
    R(2, 3) = xyz.z;
    return R;
}

HOST_DEVICE Mat4f scale(const Vec3f & xyz) {
    Mat4f R;
    R(0, 0) = xyz.x;
    R(1, 1) = xyz.y;
    R(2, 2) = xyz.z;
    return R;
}
