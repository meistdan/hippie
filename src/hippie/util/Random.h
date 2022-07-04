/**
 * \file	Random.h
 * \author	Daniel Meister
 * \date	2016/08/29
 * \brief	A header file containing random fnctions based on C++ 11.
 */

#ifndef _RANDOM_H_
#define _RANDOM_H

#include <random>
#include "Globals.h"

class Random : public std::mt19937 {

private:

    Vec3f arbitraryNormal(const Vec3f &n) {
        float dist2 = n.x * n.x + n.y * n.y;
        if (dist2 > 0.0001f) {
            float invSize = 1.0f / sqrtf(dist2);
            return Vec3f(n.y * invSize, -n.x * invSize, 0); // N x (0,0,1)
        }
        float invSize = 1.0f / sqrtf(n.z * n.z + n.x * n.x);
        return Vec3f(-n.z * invSize, 0, n.x * invSize); // N x (0,1,0)
    }

    void rightHandedBase(const Vec3f & n, Vec3f & u, Vec3f & v) {
        v = arbitraryNormal(n);
        u = cross(v, n);
    }

public:

    int randomInt(int mn, int mx) {
        std::uniform_int_distribution<int> distribution(mn, mx);
        return distribution(*this);
    }

    int randomInt(int mx) {
        std::uniform_int_distribution<int> distribution(0, mx);
        return distribution(*this);
    }

    float randomFloat(float mn = 0.0f, float mx = 1.0f) {
        std::uniform_real_distribution<float> distribution(mn, mx);
        return distribution(*this);
    }

    float randomFloat(float mx) {
        std::uniform_real_distribution<float> distribution(0, mx);
        return distribution(*this);
    }

    Vec3f uniformRandomVector(void) {
        float r1 = randomFloat();
        float r2 = randomFloat();
        float cosTheta = 1.0f - 2 * r1;
        float sinTheta = sqrtf(1 - sqrtf(cosTheta));
        float fi = 2.0f * M_PIf * r2;
        return Vec3f(sinTheta * sin(fi), cosTheta, sinTheta * cos(fi));
    }

    Vec3f cosineRandomVector(const Vec3f & n) {
        float r1 = randomFloat();
        float r2 = randomFloat();
        float theta = 2.0f * M_PIf * r2;
        float radius = sqrtf(r1);
        float x = radius * sinf(theta);
        float z = radius*cos(theta);
        float y = sqrt(1.0f - x*x - z*z);
        Vec3f dir(x, y, z);
        Vec3f u, v;
        rightHandedBase(n, u, v);
        return x * u + z * v + y * n;
    }

};

#endif /* _RANDOM_H_ */
