/**
 * \file	Camera.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Camera class header file.
 */

#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "Globals.h"
#include "util/Math.h"
#include <QVector>

class Camera {

private:

    float nearPlane;
    float farPlane;
    float fieldOfView;

    Vec2i size;
    Vec3f position;

    float deltaAngle;
    float step;

    float viewAngle;
    float elevAngle;
    float wheelAngle;

public:

    Camera(void);
    ~Camera(void);

    float getNear(void) const;
    void setNear(float near);

    float getFar(void) const;
    void setFar(float far);

    float getFieldOfView(void) const;
    void setFieldOfView(float fieldOfView);

    const Vec3f & getPosition(void);
    void setPosition(const Vec3f & position);

    Vec3f getDirection(void);
    void setDirection(const Vec3f & direction);

    const Vec2i & getSize(void) const;
    void setSize(const Vec2i & size);

    float getDeltaAngle(void) const;
    void setDeltaAngle(float deltaAngle);

    float getStep(void) const;
    void setStep(float step);

    Mat4f getProjectionMatrix(void);
    Mat4f getViewMatrix(void);
    Mat4f getProjectionViewMatrix(void);

    void moveForward(void);
    void moveBackward(void);
    void turnLeft(void);
    void turnRight(void);
    void rotateCW(void);
    void rotateCCW(void);

    float getElevAngle(void);
    float getViewAngle(void);
    float getWheelAngle(void);

    void addElevAngle(float angle);
    void setElevAngle(float angle);
    void addViewAngle(float angle);
    void setViewAngle(float angle);
    void addWheelAngle(float angle);
    void setWheelAngle(float angle);

    void reset(void);

    friend class CameraLoader;

};

#endif /* _CAMERA_H_ */
