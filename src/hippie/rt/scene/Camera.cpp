/**
 * \file	Camera.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Camera class source file.
 */

#include "Camera.h"
#include "environment/AppEnvironment.h"
#include "util/Logger.h"

Camera::Camera() :
    nearPlane(0.001f),
    farPlane(3.0f),
    fieldOfView(radians(45.0f)),
    position(Vec3f(0.0f)),
    wheelAngle(0.0f),
    deltaAngle(1.0f),
    step(0.01f)
{
    setDirection(Vec3f(0.0f, 0.0f, -1.0f));
    reset();
}

Camera::~Camera() {
}

float Camera::getNear() const {
    return nearPlane;
}
void Camera::setNear(float _near) {
    this->nearPlane = _near;
}

float Camera::getFar() const {
    return farPlane;
}

void Camera::setFar(float _far) {
    this->farPlane = _far;
}

float Camera::getFieldOfView() const {
    return degrees(fieldOfView);
}

void Camera::setFieldOfView(float fieldOfView) {
    this->fieldOfView = radians(fieldOfView);
}

const Vec3f & Camera::getPosition() {
    return position;
}

void Camera::setPosition(const Vec3f & position) {
    this->position = position;
}

Vec3f Camera::getDirection() {
    Mat4f rotMatrix = rotate(radians(-elevAngle), Vec3f(1.0f, 0.0f, 0.0f)) *
        rotate(radians(viewAngle), Vec3f(0.0f, 1.0f, 0.0f));
    return normalize(-Vec3f(rotMatrix[0][2], rotMatrix[1][2], rotMatrix[2][2]));
}

void Camera::setDirection(const Vec3f & _direction) {
    Vec3f direction = normalize(_direction);
    elevAngle = degrees(asinf(direction.y));
    viewAngle = degrees(atan2f(direction.x, -direction.z));
}

const Vec2i & Camera::getSize() const {
    return size;
}

void Camera::setSize(const Vec2i & size) {
    this->size = size;
}

float Camera::getDeltaAngle() const {
    return deltaAngle;
}

void Camera::setDeltaAngle(float rotSpeed) {
    this->deltaAngle = rotSpeed;
}

float Camera::getStep() const {
    return step;
}

void Camera::setStep(float step) {
    this->step = step;
}

Mat4f Camera::getProjectionMatrix() {
    return perspective(fieldOfView, ((float)size.x) / size.y, nearPlane, farPlane);
}

Mat4f Camera::getViewMatrix() {
    return
        rotate(radians(wheelAngle), Vec3f(0.0f, 0.0f, 1.0f)) *
        rotate(radians(-elevAngle), Vec3f(1.0f, 0.0f, 0.0f)) *
        rotate(radians(viewAngle), Vec3f(0.0f, 1.0f, 0.0f)) *
        translate(Vec3f(-position.x, -position.y, -position.z));
}

Mat4f Camera::getProjectionViewMatrix() {
    return getProjectionMatrix() * getViewMatrix();
}

void Camera::moveForward() {
    position += step * getDirection();
}

void Camera::moveBackward() {
    position -= step * getDirection();
}

void Camera::turnLeft() {
    addViewAngle(-deltaAngle);
}

void Camera::turnRight() {
    addViewAngle(deltaAngle);
}

void Camera::rotateCW() {
    addWheelAngle(deltaAngle);
}

void Camera::rotateCCW() {
    addWheelAngle(-deltaAngle);
}

float Camera::getElevAngle() {
    return elevAngle;
}

float Camera::getViewAngle() {
    return viewAngle;
}

float Camera::getWheelAngle() {
    return wheelAngle;
}

void Camera::addElevAngle(float angle) {
    viewAngle += angle * sinf(radians(wheelAngle));
    elevAngle += angle * cosf(radians(wheelAngle));
    if (elevAngle > 75.0f) elevAngle = 75.0f;
    if (elevAngle < -75.0f) elevAngle = -75.0f;
    while (viewAngle < -360.0f) viewAngle += 360.0f;
    while (viewAngle > 360.0f) viewAngle -= 360.0f;
}

void Camera::setElevAngle(float angle) {
    elevAngle = 0.0f;
    addElevAngle(angle);
}

void Camera::addViewAngle(float angle) {
    viewAngle += angle * cosf(radians(wheelAngle));
    elevAngle += -angle * sinf(radians(wheelAngle));
    if (elevAngle > 75.0f) elevAngle = 75.0f;
    if (elevAngle < -75.0f) elevAngle = -75.0f;
    while (viewAngle < -360.0f) viewAngle += 360.0f;
    while (viewAngle > 360.0f) viewAngle -= 360.0f;
}

void Camera::setViewAngle(float angle) {
    viewAngle = 0.0f;
    addViewAngle(angle);
}

void Camera::addWheelAngle(float angle) {
    wheelAngle += angle;
    while (wheelAngle < -360.0f) wheelAngle += 360.0f;
    while (wheelAngle > 360.0f) wheelAngle -= 360.0f;
}

void Camera::setWheelAngle(float angle) {
    wheelAngle = angle;
    while (wheelAngle < -360.0f) wheelAngle += 360.0f;
    while (wheelAngle > 360.0f) wheelAngle -= 360.0f;
}

void Camera::reset() {
    QVector3D _position;
    Environment::getInstance()->getVectorValue("Camera.position", _position);
    setPosition(Vec3f(_position.x(), _position.y(), _position.z()));
    QVector3D _direction;
    Environment::getInstance()->getVectorValue("Camera.direction", _direction);
    setDirection(Vec3f(_direction.x(), _direction.y(), _direction.z()));
    float _wheelAngle;
    Environment::getInstance()->getFloatValue("Camera.wheelAngle", _wheelAngle);
    setWheelAngle(_wheelAngle);
    float _nearPlane;
    Environment::getInstance()->getFloatValue("Camera.nearPlane", _nearPlane);
    setNear(_nearPlane);
    float _farPlane;
    Environment::getInstance()->getFloatValue("Camera.farPlane", _farPlane);
    setFar(_farPlane);
    float _fieldOfView;
    Environment::getInstance()->getFloatValue("Camera.fieldOfView", _fieldOfView);
    setFieldOfView(_fieldOfView);
    float _step;
    Environment::getInstance()->getFloatValue("Camera.step", _step);
    setStep(_step);
}
