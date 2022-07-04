/**
 * \file	Scene.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Scene class source file.
 */

#include "environment/AppEnvironment.h"
#include "Scene.h"

Scene::Scene() : numberOfVertices(0), numberOfTriangles(0) {
    QVector3D _light;
    Environment::getInstance()->getVectorValue("Scene.light", _light);
    setLight(Light(_light.x(), _light.y(), _light.z()));
}

Scene::~Scene() {
}

int Scene::getNumberOfVertices() const {
    return numberOfVertices;
}

int Scene::getNumberOfTriangles() const {
    return numberOfTriangles;
}

int Scene::getNumberOfMaterials() const {
    return numberOfMaterials;
}

bool Scene::hasTextures() const {
    return textureAtlas.getNumberOfTextures() > 0;
}

Buffer & Scene::getVertexBuffer() {
    return vertices;
}

Buffer & Scene::getTriangleBuffer() {
    return vertIndices;
}

Buffer & Scene::getNormalBuffer() {
    return normals;
}

Buffer & Scene::getTexCoordBuffer() {
    return texCoords;
}

Buffer & Scene::getMatIndexBuffer() {
    return matIndices;
}

Buffer & Scene::getMaterialBuffer() {
    return materials;
}

Buffer & Scene::getPseudocolorBuffer() {
    return pseudocolors;
}

const AABB & Scene::getCentroidBox() {
    return centroidBox;
}

const AABB & Scene::getSceneBox() {
    return sceneBox;
}

const Light & Scene::getLight() {
    return light;
}

void Scene::setLight(const Light & light) {
    this->light = light;
}

TextureAtlas & Scene::getTextureAtlas() {
    return textureAtlas;
}

EnvironmentMap & Scene::getEnvironmentMap() {
    return environmentMap;
}

bool Scene::isDynamic() const {
    return false;
}
