/**
 * \file	Scene.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Scene class header file.
 */

#ifndef _SCENE_H_
#define _SCENE_H_

#include "gpu/Buffer.h"
#include "Material.h"
#include "EnvironmentMap.h"
#include "TextureAtlas.h"
#include "util/AABB.h"

typedef Vec3f Light;

class Scene {

protected:

    int numberOfVertices;
    int numberOfTriangles;
    int numberOfMaterials;

    Buffer vertices;
    Buffer normals;
    Buffer texCoords;
    Buffer vertIndices;
    Buffer matIndices;
    Buffer materials;
    Buffer pseudocolors;

    AABB centroidBox;
    AABB sceneBox;

    Light light;
    TextureAtlas textureAtlas;
    EnvironmentMap environmentMap;

    Scene(void);

public:

    virtual ~Scene(void);

    int getNumberOfVertices(void) const;
    int getNumberOfTriangles(void) const;
    int getNumberOfMaterials(void) const;
    bool hasTextures(void) const;

    Buffer & getVertexBuffer(void);
    Buffer & getTriangleBuffer(void);
    Buffer & getNormalBuffer(void);
    Buffer & getTexCoordBuffer(void);
    Buffer & getMatIndexBuffer(void);
    Buffer & getMaterialBuffer(void);
    Buffer & getPseudocolorBuffer(void);

    const AABB & getSceneBox(void);
    const AABB & getCentroidBox(void);

    const Light & getLight(void);
    void setLight(const Light & light);

    TextureAtlas & getTextureAtlas(void);
    EnvironmentMap & getEnvironmentMap(void);

    virtual bool isDynamic(void) const;

    friend class SceneLoader;

};

#endif /* _SCENE_H_ */
