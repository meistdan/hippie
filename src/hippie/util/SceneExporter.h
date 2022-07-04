/**
 * \file	SceneExporter.h
 * \author	Daniel Meister
 * \date	2014/04/28
 * \brief	SceneExporter class header file.
 */

#ifndef _SCENE_EXPORTER_H_
#define _SCENE_EXPORTER_H_

#include "gpu/Buffer.h"
#include "util/AABB.h"

class SceneExporter {

public:

    virtual bool exportScene(
        const QString & filename,
        int numberOfVertices,
        int numberOfTriangles,
        int numberOfMaterials,
        Buffer & vertices,
        Buffer & normals,
        Buffer & texCoords,
        Buffer & vertIndices,
        Buffer & matIndices,
        Buffer & materials,
        QStringList & textureFilenames,
        AABB & boundingBox
    ) = 0;

};

#endif /* _SCENE_EXPORTER_H_ */ 
