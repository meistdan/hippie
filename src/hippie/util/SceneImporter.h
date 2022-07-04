/**
 * \file	SceneImporter.h
 * \author	Daniel Meister
 * \date	2014/04/28
 * \brief	SceneImporter class header file.
 */

#ifndef _SCENE_IMPORTER_H_
#define _SCENE_IMPORTER_H_

#include "gpu/Buffer.h"
#include "util/AABB.h"

class SceneImporter {

public:

    virtual bool importScene(
        const QString & filename,
        int & numberOfVertices,
        int & numberOfTriangles,
        int & numberOfMaterials,
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

#endif /* _SCENE_IMPORTER_H_ */ 
