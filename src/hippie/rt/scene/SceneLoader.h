/**
 * \file	SceneLoader.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	SceneLoader class header file.
 */

#ifndef _SCENE_LOADER_H_
#define _SCENE_LOADER_H_

#include <QStringList>
#include "DynamicScene.h"
#include "util/AABB.h"
#include "util/BinExporter.h"
#include "util/BinImporter.h"
#include "util/AssimpImporter.h"

#define BINARY_CACHE 1

class SceneLoader {

private:

    BinExporter binExporter;
    BinImporter binImporter;
    AssimpImporter assimpImporter;

    void loadScene(
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
    );

public:

    Scene * loadStaticScene(const QString & filename);
    DynamicScene * loadDynamicScene(const QStringList & filenames);
    DynamicScene * loadDynamicScene(const QString & staticFilename, const QStringList & dynamicFilenames);

};

#endif /* _SCENE_LOADER_H_ */
