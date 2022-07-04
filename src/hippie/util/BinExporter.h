/**
 * \file	SceneExporter.h
 * \author	Daniel Meister
 * \date	2014/04/29
 * \brief	SceneExporter class header file.
 */

#ifndef _BIN_EXPORTER_H_
#define _BIN_EXPORTER_H_

#include "SceneExporter.h"

class BinExporter : public SceneExporter {

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
    );

};

#endif /* _BIN_EXPORTER_H_ */
