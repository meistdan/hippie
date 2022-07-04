/**
 * \file	BinImporter.h
 * \author	Daniel Meister
 * \date	2014/04/29
 * \brief	BinImporter class header file.
 */

#ifndef _BIN_IMPORTER_H_
#define _BIN_IMPORTER_H_

#include "SceneImporter.h"

class BinImporter : public SceneImporter {

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
    );
};

#endif /* _BIN_IMPORTER_H_ */
