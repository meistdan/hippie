/**
 * \file	AssimpLoader.h
 * \author	Daniel Meister
 * \date	2014/04/28
 * \brief	AssimpImporter class header file.
 */

#ifndef _ASSIMP_IMPORTER_H_
#define _ASSIMP_IMPORTER_H_

#include "SceneImporter.h"
#include <assimp/cimport.h>
#include <assimp/config.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define REMOVE_DEGENERATES 1

#if REMOVE_DEGENERATES
#define POSTPROCESS_FLAGS ( \
	aiProcess_Triangulate			| \
	aiProcess_SortByPType			| \
	aiProcess_GenSmoothNormals		| \
	aiProcess_ValidateDataStructure | \
    aiProcess_GenUVCoords			| \
    aiProcess_FlipUVs               | \
	aiProcess_FindDegenerates		| \
    0 \
)
#else
#define POSTPROCESS_FLAGS ( \
	aiProcess_Triangulate			| \
	aiProcess_SortByPType			| \
	aiProcess_GenSmoothNormals		| \
	aiProcess_ValidateDataStructure | \
    aiProcess_GenUVCoords			| \
    aiProcess_FlipUVs               | \
    0 \
)
#endif

class AssimpImporter : public SceneImporter {

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

#endif /* _ASSIMP_LOADER_H_ */
