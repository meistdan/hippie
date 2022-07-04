/**
 * \file	BinImporter.cpp
 * \author	Daniel Meister
 * \date	2014/04/29
 * \brief	BinImporter class source file.
 */

#include "BinImporter.h"
#include "Logger.h"
#include "rt/scene/Material.h"
#include "util/Math.h"
#include <QFile>
#include <QDataStream>
#include <QStringList>

bool BinImporter::importScene(
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
) {

    // File object.
    QFile file(filename);
    if (!file.exists()) {
        logger(LOG_WARN) << "WARN <BinImporter> File '" << filename << "' does not exist.\n";
        return false;
    }

    if (!file.open(QIODevice::ReadOnly)) {
        logger(LOG_WARN) << "WARN <BinImporter> Cannot open file '" << filename << "'.\n";
        return false;
    }

    // Load header.
    int header[4];
    file.read((char*)header, 4 * sizeof(int));

    // Counts.
    numberOfVertices = header[0];
    numberOfTriangles = header[1];
    numberOfMaterials = header[2];
    int numberOfTextures = header[3];

    // Read bounding box.
    file.read((char*)&boundingBox, sizeof(AABB));

    // Resize buffers.
    vertices.resizeDiscard(sizeof(Vec3f) * numberOfVertices);
    normals.resizeDiscard(sizeof(Vec3f) * numberOfVertices);
    texCoords.resizeDiscard(sizeof(Vec2f) * numberOfVertices);
    vertIndices.resizeDiscard(sizeof(Vec3i) * numberOfTriangles);
    matIndices.resizeDiscard(sizeof(int) * numberOfTriangles);
    materials.resizeDiscard(sizeof(Material) * numberOfMaterials);

    // Read data.
    file.read((char*)vertices.getMutablePtr(), sizeof(Vec3f) * numberOfVertices);
    file.read((char*)normals.getMutablePtr(), sizeof(Vec3f) * numberOfVertices);
    file.read((char*)texCoords.getMutablePtr(), sizeof(Vec2f) * numberOfVertices);
    file.read((char*)vertIndices.getMutablePtr(), sizeof(Vec3i) * numberOfTriangles);
    file.read((char*)matIndices.getMutablePtr(), sizeof(int) * numberOfTriangles);
    file.read((char*)materials.getMutablePtr(), sizeof(Material) * numberOfMaterials);

    // Texture filenames.
    textureFilenames.clear();
    QDataStream in(&file);
    for (int i = 0; i < numberOfTextures; ++i) {
        QString textureFilename;
        in >> textureFilename;
        textureFilenames.push_back(textureFilename);
    }

    // Close file.
    file.close();

    return true;

}
