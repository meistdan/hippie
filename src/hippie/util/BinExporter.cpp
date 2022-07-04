/**
 * \file	BinExporter.cpp
 * \author	Daniel Meister
 * \date	2014/04/29
 * \brief	BinExporter class source file.
 */

#include "BinExporter.h"
#include "Logger.h"
#include "rt/scene/Material.h"
#include "util/Math.h"
#include <QStringList>
#include <QDataStream>

bool BinExporter::exportScene(
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
) {

    // File object.
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) {
        logger(LOG_WARN) << "WARN <BinExporter> Cannot open file '" << filename << "'.\n";
        return false;
    }

    // Write header.
    int numberOfTextures = textureFilenames.size();
    file.write((const char*)&numberOfVertices, sizeof(int));
    file.write((const char*)&numberOfTriangles, sizeof(int));
    file.write((const char*)&numberOfMaterials, sizeof(int));
    file.write((const char*)&numberOfTextures, sizeof(int));

    // Write bounding box.
    file.write((const char*)&boundingBox, sizeof(AABB));

    // Write data.
    file.write((const char*)vertices.getMutablePtr(), sizeof(Vec3f) * numberOfVertices);
    file.write((const char*)normals.getMutablePtr(), sizeof(Vec3f) * numberOfVertices);
    file.write((const char*)texCoords.getMutablePtr(), sizeof(Vec2f) * numberOfVertices);
    file.write((const char*)vertIndices.getMutablePtr(), sizeof(Vec3i) * numberOfTriangles);
    file.write((const char*)matIndices.getMutablePtr(), sizeof(int) * numberOfTriangles);
    file.write((const char*)materials.getMutablePtr(), sizeof(Material) * numberOfMaterials);

    // Texture filenames.
    QDataStream out(&file);
    for (int i = 0; i < numberOfTextures; ++i) {
        out << textureFilenames[i];
    }

    // Close file.
    file.close();

    return true;

}
