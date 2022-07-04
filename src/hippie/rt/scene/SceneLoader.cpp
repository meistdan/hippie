/**
 * \file	SceneLoader.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	SceneLoader class source file.
 */

#include "Material.h"
#include "SceneLoader.h"
#include "environment/Environment.h"
#include "util/Logger.h"

#include <QElapsedTimer>

void SceneLoader::loadScene(
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
    QElapsedTimer timer;
    timer.start();
#if BINARY_CACHE
    if (binImporter.importScene(
        filename + ".bin",
        numberOfVertices,
        numberOfTriangles,
        numberOfMaterials,
        vertices,
        normals,
        texCoords,
        vertIndices,
        matIndices,
        materials,
        textureFilenames,
        boundingBox)) {
        logger(LOG_INFO) << "INFO <SceneLoader> Data loaded from binary cached file '" << filename << ".bin'.\n";
    }
    else {
        if (!assimpImporter.importScene(
            filename,
            numberOfVertices,
            numberOfTriangles,
            numberOfMaterials,
            vertices,
            normals,
            texCoords,
            vertIndices,
            matIndices,
            materials,
            textureFilenames,
            boundingBox
        )) {
            logger(LOG_ERROR) << "ERROR <SceneLoader> Importing scene '" << filename << "' failed!\n";
            exit(EXIT_FAILURE);
        }
        logger(LOG_INFO) << "INFO <SceneLoader> Binary cached file was not found. Data loaded directly from file '" << filename << "'.\n";
        binExporter.exportScene(
            filename + ".bin",
            numberOfVertices,
            numberOfTriangles,
            numberOfMaterials,
            vertices,
            normals,
            texCoords,
            vertIndices,
            matIndices,
            materials,
            textureFilenames,
            boundingBox
        );
    }
#else 
    if (!assimpImporter.importScene(
        filename,
        numberOfVertices,
        numberOfTriangles,
        numberOfMaterials,
        vertices,
        normals,
        texCoords,
        vertIndices,
        matIndices,
        materials,
        textureFilenames,
        boundingBox
    )) {
        logger(LOG_ERROR) << "ERROR <SceneLoader> Importing scene '" << filename << "' failed!\n";
        exit(EXIT_FAILURE);
    }
    logger(LOG_INFO) << "INFO <SceneLoader> Data loaded directly from file '" << filename << "'.\n";
#endif
}

Scene * SceneLoader::loadStaticScene(const QString & filename) {

    // Scene.
    Scene * scene = new Scene();

    // Texture filenames.
    QStringList textureFilenames;

    // Bounding box.
    AABB sceneBox;

    // Import scene.
    loadScene(
        filename,
        scene->numberOfVertices,
        scene->numberOfTriangles,
        scene->numberOfMaterials,
        scene->vertices,
        scene->normals,
        scene->texCoords,
        scene->vertIndices,
        scene->matIndices,
        scene->materials,
        textureFilenames,
        sceneBox
    );

    // Vertices and triangles.
    Vec3f * vertices = (Vec3f*)scene->vertices.getMutablePtr();
    Vec3i * triangles = (Vec3i*)scene->vertIndices.getPtr();

    // Normalize vertices.
    Vec3f diag = sceneBox.diagonal();
    float scale = 1.0f / qMax(qMax(diag.x, diag.y), diag.z);

    for (int i = 0; i < scene->getNumberOfVertices(); ++i) {
        vertices[i] -= sceneBox.mn;
        vertices[i] *= scale;
    }

    // Centroid box.
    scene->sceneBox.reset();
    scene->centroidBox.reset();
    for (int i = 0; i < scene->getNumberOfTriangles(); ++i) {
        AABB box;
        box.grow(vertices[triangles[i].x]);
        box.grow(vertices[triangles[i].y]);
        box.grow(vertices[triangles[i].z]);
        scene->sceneBox.grow(box);
        scene->centroidBox.grow(box.centroid());
    }

    // Pseudocolor buffer.
    scene->pseudocolors.resizeDiscard(sizeof(Vec3f) * scene->numberOfTriangles);

    // Create texture atlas.
    scene->textureAtlas.set(textureFilenames);

    // Load light.
    QVector3D light;
    Environment::getInstance()->getVectorValue("Scene.light", light);
    scene->setLight(Light(light.x(), light.y(), light.z()));

    // Load environment map.
    QString environmentMapFilename;
    Environment::getInstance()->getStringValue("Scene.environment", environmentMapFilename);
    scene->environmentMap.set(environmentMapFilename);

    return scene;

}

DynamicScene * SceneLoader::loadDynamicScene(const QStringList & filenames) {

    // Scene.
    DynamicScene * scene = new DynamicScene();
    scene->frames.resize(filenames.size());

    // Texture filenames.
    QStringList textureFilenames;

    // Bounding box.
    AABB sceneBox, frameBox;

    // Import the frames.
    for (int i = filenames.size() - 1; i >= 0; --i) {
        loadScene(
            filenames[i],
            scene->numberOfVertices,
            scene->numberOfTriangles,
            scene->numberOfMaterials,
            scene->frames[i].vertices,
            scene->frames[i].normals,
            scene->texCoords,
            scene->vertIndices,
            scene->matIndices,
            scene->materials,
            textureFilenames,
            frameBox
        );
        sceneBox.grow(frameBox);
    }

    // Normalize vertices.
    Vec3f diag = sceneBox.diagonal();
    float scale = 1.0f / qMax(qMax(diag.x, diag.y), diag.z);
    for (int i = 0; i < scene->getNumberOfFrames(); ++i) {
        Vec3f * vertices = (Vec3f*)scene->frames[i].vertices.getMutablePtr();
        for (int j = 0; j < scene->getNumberOfVertices(); ++j) {
            vertices[j] -= sceneBox.mn;
            vertices[j] *= scale;
        }
    }

    // Centroid box.
    Vec3i * triangles = (Vec3i*)scene->vertIndices.getPtr();
    for (int i = 0; i < scene->getNumberOfFrames(); ++i) {
        Vec3f * vertices = (Vec3f*)scene->frames[i].vertices.getMutablePtr();
        scene->frames[i].sceneBox.reset();
        scene->frames[i].centroidBox.reset();
        for (int j = 0; j < scene->getNumberOfTriangles(); ++j) {
            AABB box;
            box.grow(vertices[triangles[j].x]);
            box.grow(vertices[triangles[j].y]);
            box.grow(vertices[triangles[j].z]);
            scene->frames[i].sceneBox.grow(box);
            scene->frames[i].centroidBox.grow(box.centroid());
        }
    }

    // Set-up the first frame as active.
    scene->sceneBox = scene->frames.front().sceneBox;
    scene->centroidBox = scene->frames.front().centroidBox;
    scene->pseudocolors.resizeDiscard(sizeof(Vec3f) * scene->numberOfTriangles);
    scene->vertices.resizeDiscard(sizeof(Vec3f) * scene->numberOfVertices);
    scene->vertices = scene->frames.front().vertices;
    scene->normals.resizeDiscard(sizeof(Vec3f) * scene->numberOfVertices);
    scene->normals = scene->frames.front().normals;

    // Create texture atlas.
    scene->textureAtlas.set(textureFilenames);

    // Load light.
    QVector3D light;
    Environment::getInstance()->getVectorValue("Scene.light", light);
    scene->setLight(Light(light.x(), light.y(), light.z()));

    // Load environment map.
    QString environmentMapFilename;
    Environment::getInstance()->getStringValue("Scene.environment", environmentMapFilename);
    scene->environmentMap.set(environmentMapFilename);

    return scene;

}

DynamicScene * SceneLoader::loadDynamicScene(const QString & staticFilename, const QStringList & dynamicFilenames) {

    // Scene.
    DynamicScene * scene = new DynamicScene();
    scene->frames.resize(dynamicFilenames.size());

    // Texture filenames.
    QStringList textureFilenames;
    QStringList frameTextureFilenames;

    // Aux. buffers.
    int numberOfVertices = 0;
    int numberOfTriangles = 0;
    int numberOfMaterials = 0;
    Buffer materials;
    Buffer texCoords;
    Buffer vertIndices;
    Buffer matIndices;

    // Bounding box.
    AABB sceneBox, frameBox;

    // Static part.
    loadScene(
        staticFilename,
        scene->numberOfVertices,
        scene->numberOfTriangles,
        scene->numberOfMaterials,
        scene->vertices,
        scene->normals,
        scene->texCoords,
        scene->vertIndices,
        scene->matIndices,
        scene->materials,
        textureFilenames,
        sceneBox
    );

    // Static offsets.
    scene->numberOfStaticVertices = scene->numberOfVertices;

    // Dynamic part.
    for (int i = dynamicFilenames.size() - 1; i >= 0; --i) {
        loadScene(
            dynamicFilenames[i],
            numberOfVertices,
            numberOfTriangles,
            numberOfMaterials,
            scene->frames[i].vertices,
            scene->frames[i].normals,
            texCoords,
            vertIndices,
            matIndices,
            materials,
            frameTextureFilenames,
            frameBox
        );
        sceneBox.grow(frameBox);
    }

    // Normalize static vertices.
    Vec3f diag = sceneBox.diagonal();
    float scale = 1.0f / qMax(qMax(diag.x, diag.y), diag.z);
    Vec3f * vertices = (Vec3f*)scene->vertices.getMutablePtr();
    for (int j = 0; j < scene->numberOfStaticVertices; ++j) {
        vertices[j] -= sceneBox.mn;
        vertices[j] *= scale;
    }

    // Normalize dynamic vertices.
    for (int i = 0; i < scene->getNumberOfFrames(); ++i) {
        Vec3f * vertices = (Vec3f*)scene->frames[i].vertices.getMutablePtr();
        for (int j = 0; j < numberOfVertices; ++j) {
            vertices[j] -= sceneBox.mn;
            vertices[j] *= scale;
        }
    }

    // Static centroid box.
    scene->staticSceneBox.reset();
    scene->staticCentroidBox.reset();
    Vec3i * triangles = (Vec3i*)scene->vertIndices.getPtr();
    for (int i = 0; i < scene->getNumberOfTriangles(); ++i) {
        AABB box;
        box.grow(vertices[triangles[i].x]);
        box.grow(vertices[triangles[i].y]);
        box.grow(vertices[triangles[i].z]);
        scene->staticSceneBox.grow(box);
        scene->staticCentroidBox.grow(box.centroid());
    }

    // Dynamic centroid box.
    triangles = (Vec3i*)vertIndices.getPtr();
    for (int i = 0; i < scene->getNumberOfFrames(); ++i) {
        Vec3f * vertices = (Vec3f*)scene->frames[i].vertices.getMutablePtr();
        scene->frames[i].sceneBox.reset();
        scene->frames[i].centroidBox.reset();
        for (int j = 0; j < numberOfTriangles; ++j) {
            AABB box;
            box.grow(vertices[triangles[j].x]);
            box.grow(vertices[triangles[j].y]);
            box.grow(vertices[triangles[j].z]);
            scene->frames[i].sceneBox.grow(box);
            scene->frames[i].centroidBox.grow(box.centroid());
        }
    }

    // Offsets.
    int vertexOffset = scene->numberOfVertices;
    int materialOffset = scene->numberOfMaterials;
    int triangleOffset = scene->numberOfTriangles;
    int textureOffset = textureFilenames.size();

    // Add triangle and material offset.
    Vec3i * vertIndicesPtr = (Vec3i*)vertIndices.getMutablePtr();
    int * matIndicesPtr = (int*)matIndices.getMutablePtr();
    for (int i = 0; i < numberOfTriangles; ++i) {
        matIndicesPtr[i] += materialOffset;
        vertIndicesPtr[i].x += vertexOffset;
        vertIndicesPtr[i].y += vertexOffset;
        vertIndicesPtr[i].z += vertexOffset;
    }

    // Add texture offset.
    Material * materialsPtr = (Material*)materials.getMutablePtr();
    for (int i = 0; i < numberOfMaterials; ++i) {
        if (materialsPtr[i].texIndex > 0)
            materialsPtr[i].texIndex += textureOffset;
    }

    // Resize buffers.
    scene->numberOfVertices += numberOfVertices;
    scene->numberOfMaterials += numberOfMaterials;
    scene->numberOfTriangles += numberOfTriangles;
    scene->vertices.resize(sizeof(Vec3f) * scene->numberOfVertices);
    scene->normals.resize(sizeof(Vec3f) * scene->numberOfVertices);
    scene->texCoords.resize(sizeof(Vec2f) * scene->numberOfVertices);
    scene->materials.resize(sizeof(Material) * scene->numberOfMaterials);
    scene->vertIndices.resize(sizeof(Vec3i) * scene->numberOfTriangles);
    scene->matIndices.resize(sizeof(int) * scene->numberOfTriangles);
    scene->pseudocolors.resizeDiscard(sizeof(Vec3f) * scene->numberOfTriangles);

    // Copy dynamic data.
    scene->sceneBox = scene->staticSceneBox;
    scene->sceneBox.grow(scene->frames.front().sceneBox);
    scene->centroidBox = scene->staticCentroidBox;
    scene->centroidBox.grow(scene->frames.front().centroidBox);
    scene->vertices.setRange(vertexOffset * sizeof(Vec3f), scene->frames.front().vertices, 0, scene->frames.front().vertices.getSize());
    scene->normals.setRange(vertexOffset * sizeof(Vec3f), scene->frames.front().normals, 0, scene->frames.front().normals.getSize());
    scene->texCoords.setRange(vertexOffset * sizeof(Vec2f), texCoords, 0, texCoords.getSize());
    scene->materials.setRange(materialOffset * sizeof(Material), materials, 0, materials.getSize());
    scene->vertIndices.setRange(triangleOffset * sizeof(Vec3i), vertIndices, 0, vertIndices.getSize());
    scene->matIndices.setRange(triangleOffset * sizeof(int), matIndices, 0, matIndices.getSize());

    // Create texture atlas.
    textureFilenames.append(frameTextureFilenames);
    scene->textureAtlas.set(textureFilenames);

    // Load light.
    QVector3D light;
    Environment::getInstance()->getVectorValue("Scene.light", light);
    scene->setLight(Light(light.x(), light.y(), light.z()));

    // Load environment map.
    QString environmentMapFilename;
    Environment::getInstance()->getStringValue("Scene.environment", environmentMapFilename);
    scene->environmentMap.set(environmentMapFilename);

    return scene;

}
