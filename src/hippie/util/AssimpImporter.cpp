/**
 * \file	AssimpImporter.cpp
 * \author	Daniel Meister
 * \date	2014/04/28
 * \brief	AssimpImporter class source file.
 */

#include "AssimpImporter.h"
#include "Logger.h"
#include "rt/scene/Material.h"
#include "util/Math.h"
#include <QDir>
#include <QFile>
#include <QFileInfo>

bool AssimpImporter::importScene(
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

    // Load assimp file.
    const aiScene * assimpScene = nullptr;
    if (QFile(filename).exists()) {
        logger(LOG_INFO) << "INFO <AssimpImporter> Loading scene '" << filename << "'..\n";
        aiPropertyStore * props = aiCreatePropertyStore();
        aiSetImportPropertyFloat(props, AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 90.0f);
        //aiSetImportPropertyInteger(props, AI_CONFIG_PP_SBP_REMOVE,  aiPrimitiveType_POINT | aiPrimitiveType_LINE);
        //aiSetImportPropertyInteger(props, AI_CONFIG_PP_FD_REMOVE,  1);
        assimpScene = aiImportFileExWithProperties(filename.toUtf8().constData(), POSTPROCESS_FLAGS, nullptr, props);
        if (assimpScene) {
            logger(LOG_INFO) << "INFO <AssimpImporter> Scene '" << filename << "' successfully loaded.\n";
        }
        else {
            logger(LOG_WARN) << "WARN <AssimpImporter> Scene '" << filename << "' failed to load.\n";
            return false;
        }
        aiReleasePropertyStore(props);
    }
    else {
        logger(LOG_WARN) << "WARN <AssimpImporter> Scene '" << filename << "' does not exist.\n";
        return false;
    }

    // Compute counts.
    numberOfVertices = 0;
    numberOfTriangles = 0;

    // Cech meshes.
    if (!assimpScene->HasMeshes()) {
        logger(LOG_WARN) << "WARN <AssimpImporter> Scene has to contain some mesh!\n";
        return false;
    }
    logger(LOG_INFO) << "INFO <AssimpImporter> Scene contains " << assimpScene->mNumMeshes << " meshes.\n";

    int numberOfInvalidTriangles = 0;
    for (unsigned int i = 0; i < assimpScene->mNumMeshes; ++i) {
        aiMesh * mesh = assimpScene->mMeshes[i];
        if (mesh->mPrimitiveTypes == aiPrimitiveType_TRIANGLE) {
            numberOfVertices += mesh->mNumVertices;
            for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
                aiFace * face = &mesh->mFaces[j];
                Vec3i t(face->mIndices[0], face->mIndices[1], face->mIndices[2]);
                Vec3f v0(mesh->mVertices[t.x].x, mesh->mVertices[t.x].y, mesh->mVertices[t.x].z);
                Vec3f v1(mesh->mVertices[t.y].x, mesh->mVertices[t.y].y, mesh->mVertices[t.y].z);
                Vec3f v2(mesh->mVertices[t.z].x, mesh->mVertices[t.z].y, mesh->mVertices[t.z].z);
                AABB box;
                box.grow(v0);
                box.grow(v1);
                box.grow(v2);
#if REMOVE_DEGENERATES
                if (box.area() > 0.0f)
                    ++numberOfTriangles;
                else
                    ++numberOfInvalidTriangles;
#else
                if (box.area() == 0.0f)
                    ++numberOfInvalidTriangles;
                ++numberOfTriangles;
#endif
            }
        }
    }

    // Triangles.
    logger(LOG_INFO) << "INFO <AssimpImporter> Scene contains " << numberOfTriangles << " triangles.\n";
    logger(LOG_INFO) << "INFO <AssimpImporter> Scene contains " << numberOfInvalidTriangles << " invalid triangles.\n";

    // Vertices and normals.
    boundingBox.reset();
    vertices.resizeDiscard(sizeof(Vec3f) * numberOfVertices);
    normals.resizeDiscard(sizeof(Vec3f) * numberOfVertices);
    Vec3f * vertexPtr = (Vec3f*)vertices.getMutablePtr();
    Vec3f * normalPtr = (Vec3f*)normals.getMutablePtr();
    for (unsigned int i = 0; i < assimpScene->mNumMeshes; ++i) {
        aiMesh * mesh = assimpScene->mMeshes[i];
        if (mesh->mPrimitiveTypes == aiPrimitiveType_TRIANGLE) {
            for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
                vertexPtr->x = mesh->mVertices[j].x;
                //boundingBox.grow(*vertexPtr);
                vertexPtr->y = mesh->mVertices[j].y;
                //boundingBox.grow(*vertexPtr);
                vertexPtr->z = mesh->mVertices[j].z;
                boundingBox.grow(*vertexPtr);
                ++vertexPtr;
                if (mesh->HasNormals()) {
                    normalPtr->x = mesh->mNormals[j].x;
                    normalPtr->y = mesh->mNormals[j].y;
                    normalPtr->z = mesh->mNormals[j].z;
                    ++normalPtr;
                }
            }
        }
    }

    // Triangle vertex indices.
    //boundingBox.reset();
    int vertexOffset = 0;
    vertIndices.resizeDiscard(sizeof(Vec3i) * numberOfTriangles);
    Vec3i * indexPtr = (Vec3i*)vertIndices.getMutablePtr();
    for (unsigned int i = 0; i < assimpScene->mNumMeshes; ++i) {
        aiMesh * mesh = assimpScene->mMeshes[i];
        if (mesh->mPrimitiveTypes == aiPrimitiveType_TRIANGLE) {
            for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
                aiFace * face = &mesh->mFaces[j];
                Vec3i t(face->mIndices[0], face->mIndices[1], face->mIndices[2]);
                Vec3f v0(mesh->mVertices[t.x].x, mesh->mVertices[t.x].y, mesh->mVertices[t.x].z);
                Vec3f v1(mesh->mVertices[t.y].x, mesh->mVertices[t.y].y, mesh->mVertices[t.y].z);
                Vec3f v2(mesh->mVertices[t.z].x, mesh->mVertices[t.z].y, mesh->mVertices[t.z].z);
                AABB box;
                box.grow(v0);
                box.grow(v1);
                box.grow(v2);
#if REMOVE_DEGENERATES
                if (box.area() > 0.0f) {
                    //boundingBox.grow(box);
                    indexPtr->x = face->mIndices[0] + vertexOffset;
                    indexPtr->y = face->mIndices[1] + vertexOffset;
                    indexPtr->z = face->mIndices[2] + vertexOffset;
                    ++indexPtr;
                }
#else
                //boundingBox.grow(box);
                indexPtr->x = face->mIndices[0] + vertexOffset;
                indexPtr->y = face->mIndices[1] + vertexOffset;
                indexPtr->z = face->mIndices[2] + vertexOffset;
                ++indexPtr;
#endif
            }
            vertexOffset += mesh->mNumVertices;
        }
    }

    // Material indices.
    matIndices.resizeDiscard(sizeof(int) * numberOfTriangles);
    if (assimpScene->HasMaterials()) {
        int * matIndexPtr = (int*)matIndices.getMutablePtr();
        for (unsigned int i = 0; i < assimpScene->mNumMeshes; ++i) {
            aiMesh * mesh = assimpScene->mMeshes[i];
            if (mesh->mPrimitiveTypes == aiPrimitiveType_TRIANGLE) {
                for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
                    aiFace * face = &mesh->mFaces[j];
                    Vec3i t(face->mIndices[0], face->mIndices[1], face->mIndices[2]);
                    Vec3f v0(mesh->mVertices[t.x].x, mesh->mVertices[t.x].y, mesh->mVertices[t.x].z);
                    Vec3f v1(mesh->mVertices[t.y].x, mesh->mVertices[t.y].y, mesh->mVertices[t.y].z);
                    Vec3f v2(mesh->mVertices[t.z].x, mesh->mVertices[t.z].y, mesh->mVertices[t.z].z);
                    AABB box;
                    box.grow(v0);
                    box.grow(v1);
                    box.grow(v2);
#if REMOVE_DEGENERATES
                    if (box.area() > 0.0f) *matIndexPtr++ = mesh->mMaterialIndex;
#else
                    *matIndexPtr++ = mesh->mMaterialIndex;
#endif
                }
            }
        }
    }
    else {
        matIndices.clear();
    }

    // Texture coordinates.
    texCoords.resizeDiscard(sizeof(Vec2f) * numberOfVertices);
    if (assimpScene->HasMaterials()) {
        Vec2f * texCoordPtr = ((Vec2f*)texCoords.getMutablePtr());
        for (unsigned int i = 0; i < assimpScene->mNumMeshes; ++i) {
            aiMesh * mesh = assimpScene->mMeshes[i];
            if (mesh->mPrimitiveTypes == aiPrimitiveType_TRIANGLE) {
                if (mesh->HasTextureCoords(0)) {
                    for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
                        texCoordPtr->x = mesh->mTextureCoords[0][j].x;
                        texCoordPtr->y = mesh->mTextureCoords[0][j].y;
                        ++texCoordPtr;
                    }
                }
            }
        }
    }

    // Materials.
    QString filepath = QFileInfo(QFile(filename)).absolutePath();
    numberOfMaterials = assimpScene->mNumMaterials;
    textureFilenames.clear();
    materials.resizeDiscard(sizeof(Material) * numberOfMaterials);
    if (assimpScene->HasMaterials()) {
        Material * materialPtr = ((Material*)materials.getMutablePtr());
        Material m;
        aiColor4D c;
        aiString texFilename;

        for (unsigned int i = 0; i < assimpScene->mNumMaterials; ++i) {
            aiMaterial * mat = assimpScene->mMaterials[i];
            if (aiReturn_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &c)) m.diffuse = Vec3f(c.r, c.g, c.b);
            else m.diffuse = Vec3f(0.8f);
            if (aiReturn_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_SPECULAR, &c)) m.specular = Vec3f(c.r, c.g, c.b);
            else m.specular = Vec3f(0.0f);
            // Note: Assimp multiplies shininess by 4.
            if (aiReturn_SUCCESS != mat->Get(AI_MATKEY_SHININESS, m.shininess)) m.shininess = 0.0f;
            if (aiReturn_SUCCESS == assimpScene->mMaterials[i]->GetTexture(aiTextureType_DIFFUSE, 0, &texFilename)) {
                m.texIndex = textureFilenames.size();
                textureFilenames.push_back(filepath + "/" + texFilename.C_Str());
            }
            else {
                m.texIndex = -1;
            }
            *materialPtr++ = m;
        }
    }
    else {
        materials.resizeDiscard(sizeof(Material));
        Material m;
        m.diffuse = Vec3f(0.8f);
        m.specular = Vec3f(0.0f);
        m.texIndex = -1;
        *(Material*)materials.getMutablePtr() = m;
    }

    // Release scene.
    aiReleaseImport(assimpScene);

    return true;

}
