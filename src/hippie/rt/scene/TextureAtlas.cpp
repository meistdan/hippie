/**
 * \file	TextureAtlas.cpp
 * \author	Daniel Meister
 * \date	2014/04/23
 * \brief	SceneLoader class source file.
 */

#include "TextureAtlas.h"
#include "util/ImageImporter.h"
#include "util/Logger.h"
#include <QStringList>
#include <QElapsedTimer>

static bool compare(const Vec2i & a, const Vec2i & b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

TextureAtlas::TextureAtlas() {
    size = Vec2i(-1);
}

TextureAtlas::~TextureAtlas() {
    texture.clear();
}

void TextureAtlas::set(const QStringList & textureFilenames) {

    // Timer.
    QElapsedTimer timer;
    timer.start();

    // Clear.
    clear();

    // Load textures.
    ImageImporter imgImporter;
    QVector<Buffer> textures(textureFilenames.size());
    textureItems.resize(textureFilenames.size());
    textures.resize(textureFilenames.size());
    for (int i = 0; i < textureFilenames.size(); ++i)
        imgImporter.importImage(textureItems[i].x, textureItems[i].y, textures[i], textureFilenames[i]);

    // Select initial size.
    int totalArea = 0;
    for (int i = 0; i < textureItems.size(); ++i)
        totalArea += textureItems[i].x * textureItems[i].y;
    Vec2i canvas = Vec2i((int)sqrt((float)totalArea));

    // Sort items in descending order of height.
    QVector<Vec2i> order;
    order.resize(textures.size());
    for (int i = 0; i < textures.size(); i++)
        order[i] = Vec2i(-textureItems[i].y, i);
    qSort(order.begin(), order.end(), compare);

    // Add items iteratively.
    QVector<Vec4i> maxRects;
    maxRects.push_back(Vec4i(0, 0, MAX_INT, MAX_INT));
    for (int t = 0; t < textures.size(); ++t) {

        // Texture size.
        TextureItem & textureItem = textureItems[order[t].y];
        Vec2i s = Vec2i(textureItem.x, textureItem.y);

        // Select position.
        Vec2i bestPos;
        Vec3i bestCost = Vec3i(MAX_INT);
        for (int i = 0; i < maxRects.size(); ++i) {
            const Vec4i & r = maxRects[i];
            if (r.x + s.x > r.z || r.y + s.y > r.w)
                continue;
            Vec3i cost;
            cost.x = qMax(canvas.x, r.x + s.x) * qMax(canvas.y, r.y + s.y); // Minimize canvas.
            cost.y = r.y + s.y; // Minimize Y.
            cost.z = r.x + s.x; // Minimize X.

            if (cost.x < bestCost.x || (cost.x == bestCost.x && (cost.y < bestCost.y || (cost.y == bestCost.y && cost.z < bestCost.z))))
            {
                bestPos = Vec2i(r);
                bestCost = cost;
            }
        }

        textureItem.z = bestPos.x;
        textureItem.w = bestPos.y;
        canvas = max(canvas, bestPos + s);
        Vec4i q(bestPos, bestPos + s);

        // Split maximal rectangles.
        for (int i = maxRects.size() - 1; i >= 0; --i) {
            Vec4i r = maxRects[i];
            if (q.x >= r.z || q.y >= r.w || q.z <= r.x || q.w <= r.y)
                continue;
            maxRects[i] = maxRects.last();
            maxRects.pop_back();
            if (q.x > r.x) maxRects.push_back(Vec4i(r.x, r.y, q.x, r.w));
            if (q.y > r.y) maxRects.push_back(Vec4i(r.x, r.y, r.z, q.y));
            if (q.z < r.z) maxRects.push_back(Vec4i(q.z, r.y, r.z, r.w));
            if (q.w < r.w) maxRects.push_back(Vec4i(r.x, q.w, r.z, r.w));
        }

        // Remove duplicates.
        for (int i = maxRects.size() - 1; i >= 0; --i) {
            const Vec4i & a = maxRects[i];
            for (int j = 0; j < maxRects.size(); ++j) {
                const Vec4i & b = maxRects[j];
                if (i != j && a.x >= b.x && a.y >= b.y && a.z <= b.z && a.w <= b.w) {
                    maxRects[i] = maxRects.last();
                    maxRects.pop_back();
                    break;
                }
            }
        }
    }

    // Determine final size.
    size = Vec2i(1);
    for (int i = 0; i < textureItems.size(); ++i)
        size = max(size, Vec2i(textureItems[i].z + textureItems[i].x, textureItems[i].w + textureItems[i].y));

    // Load data.
    Q_ASSERT(sizeof(int) == 4);
    Buffer atlasBuffer;
    atlasBuffer.resizeDiscard(sizeof(int) * size.x * size.y);
    int * atlasPtr = (int*)atlasBuffer.getMutablePtr();
    for (int t = 0; t < textures.size(); ++t) {
        int * texturePtr = (int*)textures[t].getPtr();
        for (int i = 0; i < textureItems[t].y; ++i) {
            for (int j = 0; j < textureItems[t].x; ++j) {
                int textureIndex = textureItems[t].x * i + j;
                int dataIndex = size.x * (i + textureItems[t].w) + (j + textureItems[t].z);
                atlasPtr[dataIndex] = texturePtr[textureIndex];
            }
        }
    }

    // Set the texture object
    texture.set(atlasBuffer, size.x, size.y, false);

    // Log elapsed time.
    logger(LOG_INFO) << "INFO <TextureAtlas> Texture atlas computed in " << (1.0e-3f * timer.elapsed()) << "s.\n";

}

void TextureAtlas::clear() {
    size = Vec2i(-1);
    texture.clear();
    textureItems.clear();
}

const Vec2i & TextureAtlas::getSize() const {
    return size;
}

const QVector<TextureItem> & TextureAtlas::getTextureItems() const {
    return textureItems;
}

int TextureAtlas::getNumberOfTextures() const {
    return textureItems.size();
}


hipTextureObject_t TextureAtlas::getTextureObject() {
    return texture.getTextureObject();
}
