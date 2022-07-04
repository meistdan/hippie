#include "EnvironmentMap.h"
#include "util/ImageImporter.h"
#include "util/Logger.h"
#include <QFile>

EnvironmentMap::EnvironmentMap(void) {
    size = Vec2i(-1);
}

EnvironmentMap::~EnvironmentMap() {
    texture.clear();
}

void EnvironmentMap::set(const QString & filename) {
    Buffer dataBuffer;
    QFile file(filename);
    if (file.exists()) {
        ImageImporter().importImage(size.x, size.y, dataBuffer, filename);
    }
    else {
        //const Vec3f BACKGROUND_COLOR = Vec3f(0.52f, 0.69f, 1.0f);
        const Vec3f BACKGROUND_COLOR = Vec3f(1.0f);
        size = Vec2f(1);
        dataBuffer.resizeDiscard(sizeof(int));
        *(int*)dataBuffer.getMutablePtr() = floatToByte(BACKGROUND_COLOR);
    }
    texture.set(dataBuffer, size.x, size.y, true);
}

void EnvironmentMap::clear() {
    size = Vec2i(-1);
    texture.clear();
}

const Vec2i & EnvironmentMap::getSize() const {
    return size;
}

hipTextureObject_t EnvironmentMap::getTextureObject() {
    return texture.getTextureObject();
}
