/**
 * \file	TextureAtlas.h
 * \author	Daniel Meister
 * \date	2014/04/23
 * \brief	TextureAtlas class header file.
 */

#ifndef _TEXTURE_ATLAS_H_
#define _TEXTURE_ATLAS_H_

#include "gpu/Texture.h"
#include "util/Math.h"

typedef Vec4i TextureItem;

class TextureAtlas {

private:

    Vec2i size;
    Texture texture;
    QVector<TextureItem> textureItems;

public:

    TextureAtlas(void);
    ~TextureAtlas(void);

    void set(const QStringList & textureFilenames);
    void clear(void);

    const Vec2i & getSize(void) const;
    const QVector<TextureItem> & getTextureItems(void) const;
    int getNumberOfTextures(void) const;
    hipTextureObject_t getTextureObject(void);

    friend class SceneLoader;

};

#endif /* _TEXTURE_ATLAS_H_ */
