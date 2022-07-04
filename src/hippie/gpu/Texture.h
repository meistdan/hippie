/**
 * \file	Texture.h
 * \author	Daniel Meister
 * \date	2021/12/13
 * \brief	Texture class header file.
 */

#ifndef _TEXTURE_H_
#define _TEXTURE_H_

#include "Buffer.h"

class Texture {

private:

    int width;
    int height;

    hipTextureObject_t textureObject;
    hipArray_t array;
    hipChannelFormatDesc channelDesc;

public:

    Texture(void);
    ~Texture(void);

    void set(Buffer& buf, int width, int height, bool normalizedCoords);
    void clear(void);

    int getWidth(void);
    int getHeight(void);

    hipTextureObject_t getTextureObject(void);
    hipArray_t getHipArray(void);
    hipChannelFormatDesc getChannelDesc(void);
    
};

#endif /* _TEXTURE_H_ */
