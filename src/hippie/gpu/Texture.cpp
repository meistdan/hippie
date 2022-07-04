/**
 * \file	Texture.cpp
 * \author	Daniel Meister
 * \date	2021/12/13
 * \brief	Texture class source file.
 */

#include "Texture.h"
#include "HipModule.h"
#include "util/Logger.h"

Texture::Texture() : textureObject(0), array(nullptr), width(-1), height(-1) {}

Texture::~Texture() {
    clear();
}

void Texture::set(Buffer& buf, int width, int height, bool normalizedCoords) {
    clear();
    HipModule::staticInit();
    this->channelDesc = hipCreateChannelDesc(8, 8, 8, 8, hipChannelFormatKindUnsigned);
    this->width = width;
    this->height = height;
    HipModule::checkError("hipMallocArray(", hipMallocArray(&array, &channelDesc, width, height));
    HipModule::checkError("hipMemcpyToArray", hipMemcpyToArray(array, 0, 0, buf.getPtr(), buf.getSize(), hipMemcpyHostToDevice));
    hipResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = hipResourceTypeArray;
    resDesc.res.array.array = array;
    hipTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = hipReadModeNormalizedFloat;
    texDesc.normalizedCoords = normalizedCoords;
    texDesc.filterMode = hipFilterModeLinear;
    texDesc.addressMode[0] = hipAddressModeWrap;
    texDesc.addressMode[1] = hipAddressModeWrap;
    HipModule::checkError("hipCreateTextureObject", hipCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr));
}

void Texture::clear() {
    if (textureObject) {
        HipModule::checkError("hipDestroyTextureObject;", hipDestroyTextureObject(textureObject));
        textureObject = 0;
    }
    if (array) {
        HipModule::checkError("hipFreeArray;", hipFreeArray(array));
        array = nullptr;
    }
}

hipTextureObject_t Texture::getTextureObject() {
    return textureObject;
}

hipArray_t Texture::getHipArray() {
    return array;
}

int Texture::getWidth() {
    return width;
}

int Texture::getHeight() {
    return height;
}

hipChannelFormatDesc Texture::getChannelDesc() {
    return channelDesc;
}
