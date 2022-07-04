/**
 * \file	ImageExporter.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	ImageExporter class source file.
 */

#include <IL/il.h>
#include "IL/ilu.h"
#include "ImageImporter.h"
#include "util/Logger.h"

ImageImporter::ImageImporter() {
    ilInit();
    iluInit();
}

void ImageImporter::importImage(int & width, int & height, Buffer & data, const QString & filename, bool hdr) {
    ILuint imageId = ilGenImage();
    ilBindImage(imageId);
    ilLoadImage((const ILstring)filename.toUtf8().constData());
    width = ilGetInteger(IL_IMAGE_WIDTH);
    height = ilGetInteger(IL_IMAGE_HEIGHT);
    if (qMax(width, height) > MAX_IMAGE_EDGE) {
        if (width < height) {
            float scale = float(MAX_IMAGE_EDGE) / height;
            width = MAX_IMAGE_EDGE;
            height = int(height * scale);
        }
        else {
            float scale = float(MAX_IMAGE_EDGE) / width;
            width = int(width * scale);
            height = MAX_IMAGE_EDGE;
        }
        iluScale(width, height, 1);
    }
    if (hdr) {
        data.resizeDiscard(4 * sizeof(float) * width * height);
        ilCopyPixels(0, 0, 0, width, height, 1, IL_RGBA, IL_FLOAT, (void*)data.getMutablePtr());
    }
    else {
        data.resizeDiscard(4 * sizeof(unsigned char) * width * height);
        ilCopyPixels(0, 0, 0, width, height, 1, IL_RGBA, IL_UNSIGNED_BYTE, (void*)data.getMutablePtr());
    }
    ilBindImage(0);
    ilDeleteImages(1, &imageId);
    if (ilGetError() == IL_NO_ERROR)
        logger(LOG_INFO) << "INFO <ImageImporter> Image '" << filename << "' has been successfully imported.\n";
    else
        logger(LOG_ERROR) << "WARN <ImageImporter> Can't import the image '" << filename << "'!\n";
}
