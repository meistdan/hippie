/**
 * \file	ImageImporter.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	ImageImporter class source file.
 */

#include <IL/il.h>
#include "ImageExporter.h"
#include "util/Logger.h"

ImageExporter::ImageExporter() {
    ilInit();
    ilEnable(IL_FILE_OVERWRITE);
}

void ImageExporter::exportImage(int width, int height, Buffer & data, const QString & filename, bool hdr) {
    ILuint imageId = ilGenImage();
    ilBindImage(imageId);
    if (hdr) ilTexImage(width, height, 1, 4, IL_RGBA, IL_FLOAT, (void*)data.getPtr());
    else ilTexImage(width, height, 1, 4, IL_RGBA, IL_UNSIGNED_BYTE, (void*)data.getPtr());
    ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE);
    ilSaveImage((const ILstring)filename.toUtf8().constData());
    ilBindImage(0);
    ilDeleteImages(1, &imageId);
    if (ilGetError() == IL_NO_ERROR)
        logger(LOG_INFO) << "INFO <ImageExporter> Image '" << filename << "' has been successfully exported.\n";
    else
        logger(LOG_WARN) << "WARN <ImageExporter> Can't save the image '" << filename << "'!\n";
}
