/**
 * \file	ImageExporter.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	ImageExporter class header file.
 */

#ifndef _IMAGE_EXPORTER_H_
#define _IMAGE_EXPORTER_H_

#include <QString>
#include "gpu/Buffer.h"

class ImageExporter {

public:

    ImageExporter(void);

    void exportImage(int width, int height, Buffer & data, const QString & filename, bool hdr = false);

};

#endif /* _IMAGE_EXPORTER_H_ */