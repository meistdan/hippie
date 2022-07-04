/**
 * \file	ImageImporter.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	ImageImporter class header file.
 */

#ifndef _IMAGE_IMPORTER_H_
#define _IMAGE_IMPORTER_H_

#include "gpu/Buffer.h"

//#define MAX_IMAGE_EDGE 512
#define MAX_IMAGE_EDGE 16384

class ImageImporter {

public:

    ImageImporter(void);

    void importImage(int & width, int & height, Buffer & data, const QString & filename, bool hdr = false);

};

#endif /* _IMAGE_IMPORTER_H_ */
