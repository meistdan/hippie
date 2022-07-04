/**
 * \file	BVHImporter.h
 * \author	Daniel Meister
 * \date	2016/09/07
 * \brief	BVHImporter class header file.
 */

#ifndef _BVH_IMPORTER_H_
#define _BVH_IMPORTER_H_

#include "HipBVH.h"
#include "util/Logger.h"
#include <QStack>

class BVHImporter {

public:

    virtual void importBVH(HipBVH & cbvh, const QString & filename) = 0;

};

#endif /* _BVH_IMPORTER_H_ */
