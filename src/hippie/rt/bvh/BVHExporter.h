/**
 * \file	BVHExporter.h
 * \author	Daniel Meister
 * \date	2016/09/07
 * \brief	BVHExporter class header file.
 */

#ifndef _BVH_EXPORTER_H_
#define _BVH_EXPORTER_H_

#include "rt/bvh/HipBVH.h"

class BVHExporter {

public:

    virtual void exportBVH(HipBVH & cbvh, const QString & filename) = 0;

};

#endif /* _BVH_EXPORTER_H_ */
