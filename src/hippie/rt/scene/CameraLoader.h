/**
 * \file	CameraLoader.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	CameraLoader class header file.
 */

#ifndef _CAMERA_LOADER_H_
#define _CAMERA_LOADER_H_

#include "Camera.h"
#include <QString>

class CameraLoader {

public:

    void loadFrames(const QString & filename, Camera & camera);

};

#endif /* _CAMERA_LOADER_H_ */
