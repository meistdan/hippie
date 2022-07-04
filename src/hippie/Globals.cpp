/**
 * \file	Globals.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Implementation of some common useful functions.
 */

#include "util/Logger.h"
#include "Globals.h"

void checkGLErrors() {

    GLenum err = glGetError();
    const char* name;
    switch (err) {
    case GL_NO_ERROR: name = nullptr; break;
    case GL_INVALID_ENUM: name = "GL_INVALID_ENUM"; break;
    case GL_INVALID_VALUE: name = "GL_INVALID_VALUE"; break;
    case GL_INVALID_OPERATION: name = "GL_INVALID_OPERATION"; break;
    case GL_STACK_OVERFLOW: name = "GL_STACK_OVERFLOW"; break;
    case GL_STACK_UNDERFLOW: name = "GL_STACK_UNDERFLOW"; break;
    case GL_OUT_OF_MEMORY: name = "GL_OUT_OF_MEMORY"; break;
    case GL_INVALID_FRAMEBUFFER_OPERATION: name = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
    default: name = "unknown"; break;
    }

    if (name) {
        logger(LOG_ERROR) << "ERROR <Globals> Caught GL error '" << name << "'!\n";
        exit(EXIT_FAILURE);
    }
}
