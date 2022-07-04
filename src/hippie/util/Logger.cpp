/**
 * \file	Logger.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Logger class source file.
 */

#include "Logger.h"

Logger logger;

Logger::Logger() {
}

Logger::Logger(const QString & outFilename, bool append) {
    setOut(outFilename, append);
}

Logger::~Logger() {
    if (outFile.isOpen()) outFile.close();
    if (errFile.isOpen()) errFile.close();
}
