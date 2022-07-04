/**
 * \file	CameraLoader.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	CameraLoader class source file.
 */

#include "CameraLoader.h"
#include "util/Logger.h"
#include <QFile>
#include <QFileInfo>
#include <QStringlist>
#include <QTextStream>

void CameraLoader::loadFrames(const QString & filename, Camera & camera) {

    QFile file(filename);

    if (!file.exists()) {
        logger(LOG_WARN) << "WARN <CameraLoader> File '" << filename << "' does not exist.\n";
        return;
    }

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        logger(LOG_WARN) << "WARN <CameraLoader> Cannot open file '" << filename << "'.\n";
        return;
    }

    QTextStream in(&file);
    int lineNumber = 0;

    while (!in.atEnd()) {

        // Read line.
        QString line = in.readLine();
        ++lineNumber;

        // Get rid of comments.
        int commentIndex = line.indexOf("#");
        if (commentIndex != -1) {
            line = line.mid(0, commentIndex);
        }

        // Skip empty lines.
        if (line.isEmpty()) continue;

        // Split line.
        QStringList words = line.split(QRegExp("(\\ |\\n|\\t)"), QString::SkipEmptyParts);

        if (words.size() != 6) {
            logger(LOG_WARN) << "WARN <CameraLoader> Invalid camera data at line " << lineNumber << ".\n";
            return;
        }

        // Load camera frame.
        bool ok;
        float data[6];
        for (int i = 0; i < 6; ++i) {
            data[i] = words[i].toFloat(&ok);
            if (!ok) {
                logger(LOG_WARN) << "WARN <CameraLoader> Invalid camera data at line " << lineNumber << ".\n";
                return;
            }
        }

        // Insert loaded frame.
        glm::vec3 position = glm::vec3(data[0], data[1], data[2]);
        glm::vec3 direction = glm::vec3(data[3], data[4], data[5]);
        camera.frames.push_back(CameraFrame(position, direction));

    }

}
