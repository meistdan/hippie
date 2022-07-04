/**
 * \file	Logger.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Logger class header file.
 */

#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <QFile>
#include <QTextStream>

enum {
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
};

class Logger {

private:

    QFile outFile;
    QFile errFile;

    QTextStream out;
    QTextStream err;

    int level;

public:

    Logger(void);
    Logger(const QString & outFilename, bool append = false);
    ~Logger(void);

    inline void setOut(const QString & filename, bool append = false) {
        if (outFile.isOpen()) outFile.close();
        outFile.setFileName(filename);
        if (append) outFile.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append);
        else outFile.open(QIODevice::WriteOnly | QIODevice::Text);
        out.setDevice(&outFile);
    }

    inline void setErr(const QString & filename, bool append = false) {
        if (errFile.isOpen()) errFile.close();
        errFile.setFileName(filename);
        if (append) errFile.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append);
        else errFile.open(QIODevice::WriteOnly | QIODevice::Text);
        err.setDevice(&errFile);
    }

    inline int getLevel() {
        return level;
    }

    inline void setLevel(int level) {
        this->level = level;
    }

    inline Logger & operator()(int level) {
        this->level = level;
        return *this;
    }

    template<class T>
    inline Logger & operator<<(T msg) {
        if (level == LOG_ERROR) {
            err << msg;
            err.flush();
        }
        else {
            out << msg;
            out.flush();
        }
        return *this;
    }

};

extern Logger logger;

#endif /* _LOGGER_H_ */
