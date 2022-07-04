 /**
  * \file	HipCompiler.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	HipCompiler class header file.
  */

#ifndef _HIP_COMPILER_H_	
#define _HIP_COMPILER_H_

#include "HipModule.h"

class HipCompiler {

private:

    static QHash<unsigned long long, HipModule*> moduleCache;

    QString cachePath;
    QString sourceFile;
    QStringList options;

    unsigned int sourceHash;
    unsigned int optionHash;
    unsigned long long memHash;
    bool sourceHashValid;
    bool optionHashValid;
    bool memHashValid;

    static QString concatOptions(const QStringList& options);
    static QString getSource(const QString & filename);

    HipCompiler(const HipCompiler &); // forbidden
    HipCompiler & operator=(const HipCompiler &); // forbidden

    unsigned long long getMemHash(void);

public:

    HipCompiler(void);
    ~HipCompiler(void);

    void setSourceFile(const QString & path);

    void clearOptions(void);
    void addOption(const QString & option);
    void removeOption(const QString& option);
    void include(const QString & path);

    HipModule * compile(void);

};

#endif /* _HIP_COMPILER_H_ */
