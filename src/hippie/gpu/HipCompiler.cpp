 /**
  * \file	HipCompiler.cpp
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	HipCompiler class source file.
  */

#include <QDir>
#include <QFile>
#include <hip/hiprtc.h>
#include "util/Hash.h"
#include "util/Logger.h"
#include "HipCompiler.h"

QHash<unsigned long long, HipModule*>  HipCompiler::moduleCache;

QString HipCompiler::concatOptions(const QStringList& options) {
    QString result;
    for (auto option : options) {
        result.append(option);
        result.append(" ");
    }
    return result;
}

QString HipCompiler::getSource(const QString& filename) {
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        logger(LOG_ERROR) << "ERROR <HipCompiler> Cannot open '" << filename << "'.\n";
        exit(EXIT_FAILURE);
    }
    QString result;
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        result.append(line + "\n");
    }
    file.close();
    return result;
}

unsigned long long HipCompiler::getMemHash() {

    if (memHashValid) return memHash;

    if (!sourceHashValid) {
        sourceHash = hashString(sourceFile);
        sourceHashValid = true;
    }

    if (!optionHashValid) {
        optionHash = hashString(concatOptions(options));
        optionHashValid = true;
    }

    unsigned int a = HASH_MAGIC + sourceHash;
    unsigned int b = HASH_MAGIC + optionHash;
    unsigned int c = HASH_MAGIC;

    JENKINS_MIX(a, b, c);
    memHash = ((unsigned long long) b << 32) | c;
    memHashValid = true;

    return memHash;
}

HipCompiler::HipCompiler() :
    sourceHash(0),
    optionHash(0),
    memHash(0),
    sourceHashValid(false),
    optionHashValid(false),
    memHashValid(false)
{}

HipCompiler::~HipCompiler() {}

void HipCompiler::setSourceFile(const QString & path) {
    sourceFile = path;
    sourceHashValid = false;
    memHashValid = false;
}

void HipCompiler::clearOptions() {
    options.clear();
    optionHashValid = false;
    memHashValid = false;
}

void HipCompiler::addOption(const QString & option) {
    options.append(option);
    optionHashValid = false;
    memHashValid = false;
}

void HipCompiler::removeOption(const QString& option) {
    for (int i = 0; i < options.size(); ++i) {
        if (QString(options[i]) == option)
            options.removeAt(i);
    }
    optionHashValid = false;
    memHashValid = false;
}

void HipCompiler::include(const QString & path) {
    addOption("-I " + path);
}

HipModule * HipCompiler::compile() {

    unsigned long long memHash = getMemHash();
    QHash<unsigned long long, HipModule*>::iterator found = moduleCache.find(memHash);
    if (found != moduleCache.end()) return found.value();

    logger(LOG_INFO) << "INFO <HipCompiler> Compiling '" << sourceFile << "' ...\n";
    QString code = getSource(sourceFile);
    hiprtcProgram prog;
    hiprtcResult e;
    e = hiprtcCreateProgram(&prog, code.toUtf8().constData(), sourceFile.toUtf8().constData(), 0, 0, 0);
    if (e != HIPRTC_SUCCESS) {
        logger(LOG_ERROR) << "ERROR <HipCompiler> Cannot create program from '" << sourceFile << "'.\n";
        logger(LOG_ERROR) << hiprtcGetErrorString(e) << "\n";
        exit(EXIT_FAILURE);
    }

    QVector<const char*> opts;
    QVector<QByteArray> optStrings;
#if (defined(__HIP_PLATFORM_NVCC__)||defined(__HIP_PLATFORM_NVIDIA__))
    optStrings.push_back("-use_fast_math");
#endif
    optStrings.push_back("-std=c++11");
    optStrings.push_back("-D __USE_HIP__");
    optStrings.push_back("-I ../src/hippie");
    for (auto option : options) optStrings.push_back(option.toLocal8Bit());
    for (QByteArray & optString : optStrings) opts.push_back(optString.data());
    
    e = hiprtcCompileProgram(prog, opts.size(), opts.data());
    if (e != HIPRTC_SUCCESS) {
        logger(LOG_ERROR) << "ERROR <HipCompiler> Compilation '" << sourceFile << "' failed.\n";
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);
        if (logSize) {
            QVector<char> log(logSize);
            hiprtcGetProgramLog(prog, log.data());
            logger(LOG_ERROR) << QString(log.data()) << "\n";
        }
        exit(EXIT_FAILURE);
    }
    size_t codeSize;
    hiprtcGetCodeSize(prog, &codeSize);

    QByteArray binary;;
    binary.resize(codeSize);
    hiprtcGetCode(prog, binary.data());
    hiprtcDestroyProgram(&prog);
    logger(LOG_INFO) << "INFO <HipCompiler> Done.\n";
    
    HipModule* module = new HipModule(binary.data());
    moduleCache.insert(memHash, module);
    return module;

}
