/**
  * \file	HipModule.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	HipModule class header file.
  */

#ifndef _HIP_MODULE_H_
#define _HIP_MODULE_H_

#include <QHash>
#include <QString>
#include "HipKernel.h"

class HipModule {

private:

    static bool inited;
    static bool available;
    static hipDevice_t device;
    static hipCtx_t context;
    static hipEvent_t startEvent;
    static hipEvent_t endEvent;

    hipModule_t module;
    QHash<QString, hipFunction_t> kernels;
    QVector<Buffer*> globals;
    QHash<QString, int> globalHash;
    QHash<QString, int> texRefHash;

    static hipDevice_t selectDevice(void);
    static void printDeviceInfo(hipDevice_t device);

    HipModule(const HipModule & cpy); // forbidden
    HipModule & operator=(const HipModule & other); // forbidden
    hipFunction_t findKernel(const QString & name);

public:

    static void staticInit(void);
    static void staticDeinit(void);

    static bool isAvailable(void);
    static long long getMemoryUsed(void);
    static long long getMemoryFree(void);
    static long long getMemoryTotal(void);
    static int getSMCount(void);

    static void sync(bool yield = true); // False = low latency but keeps the CPU busy. True = long latency but relieves the CPU.
    static void checkError(const char* funcName, hipError_t res);

    static hipDevice_t getDeviceHandle(void);
    static int getDriverVersion(void); // e.g. 23 = 2.3
    static int getComputeCapability(void); // e.g. 13 = 1.3
    static hipEvent_t getStartEvent(void);
    static hipEvent_t getEndEvent(void);

    HipModule(const void * cubin);
    HipModule(const QString & cubinFile);
    ~HipModule(void);

    hipModule_t getHandle(void);

    bool hasKernel(const QString & name);
    HipKernel getKernel(const QString & name);

    Buffer & getGlobal(const QString & name);
    void updateGlobals(bool async = false); // copy to the device if modified

};

#endif /* _HIP_MODULE_H_ */