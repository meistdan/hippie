/**
  * \file	HipModule.cpp
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	HipModule class source file.
  */

#include <QThread>
#include "util/Logger.h"
#include "HipModule.h"

bool HipModule::inited = false;
bool HipModule::available = false;
hipDevice_t HipModule::device = 0;
hipCtx_t HipModule::context = nullptr;
hipEvent_t HipModule::startEvent = nullptr;
hipEvent_t HipModule::endEvent = nullptr;

hipDevice_t HipModule::selectDevice() {

    int numDevices;
    hipDevice_t device = 0;
    int bestScore = MIN_INT;

    checkError("hipGetDeviceCount", hipGetDeviceCount(&numDevices));

    for (int i = 0; i < numDevices; i++) {

        hipDevice_t dev;
        checkError("hipDeviceGet", hipDeviceGet(&dev, i));

        int clockRate;
        int numProcessors;
        checkError("hipDeviceGetAttribute", hipDeviceGetAttribute(&clockRate, hipDeviceAttributeClockRate, dev));
        checkError("hipDeviceGetAttribute", hipDeviceGetAttribute(&numProcessors, hipDeviceAttributeMultiprocessorCount, dev));

        int score = clockRate * numProcessors;
        if (score > bestScore) {
            device = dev;
            bestScore = score;
        }
    }

    if (bestScore == MIN_INT) {
        logger(LOG_ERROR) << "ERROR <HipModule> No appropriate HIP device found.\n";
        exit(EXIT_FAILURE);
    }

    return device;
}

void HipModule::printDeviceInfo(hipDevice_t device) {

    static const struct {
        hipDeviceAttribute_t attrib;
        const char *         name;
    } attribs[] = {
        { hipDeviceAttributeClockRate,                          "Clock rate" },
        { hipDeviceAttributeMemoryClockRate,                    "Memory clock rate" },
        { hipDeviceAttributeMultiprocessorCount,                "Number of SMs" },
        { hipDeviceAttributeMaxThreadsPerBlock,                 "Max threads per block" },
        { hipDeviceAttributeMaxThreadsPerMultiProcessor,        "Max threads per SM" },
        { hipDeviceAttributeMaxRegistersPerBlock,               "Max registers per block" },
        { hipDeviceAttributeMaxSharedMemoryPerBlock,            "Max shared mem per block" },
        { hipDeviceAttributeTotalConstantMemory,                "Constant memory" },
        { hipDeviceAttributeMaxBlockDimX,                       "Max blockDim.x" },
        { hipDeviceAttributeMaxGridDimX,                        "Max gridDim.x" },
        { hipDeviceAttributeConcurrentKernels,                  "Concurrent launches supported" },
        { hipDeviceAttributeDeviceOverlap,                      "Concurrent memcopy supported" },
        { hipDeviceAttributeAsyncEngineCount,                   "Max concurrent memcopies" },
        { hipDeviceAttributeUnifiedAddressing,                  "Unified addressing supported" },
        { hipDeviceAttributeCanMapHostMemory,                   "Can map host memory" },
        { hipDeviceAttributeEccEnabled,                         "ECC enabled" }
    };

    const unsigned int MAX_NAME_SIZE = 256;
    char name[MAX_NAME_SIZE];
    int major, minor;
    size_t memory;

    checkError("hipDeviceGetName", hipDeviceGetName(name, MAX_NAME_SIZE - 1, device));
    checkError("hipDeviceComputeCapability", hipDeviceComputeCapability(&major, &minor, device));
    checkError("hipDeviceTotalMem", hipDeviceTotalMem(&memory, device));
    name[MAX_NAME_SIZE - 1] = '\0';

    int bits = 107 << 23;
    float memInMBs = (float)memory * *(float*)&bits;

    logger(LOG_INFO) << "INFO <HipModule> Device details.\n";
    logger(LOG_INFO) << "\tHIP device: " << device << " " << name << "\n";
    logger(LOG_INFO) << "\tCompute capability: " << major << "." << minor << "\n";
    logger(LOG_INFO) << "\tTotal memory: " << memInMBs << " MB\n";

    unsigned int numAttribs = sizeof(attribs) / sizeof(attribs[0]);
    for (unsigned int i = 0; i < numAttribs; i++) {
        int value;
        if (hipDeviceGetAttribute(&value, attribs[i].attrib, device) == hipSuccess)
            logger(LOG_INFO) << "\t" << attribs[i].name << ": " << value << "\n";
    }

}

hipFunction_t HipModule::findKernel(const QString & name) {

    // Search from hash.
    QHash<QString, hipFunction_t>::Iterator found = kernels.find(name);
    if (found != kernels.end()) return found.value();

    // Search from module.
    hipFunction_t kernel = nullptr;
    hipModuleGetFunction(&kernel, module, name.toUtf8().constData());
    if (!kernel) hipModuleGetFunction(&kernel, module, (QString("__globfunc_") + name).toUtf8().constData());
    if (!kernel) return nullptr;

    // Add to hash.
    kernels.insert(name, kernel);

    return kernel;
}

void HipModule::staticInit() {

    if (inited) return;
    inited = true;
    available = false;

    hipError_t res = hipInit(0);
    if (res != hipSuccess) {
        if (res != hipErrorNoDevice)
            checkError("hipInit", res);
        return;
    }

    available = true;
    device = selectDevice();
    printDeviceInfo(device);

    unsigned int flags = 0;
    flags |= hipDeviceScheduleSpin; // use sync() if you want to yield
    flags |= hipDeviceLmemResizeToMax; // reduce launch overhead with large localmem

    // Create HIP contex.
    checkError("hipCtxCreate", hipCtxCreate(&context, flags, device));

    // Create events.
    checkError("hipEventCreate", hipEventCreate(&startEvent));
    checkError("hipEventCreate", hipEventCreate(&endEvent));

}

void HipModule::staticDeinit() {
    if (!inited) return;
    inited = false;
    if (startEvent) checkError("hipEventDestroy", hipEventDestroy(startEvent));
    startEvent = nullptr;
    if (endEvent) checkError("hipEventDestroy", hipEventDestroy(endEvent));
    endEvent = nullptr;
    if (context) checkError("hipCtxDestroy", hipCtxDestroy(context));
    context = nullptr;
    device = 0;
}

bool HipModule::isAvailable() {
    staticInit();
    return available;
}

long long HipModule::getMemoryUsed() {
    staticInit();
    if (!available) return 0;
    size_t free = 0, total = 0;
    checkError("hipMemGetInfo", hipMemGetInfo(&free, &total));
    return total - free;
}

long long HipModule::getMemoryFree() {
    staticInit();
    if (!available) return 0;
    size_t free = 0, total = 0;
    checkError("hipMemGetInfo", hipMemGetInfo(&free, &total));
    return free;
}

long long HipModule::getMemoryTotal() {
    staticInit();
    if (!available) return 0;
    size_t free = 0, total = 0;
    checkError("hipMemGetInfo", hipMemGetInfo(&free, &total));
    return total;
}

int HipModule::getSMCount() {
    int smCount;
    checkError("hipDeviceGetAttribute", hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
    return smCount;
}

void HipModule::sync(bool yield) {
    if (!inited) return;
    if (!yield || !endEvent) {
        checkError("hipDeviceSynchronize", hipDeviceSynchronize());
        return;
    }

    checkError("hipEventRecord", hipEventRecord(endEvent, nullptr));
    for (;;) {
        hipError_t res = hipEventQuery(endEvent);
        if (res != hipErrorNotReady) {
            checkError("hipEventQuery", res);
            break;
        }
        QThread::yieldCurrentThread();
    }
}

void HipModule::checkError(const char * funcName, hipError_t res) {
    if (res != hipSuccess) {
        logger(LOG_ERROR) << "ERROR <HipModule> " << funcName << "() failed: " << hipGetErrorString(res) << "\n";
        Q_ASSERT(false);
        exit(EXIT_FAILURE);
    }
}


hipDevice_t HipModule::getDeviceHandle() {
    staticInit();
    return device;
}

int HipModule::getDriverVersion() {
    int version = 2010;
    hipDriverGetVersion(&version);
    version /= 10;
    return version / 10 + version % 10;
}

int HipModule::getComputeCapability() {
    staticInit();
    if (!available) return 10;
    int major, minor;
    checkError("hipDeviceComputeCapability", hipDeviceComputeCapability(&major, &minor, device));
    return major * 10 + minor;
}

hipEvent_t HipModule::getStartEvent() {
    staticInit();
    return startEvent;
}

hipEvent_t HipModule::getEndEvent() {
    staticInit();
    return endEvent;
}

HipModule::HipModule(const void * cubin) {
    staticInit();
    checkError("hipModuleLoadData", hipModuleLoadData(&module, cubin));
}

HipModule::HipModule(const QString & cubinFile) {
    staticInit();
    checkError("hipModuleLoad", hipModuleLoad(&module, cubinFile.toUtf8().constData()));
}

HipModule::~HipModule() {
    for (int i = 0; i < globals.size(); i++)
        delete globals[i];
    checkError("hipModuleUnload", hipModuleUnload(module));
}

hipModule_t HipModule::getHandle() {
    return module;
}

bool HipModule::hasKernel(const QString & name) {
    return (findKernel(name) != nullptr);
}

HipKernel HipModule::getKernel(const QString & name) {
    hipFunction_t kernel = findKernel(name);
    if (!kernel) {
        logger(LOG_ERROR) << "ERROR <HipModule> Kernel '" << name << "' not found.\n";
        exit(EXIT_FAILURE);
    }
    return HipKernel(this, kernel);
}

Buffer & HipModule::getGlobal(const QString & name) {
    QHash<QString, int>::iterator found = globalHash.find(name);
    if (found != globalHash.end())
        return *globals[found.value()];
    hipDeviceptr_t ptr;
    size_t size;
    checkError("hipModuleGetGlobal", hipModuleGetGlobal(&ptr, &size, module, name.toUtf8().constData()));
    Buffer* buffer = new Buffer();
    buffer->wrapHip(ptr, size);
    globalHash.insert(name, globals.size());
    globals.push_back(buffer);
    return *buffer;
}

void HipModule::updateGlobals(bool async) {
    for (int i = 0; i < globals.size(); i++)
        globals[i]->setOwner(Buffer::HIP, true);
}
