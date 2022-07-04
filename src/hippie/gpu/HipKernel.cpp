/**
  * \file	HipKernel.cpp
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	HipKernel class source file.
  */

#include <QTime>
#include "HipKernel.h"
#include "HipModule.h"
#include "Globals.h"

#include "util/Logger.h"

HipKernel::Param::Param(Buffer & v) {
    setHipPtr((hipDeviceptr_t)v.getMutableHipPtr());
}

void HipKernel::Param::setHipPtr(hipDeviceptr_t ptr) {
    size = sizeof(hipDeviceptr_t);
    align = __alignof(hipDeviceptr_t);
    value = &hipPtr;
    hipPtr = ptr;
}

bool HipKernel::prepareLaunch() {

    // Nothing to do => skip.
    if (!module || !function || qMin(gridSize.x, gridSize.y) == 0) return false;

    // Update globals.
    module->updateGlobals();

    return true;
}

void HipKernel::performLaunch() {
    HipModule::checkError("hipModuleLaunchKernel", hipModuleLaunchKernel(function, gridSize.x, 
        gridSize.y, 1, blockSize.x, blockSize.y, 1, sharedMemorySize, nullptr, paramPtrs.data(), nullptr));
}

HipKernel::HipKernel(HipModule * _module, hipFunction_t _function) :
    module(_module),
    function(_function),
    sharedMemorySize(0),
    gridSize(1, 1),
    blockSize(1, 1)
{
}

HipKernel::HipKernel(const HipKernel & other) {
    operator=(other);
}

HipKernel::~HipKernel() {
}

HipModule * HipKernel::getModule() const {
    return module;
}

hipFunction_t HipKernel::getHandle() const {
    return function;
}

HipKernel & HipKernel::setParams(const Param * const * _params, int numParams) {

    Q_ASSERT(numParams == 0 || _params);
    Q_ASSERT(numParams >= 0);

    int size = 0;
    for (int i = 0; i < numParams; i++) {
        size = (size + _params[i]->align - 1) & -_params[i]->align;
        size += _params[i]->size;
    }

    params.clear();
    params.resize(size);
    paramPtrs.clear();
    paramPtrs.resize(size);

    int ofs = 0;
    for (int i = 0; i < numParams; i++) {
        ofs = (ofs + _params[i]->align - 1) & -_params[i]->align;
        memcpy(params.data() + ofs, _params[i]->value, _params[i]->size);
        paramPtrs[i] = params.data() + ofs;
        ofs += _params[i]->size;
    }

    return *this;
}

HipKernel & HipKernel::setParams() {
    return setParams((const Param* const*)nullptr, 0);
}

HipKernel & HipKernel::setParams(P p0) {
    const Param* p[] = { &p0 };
    return setParams(p, 1);
}

HipKernel & HipKernel::setParams(P p0, P p1) {
    const Param* p[] = { &p0, &p1 };
    return setParams(p, 2);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2) {
    const Param* p[] = { &p0, &p1, &p2 };
    return setParams(p, 3);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3) {
    const Param* p[] = { &p0, &p1, &p2, &p3 };
    return setParams(p, 4);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4 };
    return setParams(p, 5);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5 };
    return setParams(p, 6);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6 };
    return setParams(p, 7);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7 };
    return setParams(p, 8);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8 };
    return setParams(p, 9);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9 };
    return setParams(p, 10);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10 };
    return setParams(p, 11);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11 };
    return setParams(p, 12);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12 };
    return setParams(p, 13);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13 };
    return setParams(p, 14);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14 };
    return setParams(p, 15);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15 };
    return setParams(p, 16);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16 };
    return setParams(p, 17);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17 };
    return setParams(p, 18);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18 };
    return setParams(p, 19);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19 };
    return setParams(p, 20);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20 };
    return setParams(p, 21);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21 };
    return setParams(p, 22);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21, P p22) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &p22 };
    return setParams(p, 23);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21, P p22, P p23) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &p22, &p23 };
    return setParams(p, 24);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21, P p22, P p23, P p24) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &p22, &p23, &p24 };
    return setParams(p, 25);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21, P p22, P p23, P p24, P p25) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &p22, &p23, &p24, &p25 };
    return setParams(p, 26);
}

HipKernel & HipKernel::setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21, P p22, P p23, P p24, P p25, P p26) {
    const Param* p[] = { &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &p22, &p23, &p24, &p25, &p26 };
    return setParams(p, 27);
}

HipKernel & HipKernel::setSharedMemorySize(int bytes) {
    Q_ASSERT(bytes >= 0);
    sharedMemorySize = bytes;
    return *this;
}

int HipKernel::getNumSmem() const {
    int numSmem;
    HipModule::checkError("hipFuncGetAttribute", hipFuncGetAttribute(&numSmem, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
    return numSmem;
}

int HipKernel::getNumRegs() const {
    int numRegs;
    HipModule::checkError("hipFuncGetAttribute", hipFuncGetAttribute(&numRegs, HIP_FUNC_ATTRIBUTE_NUM_REGS, function));
    return numRegs;
}

int HipKernel::getSharedMemorySize() const {
    return sharedMemorySize;
}

Vec2i HipKernel::getDefaultBlockSize() const {
    int minGridSize, blockSize;
    HipModule::checkError("hipModuleOccupancyMaxPotentialBlockSize",
        hipModuleOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, 0, 0));
    return Vec2i(blockSize, 1);
}

HipKernel & HipKernel::setGrid(int numThreads, const Vec2i & _blockSize) {

    Q_ASSERT(numThreads >= 0);
    blockSize = (qMin(_blockSize.x, _blockSize.y) > 0) ? _blockSize : getDefaultBlockSize();

    int maxGridWidth = 2147483647;

    int threadsPerBlock = blockSize.x * blockSize.y;
    gridSize = Vec2i((numThreads + threadsPerBlock - 1) / threadsPerBlock, 1);

    while (gridSize.x > maxGridWidth) {
        gridSize.x = (gridSize.x + 1) >> 1;
        gridSize.y <<= 1;
    }

    return *this;
}

HipKernel & HipKernel::setGrid(const Vec2i & sizeThreads, const Vec2i & _blockSize) {
    Q_ASSERT(qMin(sizeThreads.x, sizeThreads.y) >= 0);
    blockSize = (qMin(_blockSize.x, _blockSize.y) > 0) ? _blockSize : getDefaultBlockSize();
    gridSize = (sizeThreads + blockSize - 1) / blockSize;
    return *this;
}

HipKernel & HipKernel::launch() {
    if (prepareLaunch()) performLaunch();
    return *this;
}

HipKernel & HipKernel::launch(int numThreads, const Vec2i & _blockSize) {
    setGrid(numThreads, _blockSize);
    return launch();
}

float HipKernel::launchTimed(bool yield) {

    // Prepare and sync before timing.
    if (!prepareLaunch()) return 0.0f;
    sync(false); // wait is short => spin

    // Events not supported => use CPU-based timer.
    hipEvent_t startEvent = HipModule::getStartEvent();
    hipEvent_t endEvent = HipModule::getEndEvent();

    if (!startEvent) {
        QTime timer;
        timer.start();
        performLaunch();
        sync(false); // need accurate timing => spin
        return timer.elapsed() * 1.0e-3f;
    }

    // Launch and record events.
    HipModule::checkError("hipEventRecord", hipEventRecord(startEvent, nullptr));
    performLaunch();
    HipModule::checkError("hipEventRecord", hipEventRecord(endEvent, nullptr));
    sync(yield);

    // Query GPU time between the events.
    float time = 0.0f;
    HipModule::checkError("hipEventElapsedTime", hipEventElapsedTime(&time, startEvent, endEvent));

    return time * 1.0e-3f;
}

float HipKernel::launchTimed(int numThreads, const Vec2i & _blockSize, bool yield) {
#if 1
    setGrid(numThreads, _blockSize);
    return launchTimed(yield);
#else
    launch(numThreads, _blockSize);
    return 0.0f;
#endif
}

HipKernel & HipKernel::sync(bool yield) {
    HipModule::sync(yield);
    return *this;
}

HipKernel & HipKernel::operator=(const HipKernel & other) {
    module = other.module;
    function = other.function;
    params = other.params;
    gridSize = other.gridSize;
    blockSize = other.blockSize;
    return *this;
}
