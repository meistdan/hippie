/**
  * \file	HipKernel.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	HipKernel class header file.
  */

#ifndef _HIP_KERNEL_H_
#define _HIP_KERNEL_H_

#include "gpu/Buffer.h"
#include "util/Math.h"

class HipModule;

class HipKernel {

public:

    struct Param { // Wrapper for converting kernel parameters to HIP-compatible types.
        int size;
        int align;
        const void * value;
        hipDeviceptr_t hipPtr;
        Buffer buffer;

        template <class T> Param(const T & v) { size = sizeof(T); align = __alignof(T); value = &v; }
        template <class T> Param(const T * ptr, int num = 1) { buffer.wrapCPU(ptr, num * sizeof(T)); setHipPtr(buffer.getHipPtr()); }
        template <class T> Param(const QVector<T> & v) { buffer.wrapCPU(v.getPtr(), v.getNumBytes()); setHipPtr(buffer.getHipPtr()); }
        template <class T> Param(QVector<T>& v) { buffer.wrapCPU(v.getPtr(), v.getNumBytes()); setHipPtr(buffer.getMutableHipPtr()); }
        Param(Buffer & v);
        void setHipPtr(hipDeviceptr_t ptr);
    };

    typedef const Param & P; // To reduce the amount of code in setParams() overloads.

private:

    HipModule * module;
    hipFunction_t function;
    QVector<unsigned char> params;
    QVector<void*> paramPtrs;
    int sharedMemorySize;
    Vec2i gridSize;
    Vec2i blockSize;

    bool prepareLaunch(void);
    void performLaunch(void);

public:

    HipKernel(HipModule * module = nullptr, hipFunction_t function = nullptr);
    HipKernel(const HipKernel & other);
    ~HipKernel(void);

    HipModule * getModule(void) const;
    hipFunction_t getHandle(void) const;

    HipKernel & setParams(const Param * const * params, int numParams);

    HipKernel & setParams(void);
    HipKernel & setParams(P p0);
    HipKernel & setParams(P p0, P p1);
    HipKernel & setParams(P p0, P p1, P p2);
    HipKernel & setParams(P p0, P p1, P p2, P p3);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21, P p22);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21, P p22, P p23);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21, P p22, P p23, P p24);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21, P p22, P p23, P p24, P p25);
    HipKernel & setParams(P p0, P p1, P p2, P p3, P p4, P p5, P p6, P p7, P p8, P p9, P p10, P p11, P p12, P p13, P p14, P p15, P p16, P p17, P p18, P p19, P p20, P p21, P p22, P p23, P p24, P p25, P p26);

    HipKernel & setSharedMemorySize(int bytes);

    int getNumSmem(void) const;
    int getNumRegs(void) const;
    int getSharedMemorySize(void) const;
    Vec2i getDefaultBlockSize(void) const; // Smallest block that reaches maximal occupancy.

    HipKernel & setGrid(int numThreads, const Vec2i & blockSize = Vec2i()); // Generates at least numThreads.
    HipKernel & setGrid(const Vec2i & sizeThreads, const Vec2i & blockSize = Vec2i()); // Generates at least sizeThreads in both X and Y.

    HipKernel & launch(void);
    HipKernel & launch(int numThreads, const Vec2i & blockSize = Vec2i());

    float launchTimed(bool yield = true); // Returns GPU time in seconds.
    float launchTimed(int numThreads, const Vec2i & blockSize = Vec2i(), bool yield = true);

    HipKernel & sync(bool yield = true); // False = low latency but keeps the CPU busy. True = long latency but relieves the CPU.
    HipKernel & operator=(const HipKernel & other);

};

#endif /* _HIP_KERNEL_H_ */
