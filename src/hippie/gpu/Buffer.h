/**
  * \file	Buffer.h
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	Buffer class header file.
  */

#ifndef _BUFFER_H_
#define _BUFFER_H_

#include <QVector>
#include "Globals.h"

class Buffer {

public:

    enum Module {
        CPU = 1 << 0,
        GL = 1 << 1,
        HIP = 1 << 2,
        Module_None = 0,
        Module_All = (1 << 3) - 1
    };

private:

    long long size;
    Module original;
    Module owner;
    unsigned int exists;
    unsigned int dirty;

    unsigned char * cpuPtr;
    hipDeviceptr_t hipPtr;
    GLuint glBuffer;

    static void cpuAlloc(unsigned char *& cpuPtr, long long size);
    static void cpuFree(unsigned char *& cpuPtr);
    static void hipAlloc(hipDeviceptr_t & hipPtr, long long size, GLuint glBuffer);
    static void hipFree(hipDeviceptr_t& hipPtr, GLuint glBuffer);
    static void glAlloc(GLuint& glBuffer, long long size, const void* data);
    static void glFree(GLuint& glBuffer);
    static void checkSize(long long size, int bits, const QString & funcName);
    static void memcpyXtoX(void * dstHost, hipDeviceptr_t dstDevice, const void * srcHost, hipDeviceptr_t srcDevice, long long size);

    void init(long long size);
    void deinit(void);
    void wrap(Module module, long long size);
    void realloc(long long size);
    void validateCPU(void);

public:

    static void memcpyHtoD(hipDeviceptr_t dst, const void * src, long long size);
    static void memcpyDtoH(void * dst, hipDeviceptr_t src, long long size);
    static void memcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, long long size);

    Buffer(void);
    Buffer(Buffer & other);
    ~Buffer(void);

    void wrapCPU(void * cpuPtr, long long size);
    void wrapHip(hipDeviceptr_t hipPtr, long long size);
    void wrapGL(GLuint glBuffer);

    long long getSize(void) const;

    void reset(void);
    void reset(const void * ptr, long long size);
    void resize(long long size);
    void resizeDiscard(long long size);
    void free(Module module);
    void free(void);

    void getRange(void * dst, long long srcOfs, long long size) const;
    void get(void * ptr);
    template <class T> void get(QVector<T> & data) { Q_ASSERT(data.size() * sizeof(T) == getSize()); get(data.data()); }

    void setRange(long long dstOfs, const void * src, long long size);
    void setRange(long long dstOfs, Buffer& src, long long srcOfs, long long size);
    void set(const void * ptr);
    void set(const void * ptr, long long size);
    void set(Buffer & other);
    template <class T> void set(const QVector<T> & data) { set(data.data(), sizeof(T) * data.size()); }
    template <class T> void set(const QVector<T> & data, int start, int end) { set(data.data() + start, (end - start) * sizeof(T)); }

    void clearRange(long long dstOfs, int value, long long size);
    void clear(int value = 0);

    void setOwner(Module module, bool modify);
    Module getOwner(void) const;
    void discard(void);

    const unsigned char * getPtr(long long ofs = 0);
    hipDeviceptr_t getHipPtr(long long ofs = 0);
    GLuint getGLBuffer(void);

    unsigned char * getMutablePtr(long long ofs = 0);
    hipDeviceptr_t getMutableHipPtr(long long ofs = 0);
    GLuint getMutableGLBuffer(void);

    unsigned char * getMutablePtrDiscard(long long ofs = 0);
    hipDeviceptr_t getMutableHipPtrDiscard(long long ofs = 0);
    GLuint getMutableGLBufferDiscard(void);

    Buffer & operator=(Buffer & other);

};

#endif /* _BUFFER_H_ */