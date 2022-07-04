/**
  * \file	Buffer.cpp
  * \author	Daniel Meister
  * \date	2014/05/10
  * \brief	Buffer class source file.
  */

#include "util/Logger.h"
#include "Buffer.h"
#include "HipModule.h"

void Buffer::cpuAlloc(unsigned char *& cpuPtr, long long size) {
    checkSize(size, sizeof(unsigned char*) * 8 - 1, "malloc");
    cpuPtr = new unsigned char[(size_t)(size)];
}

void Buffer::cpuFree(unsigned char *& cpuPtr) {
    if (cpuPtr) {
        delete[] cpuPtr;
        cpuPtr = nullptr;
    }
}

void Buffer::hipAlloc(hipDeviceptr_t & hipPtr, long long size, GLuint glBuffer) {
    HipModule::staticInit();
    checkSize(size, 32, "hipMalloc");
    HipModule::checkError("hipMalloc", hipMalloc((void**)&hipPtr, qMax(1U, (unsigned int)(size))));
}

void Buffer::hipFree(hipDeviceptr_t & hipPtr, GLuint glBuffer) {
    if (hipPtr) {
        HipModule::checkError("hipFree", ::hipFree((void*)hipPtr));
        hipPtr = 0;
    }
}

void Buffer::checkSize(long long size, int bits, const QString & funcName) {
    Q_ASSERT(size >= 0);
    if ((unsigned long long)size > (((unsigned long long)1 << bits) - 1)) {
        logger(LOG_ERROR) << "ERROR <Buffer> Buffer too large for '" << funcName << "()'!\n";
        exit(EXIT_FAILURE);
    }
}

void Buffer::glAlloc(GLuint& glBuffer, long long size, const void* data) {
    Q_ASSERT(size >= 0);
    GLint oldBuffer;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldBuffer);
    glGenBuffers(1, &glBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, glBuffer);
    checkSize(size, sizeof(GLsizeiptr) * 8 - 1, "glBufferData");
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)size, data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, oldBuffer);
    checkGLErrors();
}

void Buffer::glFree(GLuint& glBuffer) {
    if (glBuffer) {
        glDeleteBuffers(1, &glBuffer);
        checkGLErrors();
        glBuffer = 0;
    }
}

void Buffer::memcpyXtoX(void * dstHost, hipDeviceptr_t dstDevice, const void * srcHost, hipDeviceptr_t srcDevice, long long size) {

    hipError_t res;
    if (size <= 0) return;

    // Try to copy.
    if (dstHost && srcHost) {
        memcpy(dstHost, srcHost, (size_t)size);
        res = hipSuccess;
    }
    else if (srcHost) {
        res = hipMemcpyHtoD(dstDevice, (void*)srcHost, (unsigned int)size);
    }
    else if (dstHost) {
        res = hipMemcpyDtoH(dstHost, srcDevice, (unsigned int)size);
    }
    else {
        res = hipMemcpyDtoD(dstDevice, srcDevice, (unsigned int)size);
    }

    // Success => done.
    if (res == hipSuccess) return;

    // Single byte => fail.
    if (size == 1) HipModule::checkError("hipMemcpyXtoX", res);

    // Otherwise => subdivide.
    // Driver does not allow memcpy() to cross allocation boundaries.
    long long mid = size >> 1;
    memcpyXtoX(dstHost, dstDevice, srcHost, srcDevice, mid);

    memcpyXtoX(
        (dstHost) ? (unsigned char*)dstHost + mid : nullptr,
        (dstHost) ? 0 : (hipDeviceptr_t)((unsigned char*)dstDevice + mid),
        (srcHost) ? (const unsigned char*)srcHost + mid : nullptr,
        (srcHost) ? 0 : (hipDeviceptr_t)((unsigned char*)srcDevice + mid),
        size - mid);
}

void Buffer::init(long long _size) {
    Q_ASSERT(_size >= 0);
    size = _size;
    original = Module_None;
    owner = Module_None;
    exists = Module_None;
    dirty = Module_None;
    cpuPtr = nullptr;
    glBuffer = 0;
    hipPtr = 0;
}

void Buffer::deinit() {
    // Wrapped buffer => ensure that the original is up-to-date.
    if (original != Module_None) setOwner(original, false);
    // Free buffers.
    if (original != HIP) hipFree(hipPtr, glBuffer);
    if (original != CPU) cpuFree(cpuPtr);
    if (original != GL)  glFree(glBuffer);
}

void Buffer::wrap(Module module, long long _size) {
    Q_ASSERT(_size >= 0);
    Q_ASSERT(exists == Module_None);
    size = _size;
    original = module;
    owner = module;
    exists = module;
}

void Buffer::realloc(long long _size) {

    Q_ASSERT(_size >= 0);

    // No change => done.
    if (size == _size)
        return;

    // Wrapped buffer => free others.
    if (original) {
        for (int i = 1; i < (int)Module_All; i <<= 1) 
            free((Module)i);
        return;
    }

    // No need to retain old data => reset.
    if (!_size || !size || exists == Module_None) {
        reset(nullptr, _size);
        return;
    }

    // HIP buffer => device-to-device copy.
    if (owner == HIP) {
        hipDeviceptr_t _hipPtr;
        hipAlloc(_hipPtr, _size, 0);
        memcpyXtoX(nullptr, _hipPtr, nullptr, getHipPtr(), qMin(_size, size));
        reset(nullptr, _size);
        exists = HIP;
        hipPtr = _hipPtr;
        return;
    }

    // Host-to-host copy.
    unsigned char * _cpuPtr;
    cpuAlloc(_cpuPtr, _size);
    memcpy(_cpuPtr, getPtr(), (size_t)qMin(_size, size));
    reset(nullptr, _size);
    exists = CPU;
    cpuPtr = _cpuPtr;
}

void Buffer::validateCPU(void) {

    // Already valid => done.
    if ((exists & CPU) != 0 && (dirty & CPU) == 0) return;
    dirty &= ~CPU;

    // Find source for the data.
    Module source = Module_None;
    for (int i = 1; i < (int)Module_All; i <<= 1) {
        if (i != CPU && (exists & i) != 0 && (dirty & i) == 0) {
            source = (Module)i;
            break;
        }
    }

    // No source => done.
    if (source == Module_None) return;

    // No buffer => allocate one.
    if ((exists & CPU) == 0) {
        cpuAlloc(cpuPtr, size);
        exists |= CPU;
    }

    // No valid data => no need to copy.
    if (!size) return;

    // Copy data from the source.
    if (source == GL) {
        GLint oldBuffer;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, glBuffer);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)size, cpuPtr);
        glBindBuffer(GL_ARRAY_BUFFER, oldBuffer);
        checkGLErrors();
    }
    else {
        Q_ASSERT(source == HIP);
        memcpyDtoH(cpuPtr, hipPtr, (unsigned int)size);
    }
}

void Buffer::memcpyHtoD(hipDeviceptr_t dst, const void * src, long long size) {
    memcpyXtoX(nullptr, dst, src, 0, size);
}

void Buffer::memcpyDtoH(void * dst, hipDeviceptr_t src, long long size) {
    memcpyXtoX(dst, 0, nullptr, src, size);
}

void Buffer::memcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, long long size) {
    memcpyXtoX(nullptr, dst, nullptr, src, size);
}

Buffer::Buffer(void) {
    init(0);
}

Buffer::Buffer(Buffer & other) {
    init(other.getSize()); 
    setRange(0, other, 0, other.getSize());
}

Buffer::~Buffer() {
    deinit();
}

void Buffer::wrapCPU(void * _cpuPtr, long long _size) {
    Q_ASSERT(_cpuPtr || !_size);
    Q_ASSERT(_size >= 0);
    cpuPtr = (unsigned char*)_cpuPtr;
    wrap(CPU, _size);
}

void Buffer::wrapGL(GLuint _glBuffer) {
    Q_ASSERT(_glBuffer != 0);
    GLint _size;
    {
        GLint oldBuffer;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, _glBuffer);
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &_size);
        glBindBuffer(GL_ARRAY_BUFFER, oldBuffer);
        checkGLErrors();
    }
    glBuffer = _glBuffer;
    wrap(GL, _size);
}

void Buffer::wrapHip(hipDeviceptr_t _hipPtr, long long _size) {
    Q_ASSERT(_hipPtr || !_size);
    hipPtr = _hipPtr;
    wrap(HIP, _size);
}

long long Buffer::getSize() const {
    return size;
}

void Buffer::reset(void) {
    deinit();
    init(0);
}

void Buffer::reset(const void * ptr, long long _size) {
    deinit();
    init(_size);
    if (ptr) setRange(0, ptr, _size);
}

void Buffer::resize(long long _size) {
    realloc(_size);
}

void Buffer::resizeDiscard(long long _size) {
    if (size != _size) reset(nullptr, _size);
}

void Buffer::free(Module module) {

    if ((exists & module) == 0 || exists == (unsigned int)module || original == module)
        return;

    setOwner(module, false);

    if (owner == module)
        for (int i = 1; i < (int)Module_All; i <<= 1)
            if (module != i && (exists & i) != 0 && (dirty & i) == 0) {
                setOwner((Module)i, false);
                break;
            }

    if (owner == module)
        for (int i = 1; i < (int)Module_All; i <<= 1)
            if (module != i && (exists & i) != 0) {
                setOwner((Module)i, false);
                break;
            }

    switch (module) {
    case CPU:   cpuFree(cpuPtr); break;
    case HIP:   hipFree(hipPtr, glBuffer); break;
    case GL:    glFree(glBuffer); break;
    }
    exists &= ~module;
}

void Buffer::free() {
    deinit();
    init(0);
}

void Buffer::getRange(void * dst, long long srcOfs, long long _size) const {

    Q_ASSERT(dst || !_size);
    Q_ASSERT(srcOfs >= 0 && srcOfs <= size - _size);
    Q_ASSERT(_size >= 0);

    if (!_size) return;

    switch (owner) {
    case GL:
    {
        GLint oldBuffer;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, glBuffer);
        glGetBufferSubData(GL_ARRAY_BUFFER, (GLintptr)srcOfs, (GLsizeiptr)_size, dst);
        glBindBuffer(GL_ARRAY_BUFFER, oldBuffer);
        checkGLErrors();
    }
    break;

    case HIP:
        memcpyDtoH(dst, (hipDeviceptr_t)((unsigned char*)hipPtr + (unsigned int)srcOfs), (unsigned int)_size);
        break;

    default:
        if ((exists & CPU) != 0)
            memcpy(dst, cpuPtr + srcOfs, (size_t)_size);
        break;
    }
}

void Buffer::get(void * ptr) {
    getRange(ptr, 0, getSize());
}

void Buffer::setRange(long long dstOfs, const void * src, long long _size) {

    Q_ASSERT(dstOfs >= 0 && dstOfs <= size - _size);
    Q_ASSERT(src || !_size);
    Q_ASSERT(_size >= 0);

    if (!_size)
        return;

    switch (owner)
    {
    case GL:
    {
        GLint oldBuffer;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, getMutableGLBuffer());
        glBufferSubData(GL_ARRAY_BUFFER, (GLintptr)dstOfs, (GLsizeiptr)_size, src);
        glBindBuffer(GL_ARRAY_BUFFER, oldBuffer);
        checkGLErrors();
    }
    break;

    case HIP:
        memcpyHtoD(getMutableHipPtr(dstOfs), src, (unsigned int)_size);
        break;

    default:
        memcpy(getMutablePtr(dstOfs), src, (size_t)_size);
        break;
    }
}

void Buffer::setRange(long long dstOfs, Buffer& src, long long srcOfs, long long _size) {

    Q_ASSERT(_size >= 0);
    Q_ASSERT(dstOfs >= 0 && dstOfs <= size - _size);
    Q_ASSERT(srcOfs >= 0 && srcOfs <= src.size - _size);

    if (!_size) return;

    if ((src.exists & HIP) != 0 && (src.dirty & HIP) == 0 && (owner == HIP || owner == Module_None))
        memcpyDtoD(getMutableHipPtr(dstOfs), src.getHipPtr(srcOfs), (unsigned int)_size);
    else if ((src.exists & CPU) != 0 && (src.dirty & CPU) == 0)
        setRange(dstOfs, src.getPtr(srcOfs), _size);
    else
        src.getRange(getMutablePtr(dstOfs), srcOfs, _size);
}

void Buffer::set(const void * ptr) {
    setRange(0, ptr, getSize());
}

void Buffer::set(const void * ptr, long long _size) {
    resizeDiscard(_size); setRange(0, ptr, _size);
}

void Buffer::set(Buffer & other) {
    if (&other != this) {
        resizeDiscard(other.getSize());
        setRange(0, other, 0, other.getSize());
    }
}

void Buffer::clearRange(long long dstOfs, int value, long long _size) {
    Q_ASSERT(_size >= 0);
    Q_ASSERT(dstOfs >= 0 && dstOfs <= size - _size);
    if (!_size) return;
    if (owner == HIP) HipModule::checkError("hipMemsetD8", hipMemsetD8(getMutableHipPtr(dstOfs), (unsigned char)value, (unsigned int)size));
    else memset(getMutablePtr(dstOfs), value, (size_t)_size);
}

void Buffer::clear(int value) {
    clearRange(0, value, size);
}

void Buffer::setOwner(Module module, bool modify) {

    Q_ASSERT((module & ~Module_All) == 0);
    Q_ASSERT((module & (module - 1)) == 0);

    // Same owner => done.
    if (owner == module) {
        if (modify) dirty = Module_All - module;
        return;
    }

    // Validate CPU.
    if (module == CPU) {
        if ((exists & CPU) == 0) {
            cpuAlloc(cpuPtr, size);
            exists |= CPU;
            dirty |= CPU;
        }
        validateCPU();
    }

    // Validate GL.
    if (module == GL && (exists & GL) == 0) {
        validateCPU();
        glAlloc(glBuffer, size, cpuPtr);
        exists |= GL;
        dirty &= ~GL;
    }
    else if (module == GL && (dirty & GL) != 0) {
        validateCPU();
        Q_ASSERT((exists & CPU) != 0);
        if (size) {
            GLint oldBuffer;
            glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, glBuffer);
            glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)size, cpuPtr);
            glBindBuffer(GL_ARRAY_BUFFER, oldBuffer);
            checkGLErrors();
        }
        dirty &= ~GL;
    }

    // Validate HIP.
    if (module == HIP) {
        if ((exists & HIP) == 0) {
            hipAlloc(hipPtr, size, glBuffer);
            exists |= HIP;
            dirty |= HIP;
        }

        if ((dirty & HIP) != 0) {
            validateCPU();
            if ((exists & CPU) != 0 && size)
                memcpyHtoD(hipPtr, cpuPtr, (unsigned int)size);
            dirty &= ~HIP;
        }
    }

    // Set the new owner.
    owner = module;
    if (modify) dirty = Module_All - module;
}

Buffer::Module Buffer::getOwner() const {
    return owner;
}

void Buffer::discard() {
    dirty = 0;
}

const unsigned char * Buffer::getPtr(long long ofs) {
    Q_ASSERT(ofs >= 0 && ofs <= size);
    setOwner(CPU, false);
    return cpuPtr + ofs;
}

hipDeviceptr_t Buffer::getHipPtr(long long ofs) {
    Q_ASSERT(ofs >= 0 && ofs <= size);
    setOwner(HIP, false);
    return (hipDeviceptr_t)((unsigned char*)hipPtr + (unsigned int)ofs);
}

GLuint Buffer::getGLBuffer() {
    setOwner(GL, false);
    return glBuffer;
}

unsigned char * Buffer::getMutablePtr(long long ofs) {
    Q_ASSERT(ofs >= 0 && ofs <= size);
    setOwner(CPU, true);
    return cpuPtr + ofs;
}

hipDeviceptr_t Buffer::getMutableHipPtr(long long ofs) {
    Q_ASSERT(ofs >= 0 && ofs <= size);
    setOwner(HIP, true);
    return (hipDeviceptr_t)((unsigned char*)hipPtr + (unsigned int)ofs);
}

GLuint Buffer::getMutableGLBuffer() {
    setOwner(GL, true);
    return glBuffer;
}

unsigned char * Buffer::getMutablePtrDiscard(long long ofs) {
    discard();
    return getMutablePtr(ofs);
}

hipDeviceptr_t Buffer::getMutableHipPtrDiscard(long long ofs) {
    discard();
    return getMutableHipPtr(ofs);
}

GLuint Buffer::getMutableGLBufferDiscard() {
    discard();
    return getMutableGLBuffer();
}

Buffer & Buffer::operator=(Buffer & other) {
    set(other);
    return *this;
}
