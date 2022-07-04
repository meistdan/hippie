/**
 * \file	TaskQueue.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	TaskQueue class header file.
 */

#ifndef _TASK_QUEUE_H_
#define _TASK_QUEUE_H_

#include "gpu/Buffer.h"

template <typename Task>
class TaskQueue {

protected:

    Buffer queue[2];
    Buffer size[2];
    bool swapBuffers;

public:

    TaskQueue(void) : swapBuffers(false) {
        size[0].resizeDiscard(sizeof(int));
        size[1].resizeDiscard(sizeof(int));
    }

    virtual ~TaskQueue(void) {
    }

    virtual void init(void) {
        swapBuffers = false;
        *(int*)size[0].getMutablePtr() = 0;
        *(int*)size[1].getMutablePtr() = 0;
    }

    virtual void init(int _size) {
        swapBuffers = false;
        *(int*)size[1].getMutablePtr() = 0;
        *(int*)size[0].getMutablePtr() = 1;
        queue[0].resizeDiscard(_size * sizeof(Task));
        queue[1].resizeDiscard(_size * sizeof(Task));
    }

    void clear(void) {
        swapBuffers = false;
        queue[0].free();
        queue[1].free();
        size[0].free();
        size[1].free();
    }

    void swap(void) {
        swapBuffers = !swapBuffers;
    }

    void resetOutSize(void) {
        if (swapBuffers) *(int*)size[0].getMutablePtr() = 0;
        else *(int*)size[1].getMutablePtr() = 0;
    }

    int getInSize(void) {
        return swapBuffers ? *(int*)size[1].getPtr() : *(int*)size[0].getPtr();
    }

    int getOutSize(void) {
        return swapBuffers ? *(int*)size[0].getPtr() : *(int*)size[1].getPtr();
    }

    Buffer & getInBuffer(void) {
        return swapBuffers ? queue[1] : queue[0];
    }

    Buffer & getOutBuffer(void) {
        return swapBuffers ? queue[0] : queue[1];
    }

    Buffer & getInSizeBuffer(void) {
        return swapBuffers ? size[1] : size[0];
    }

    Buffer & getOutSizeBuffer(void) {
        return swapBuffers ? size[0] : size[1];
    }

};

#endif /* _TASK_QUEUE_H_ */
