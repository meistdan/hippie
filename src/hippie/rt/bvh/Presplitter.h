/**
* \file	    Presplitter.h
* \author	Daniel Meister
* \date	    2019/05/02
* \brief	Presplitter class header file.
*/

#ifndef _PRESPLITTER_H_
#define _PRESPLITTER_H_

#include "gpu/HipCompiler.h"
#include "rt/TaskQueue.h"
#include "rt/scene/Scene.h"
#include "PresplitterKernels.h"

class SplitQueue : public TaskQueue<SplitTask> {

protected:

    Buffer boxes[4];

public:

    SplitQueue(void);
    virtual ~SplitQueue(void);

    void init(int maxSize, int size);
    virtual void clear(void);

    Buffer & getInBoxBuffer(int i);
    Buffer & getOutBoxBuffer(int i);

};

class Presplitter {

private:

    HipCompiler compiler;

    Buffer priorities;

    SplitQueue queue;

    float beta;

    float computePriorities(Scene * scene);
    float sumPriorities(int numberOfTriangles, int & S, float D);
    float sumPrioritiesRound(int numberOfTriangles, int & S, float D);
    float computeD(int numberOfTriangles, int Smax, int & S, float & D);
    float initSplitTasks(Scene * scene, int numberOfReferences, float D);

    float split(Scene * scene, int numberOfReferences, Buffer & referenceBoxesMin, Buffer & referenceBoxesMax, Buffer & triangleIndices);

public:

    Presplitter(void);
    ~Presplitter(void);

    float presplit(Scene * scene, Buffer & referenceBoxesMin, Buffer & referenceBoxesMax, Buffer & triangleIndices, int & numberOfReferences);

    float getBeta(void);
    void setBeta(float beta);

    void clear(void);

};

#endif /* _PRESPLITTER_H_ */
