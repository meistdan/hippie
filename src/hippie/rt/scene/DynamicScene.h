/**
 * \file	DynamicScene.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	DynamicScene class header file.
 */

#ifndef _DYNAMIC_SCENE_H_
#define _DYNAMIC_SCENE_H_

#include <QVector>
#include "Scene.h"

#define DELTA_FRAME_RATE 5.0f
#define MAX_FRAME_RATE 500.0f

struct Frame {
    AABB sceneBox;
    AABB centroidBox;
    Buffer vertices;
    Buffer normals;
};

class DynamicScene : public Scene {

protected:

    AABB staticSceneBox;
    AABB staticCentroidBox;
    int numberOfStaticVertices;

    bool loop;
    float time;
    float length;
    float frameRate;
    int frameIndex;
    QVector<Frame> frames;

    DynamicScene(void);

public:

    virtual ~DynamicScene(void);

    float getFrameRate(void);
    void setFrameRate(float frameRate);
    bool isLooped(void);
    void setLoop(bool loop);
    float getTime(void);
    void setTime(float time);
    void resetTime(void);
    int getNumberOfFrames(void);

    int getNumberOfStaticVertices(void);
    int getNumberOfDynamicVertices(void);

    void increaseFrameRate(void);
    void decreaseFrameRate(void);

    int getFrameIndex(void);
    void setFrameIndex(int index);
    void setNextFrame(void);
    void setPreviousFrame(void);

    virtual bool isDynamic(void) const;

    friend class Interpolator;
    friend class SceneLoader;

};

#endif /* _DYNAMIC_SCENE_H_ */
