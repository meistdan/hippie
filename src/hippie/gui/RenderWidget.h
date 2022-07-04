/**
 * \file	RenderWidget.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	RenderWidget class header file.
 */

#ifndef _RENDER_WIDGET_H_
#define _RENDER_WIDGET_H_

#include "rt/bvh/PLOCBuilder.h"
#include "rt/bvh/SBVHBuilder.h"
#include "rt/bvh/LBVHBuilder.h"
#include "rt/bvh/ATRBuilder.h"
#include "rt/bvh/TRBuilder.h"
#include "rt/bvh/InsertionBuilder.h"
#include "rt/renderer/Renderer.h"
#include "rt/scene/Camera.h"
#include "rt/scene/Interpolator.h"
#include "rt/scene/SceneLoader.h"
#include "util/ImageExporter.h"
#include <QGLWidget>

#define VERTEX_SHADER_FILE "../src/hippie/shader/vertex.vs"
#define FRAGMENT_SHADER_FILE "../src/hippie/shader/fragment.fs"
#define SCREENSHOT_PREFIX "screenshot_"

class RenderWidget : public QGLWidget {

    Q_OBJECT

public:

    enum BVHUpdateMethod {
        REBUILD,
        REFIT,
    };

private:

    BVHBuilder * builder;
    Scene * scene;
    Buffer pixels;
    Buffer framePixels;
    GLuint board;
    GLuint texture;
    Vec2i prevTexSize;
    Vec2i texSize;
    GLuint program;
    GLuint vertexShader;
    GLuint fragmentShader;
    QPoint lastMousePosition;
    QString screenshotsDir;

    HipBVH * bvh;

    bool headlight;
    bool statsEnabled;
    bool animationPause;
    bool animationLoop;
    float animationFrameRate;
    float animationLength;
    int nodeSizeThreshold;

    BVHUpdateMethod updateMethod;

    void getSource(const QString & filename, QString & source);
    Vec2i getTexSize(const Vec2i & imageSize);

    void rebuildBVH(void);
    void deleteScene(void);

public:

    ImageExporter exporter;
    Renderer renderer;

    LBVHBuilder lbvhBuilder;
    HLBVHBuilder hlbvhBuilder;
    SBVHBuilder sbvhBuilder;
    PLOCBuilder plocBuilder;
    ATRBuilder atrBuilder;
    TRBuilder trBuilder;
    InsertionBuilder insertionBuilder;

    Interpolator interpolator;

    SceneLoader sceneLoader;
    Camera camera;

    Light light;

    RenderWidget(QWidget *parent);
    ~RenderWidget(void);

    void initializeGL(void);
    void paintGL(void);
    void resizeGL(int width, int height);

    void mousePressEvent(QMouseEvent * ev);
    void mouseMoveEvent(QMouseEvent * ev);
    void keyPressEvent(QKeyEvent * ev);

public slots:

    void openScene(
        const QString & staticFilename,
        const QStringList & dynamicFilenames
    );
    void closeScene(void);

    void setSBVH(void);
    void setPLOC(void);
    void setATR(void);
    void setTR(void);
    void setLBVH(void);
    void setHLBVH(void);
    void setInsertion(void);

    BVHUpdateMethod getUpdateMethod(void);
    void setUpdateMethod(BVHUpdateMethod updateMethod);

    void enableStats(bool enable);
    void takeScreenshot(void);

    void setAnimationFrameRate(float animatonFrameRate);
    void setAnimationLength(float animationLength);
    void setAnimationLoop(bool animationLoop);
    void setAnimationPause(bool animationPause);

    void setScreenshotsDirectory(const QString & screenshotsDir);

signals:

    void changedNodeSizeBounds(int min, int max, int step);

};

#endif /* _RENDER_WIDGET_H_ */
