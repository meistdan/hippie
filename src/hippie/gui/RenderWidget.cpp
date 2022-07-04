/**
 * \file	RenderWidget.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	RenderWidget class source file.
 */

#include <QFile>
#include <QKeyEvent>
#include "environment/AppEnvironment.h"
#include "Globals.h"
#include "RenderWidget.h"
#include "util/Logger.h"

void RenderWidget::getSource(const QString & filename, QString & source) {
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        logger(LOG_ERROR) << "ERROR <RenderWidget> Cannot open '" << filename << "'.\n";
        exit(EXIT_FAILURE);
    }
    QTextStream in(&file);
    QString line;
    while (!in.atEnd()) {
        line = in.readLine();
        source.append(line + "\n");
    }
    file.close();
}

Vec2i RenderWidget::getTexSize(const Vec2i & imageSize) {
    Vec2i result = Vec2i(1, 1);
    while (result.x < imageSize.x) result.x *= 2;
    while (result.y < imageSize.y) result.y *= 2;
    if (result.x >= 512 * 512 / result.y) result = imageSize;
    return result;
}

void RenderWidget::rebuildBVH() {
    if (scene) {
        delete bvh;
        bvh = builder->build(scene);
        renderer.resetFrameIndex();
    }
}

void RenderWidget::deleteScene() {
    if (scene) {
        delete scene;
        scene = nullptr;
    }
}

RenderWidget::RenderWidget(QWidget * parent) :
    QGLWidget(parent),
    scene(nullptr),
    bvh(nullptr),
    texture(0),
    program(0),
    vertexShader(0),
    fragmentShader(0),
    statsEnabled(true),
    animationPause(false),
    headlight(false),
    updateMethod(REBUILD)
{
    // Default builder.
    builder = &lbvhBuilder;
    // Light.
    QVector3D _light;
    Environment::getInstance()->getVectorValue("Scene.light", _light);
    light = Light(_light.x(), _light.y(), _light.z());
}

RenderWidget::~RenderWidget() {
    // Delete program and shaders.
    if (vertexShader) glDeleteShader(vertexShader);
    if (fragmentShader) glDeleteShader(fragmentShader);
    if (program) glDeleteProgram(program);
    // Delete texture.
    if (texture) glDeleteTextures(1, &texture);
    // Delete scene and BVHs.
    deleteScene();
    delete bvh;
}

void RenderWidget::initializeGL() {

    // Initialize GLEW.
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        logger(LOG_ERROR) << "ERROR <RenderWidget> glewInit() failed '" << (const char *)glewGetErrorString(err) << "'!\n";
        exit(EXIT_FAILURE);
    }

    // Initialize OpenGL.
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Create texture.
    texSize = prevTexSize = getTexSize(Vec2i(width(), height()));
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texSize.x, texSize.y, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create board VBO.
    float posAttrib[] = { -1.0f, -1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 0.0f, 1.0f,
                          -1.0f,  1.0f, 0.0f, 1.0f, 1.0f,  1.0f, 0.0f, 1.0f };
    glGenBuffers(1, &board);
    glBindBuffer(GL_ARRAY_BUFFER, board);
    glBufferData(GL_ARRAY_BUFFER, 24 * sizeof(float), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 16 * sizeof(float), posAttrib);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Shaders source code.
    QString vertexSource, fragmentSource;
    getSource(VERTEX_SHADER_FILE, vertexSource);
    getSource(FRAGMENT_SHADER_FILE, fragmentSource);
    QByteArray vertexSourceUtf = vertexSource.toUtf8();
    QByteArray fragmentSourceUtf = fragmentSource.toUtf8();
    const GLchar * vertexSourcePtr = vertexSourceUtf.constData();
    const GLchar * fragmentSourcePtr = fragmentSourceUtf.constData();

    // Vertex shader.
    GLint vertexStatus = GL_FALSE;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSourcePtr, nullptr);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vertexStatus);
    Q_ASSERT(vertexStatus == GL_TRUE);

    // Fragment shader.
    GLint fragmentStatus = GL_FALSE;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSourcePtr, nullptr);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fragmentStatus);
    Q_ASSERT(fragmentStatus == GL_TRUE);

    // Program and link.
    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // Set viewport.
    glViewport(0, 0, width(), height());

    // Check OpenGL errors.
    checkGLErrors();

}

void RenderWidget::paintGL() {

    if (scene) {

        // Size.
        Vec2i size(width(), height());

        // Time.
        float time = 0.0f;

        // Try cast to dynamic scene.
        DynamicScene * dynamicScene = dynamic_cast<DynamicScene*>(scene);

        // Update scene and BVH.
        if (scene->isDynamic() && !animationPause) {

            // Update geometry.
            time += interpolator.update(*dynamicScene);

            // Refit.
            if (updateMethod == REFIT) {
                time += bvh->update();
            }

            // Rebuild.
            else {
                time += builder->rebuild(*bvh);
            }

            renderer.resetFrameIndex();

        }

        // Render.
        scene->setLight(headlight ? camera.getPosition() : light);
        time += renderer.render(*scene, *bvh, camera, pixels, framePixels);

        // Texture size.
        texSize = getTexSize(size);

        // Image - texture size ratio.
        static Vec2f prevRatio;
        Vec2f ratio = Vec2f(size) / Vec2f(texSize);

        // Setup texture coordinates.
        float texAttrib[] = { 0.0f, 0.0f, ratio.x, 0.0f,  0.0f, ratio.y, ratio.x, ratio.y };

        // Display result.
        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_DEPTH_TEST);
        glActiveTexture(GL_TEXTURE0);
        glUseProgram(program);

        // Update texture coordinates.
        glBindBuffer(GL_ARRAY_BUFFER, board);
        if (prevRatio.x != ratio.x || prevRatio.y != ratio.y)
            glBufferSubData(GL_ARRAY_BUFFER, 16 * sizeof(float), 8 * sizeof(float), texAttrib);

        // Bind texture and resize if necessary.
        glBindTexture(GL_TEXTURE_2D, texture);
        if (prevTexSize.x != texSize.x || prevTexSize.y != texSize.y)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texSize.x, texSize.y, 0, GL_RGBA, GL_FLOAT, nullptr);

        // Data is already on the GPU => transfer to the texture.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, framePixels.getGLBuffer());
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.x, size.y, GL_RGBA, GL_FLOAT, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glUniform1i(glGetUniformLocation(program, "texSampler"), 0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, (void*)(16 * sizeof(GLfloat)));
        glEnableVertexAttribArray(3);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisableVertexAttribArray(3);
        glDisableVertexAttribArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glPopAttrib();

        // Set previous texture size.
        prevTexSize = texSize;

        // Check OpenGL errors.
        checkGLErrors();

        // Show stats.
        if (statsEnabled) {
            int numberOfRays = renderer.getNumberOfRays();
            renderText(20, 20, QString().sprintf("%d triangles, %.2f million rays", scene->getNumberOfTriangles(), numberOfRays * 1.0e-6f), QFont());
            renderText(20, 40, QString().sprintf("%.2f MRays/s, FPS %.2f", renderer.getTracePerformance(), 1.0f / time), QFont());
            renderText(20, 60, QString().sprintf("light [%.2f, %.2f, %.2f]", light.x, light.y, light.z), QFont());
            renderText(20, 80, QString().sprintf("keyValue %.2f, whitePoint %.2f", renderer.getKeyValue(), renderer.getWhitePoint()), QFont());
            if (scene->isDynamic()) renderText(20, 100, QString().sprintf("animation frame rate %.2f", dynamicScene->getFrameRate()), QFont());
            renderText(300, 20, QString().sprintf("position [%.2f, %.2f, %.2f]", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z), QFont());
            renderText(300, 40, QString().sprintf("direction [%.2f, %.2f, %.2f]", camera.getDirection().x, camera.getDirection().y, camera.getDirection().z), QFont());
            renderText(300, 60, QString().sprintf("angle %.2f", camera.getWheelAngle()), QFont());
            renderText(580, 20, QString().sprintf("AO radius %.3f, shadow radius %.3f, recursion depth %d", renderer.getAORadius(),
                renderer.getShadowRadius(), renderer.getRecursionDepth()), QFont());
            renderText(580, 40, QString().sprintf("%d primary samples, %d AO samples, %d shadow samples", renderer.getNumberOfPrimarySamples(), renderer.getNumberOfAOSamples(), renderer.getNumberOfShadowSamples()), QFont());
            renderText(580, 60, QString().sprintf("node size threshold %d, thermal threshold %d", renderer.getNodeSizeThreshold(), renderer.getThermalThreshold()), QFont());
            renderText(580, 80, QString().sprintf("BVH cost %.2f", bvh->getCost()), QFont());
        }

        // Update widget.
        update();

    }

}

void RenderWidget::resizeGL(int width, int height) {
    glViewport(0, 0, width, height);
    if (scene) {
        camera.setSize(Vec2i(width, height));
        renderer.resetFrameIndex();
    }
}

void RenderWidget::mousePressEvent(QMouseEvent * ev) {
    lastMousePosition = ev->pos();
}

void RenderWidget::mouseMoveEvent(QMouseEvent * ev) {
    if (scene) {
        QPoint pos = ev->pos();
        camera.addViewAngle((pos.x() - lastMousePosition.x()) / 6.0f);
        camera.addElevAngle((lastMousePosition.y() - pos.y()) / 6.0f);
        lastMousePosition = pos;
        renderer.resetFrameIndex();
    }
}

void RenderWidget::keyPressEvent(QKeyEvent * ev) {

    if (scene) {

        bool ctrlHeld = ev->modifiers() & Qt::ControlModifier;

        switch (ev->key()) {

        case Qt::Key_M: {
            camera.rotateCW();
            updateGL();
            renderer.resetFrameIndex();
            break;
        }

        case Qt::Key_N: {
            camera.rotateCCW();
            updateGL();
            renderer.resetFrameIndex();
            break;
        }

        case Qt::Key_W: {
            camera.moveForward();
            updateGL();
            renderer.resetFrameIndex();
            break;
        }

        case Qt::Key_A: {
            camera.turnLeft();
            updateGL();
            renderer.resetFrameIndex();
            break;
        }

        case Qt::Key_S: {
            camera.moveBackward();
            updateGL();
            renderer.resetFrameIndex();
            break;
        }

        case Qt::Key_D: {
            camera.turnRight();
            updateGL();
            renderer.resetFrameIndex();
            break;
        }

        case Qt::Key_H: {
            headlight = !headlight;
            updateGL();
            renderer.resetFrameIndex();
            break;
        }

        case Qt::Key_R: {
            if (!ctrlHeld) {
                DynamicScene * dynamicScene = dynamic_cast<DynamicScene*>(scene);
                if (dynamicScene) {
                    dynamicScene->resetTime();
                }
            }
            break;
        }

        case Qt::Key_Plus: {
            if (!ctrlHeld) {
                DynamicScene * dynamicScene = dynamic_cast<DynamicScene*>(scene);
                if (dynamicScene) {
                    dynamicScene->increaseFrameRate();
                    updateGL();
                }
            }
            break;
        }

        case Qt::Key_Minus: {
            if (!ctrlHeld) {
                DynamicScene * dynamicScene = dynamic_cast<DynamicScene*>(scene);
                if (dynamicScene) {
                    dynamicScene->decreaseFrameRate();
                    updateGL();
                }
            }
            break;
        }

        case Qt::Key_Space: {
            if (!ctrlHeld) {
                DynamicScene * dynamicScene = dynamic_cast<DynamicScene*>(scene);
                if (dynamicScene) animationPause = !animationPause;
            }
            break;
        }

        default: {
            ev->ignore();
            break;
        }
        }

    }
}

void RenderWidget::openScene(
    const QString & staticFilename,
    const QStringList & dynamicFilenames
) {

    // Close scene.
    closeScene();

    if (!staticFilename.isEmpty() || dynamicFilenames.size() > 1) {

        // Static scene.
        if (!staticFilename.isEmpty() && dynamicFilenames.size() <= 1) {
            scene = sceneLoader.loadStaticScene(staticFilename);
            emit changedNodeSizeBounds(1, scene->getNumberOfTriangles(), scene->getNumberOfTriangles() / 100);
        }

        // Dynamic scnee.
        else if (dynamicFilenames.size() > 1) {
            DynamicScene * dynamicScene = nullptr;
            if (staticFilename.isEmpty()) dynamicScene = sceneLoader.loadDynamicScene(dynamicFilenames);
            else dynamicScene = sceneLoader.loadDynamicScene(staticFilename, dynamicFilenames);
            dynamicScene->setFrameRate(animationFrameRate);
            dynamicScene->setLoop(animationLoop);
            scene = dynamicScene;
            emit changedNodeSizeBounds(1, scene->getNumberOfTriangles(), scene->getNumberOfTriangles() / 100);
        }

        // BVH.
        bvh = builder->build(scene);
        logger(LOG_INFO) << "INFO <RenderWidget> BVH cost is " << bvh->getCost() << ".\n";

        // Camera size.
        camera.setSize(Vec2i(width(), height()));

    }

}

void RenderWidget::closeScene() {
    deleteScene();
    delete bvh;
    renderer.resetFrameIndex();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void RenderWidget::setSBVH() {
    builder = &sbvhBuilder;
    rebuildBVH();
}

void RenderWidget::setPLOC() {
    builder = &plocBuilder;
    rebuildBVH();
}

void RenderWidget::setATR() {
    builder = &atrBuilder;
    rebuildBVH();
}

void RenderWidget::setTR() {
    builder = &trBuilder;
    rebuildBVH();
}

void RenderWidget::setLBVH() {
    builder = &lbvhBuilder;
    rebuildBVH();
}

void RenderWidget::setHLBVH() {
    builder = &hlbvhBuilder;
    rebuildBVH();
}

void RenderWidget::setInsertion() {
    builder = &insertionBuilder;
    rebuildBVH();
}

RenderWidget::BVHUpdateMethod RenderWidget::getUpdateMethod() {
    return updateMethod;
}

void RenderWidget::setUpdateMethod(BVHUpdateMethod updateMethod) {
    this->updateMethod = updateMethod;
    renderer.resetFrameIndex();
}

void RenderWidget::enableStats(bool enable) {
    statsEnabled = enable;
}

void RenderWidget::takeScreenshot() {
    QString filename;
    int i = -1;
    do {
        filename = screenshotsDir + "/" +
            SCREENSHOT_PREFIX + QString::number(++i) + ".png";
    } while (QFile(filename).exists());
    exporter.exportImage(width(), height(), framePixels, filename, true);
}

void RenderWidget::setAnimationFrameRate(float animationFrameRate) {
    if (animationFrameRate <= 0.0f || animationFrameRate > MAX_FRAME_RATE)
        logger(LOG_WARN) << "WARN <RenderWidget> Animation frame rate must be within range (0," << MAX_FRAME_RATE << "].\n";
    else
        this->animationFrameRate = animationFrameRate;
}

void RenderWidget::setAnimationLength(float animationLength) {
    if (animationLength <= 0.0f)
        logger(LOG_WARN) << "WARN <RenderWidget> Animation length must be positive.\n";
    else
        this->animationLength = animationLength;
}

void RenderWidget::setAnimationLoop(bool animationLoop) {
    this->animationLoop = animationLoop;
}

void RenderWidget::setAnimationPause(bool animationPause) {
    this->animationPause = animationPause;
}

void RenderWidget::setScreenshotsDirectory(const QString & screenshotsDir) {
    this->screenshotsDir = screenshotsDir;
}
