/**
 * \file	MainWindow.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	MainWindow class source file.
 */

#include <QFileDialog>
#include "Environment/AppEnvironment.h"
#include "util/Logger.h"
#include "MainWindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);
    setCentralWidget(ui.glwidget);
    bvhGroup = new QActionGroup(this);
    ui.actionLBVH->setActionGroup(bvhGroup);
    ui.actionHLBVH->setActionGroup(bvhGroup);
    ui.actionSBVH->setActionGroup(bvhGroup);
    ui.actionPLOC->setActionGroup(bvhGroup);
    ui.actionATR->setActionGroup(bvhGroup);
    ui.actionTR->setActionGroup(bvhGroup);
    ui.actionInsertion->setActionGroup(bvhGroup);
    ui.actionImport->setActionGroup(bvhGroup);
    rayGroup = new QActionGroup(this);
    ui.actionPrimary_Rays->setActionGroup(rayGroup);
    ui.actionShadow_Rays->setActionGroup(rayGroup);
    ui.actionAO_Rays->setActionGroup(rayGroup);
    ui.actionPath_Rays->setActionGroup(rayGroup);
    ui.actionPseudocolor_Rays->setActionGroup(rayGroup);
    ui.actionThermal_Rays->setActionGroup(rayGroup);
    updateGroup = new QActionGroup(this);
    ui.actionRebuild->setActionGroup(updateGroup);
    ui.actionRefit->setActionGroup(updateGroup);
    // Scene, BVH and camera.
    connect(&openSceneDialog, SIGNAL(openScene(const QString&, const QStringList&)),
        ui.glwidget, SLOT(openScene(const QString&, const QStringList&)));
    connect(&openSceneDialog, SIGNAL(updateGUI(bool, bool)), this, SLOT(updateGUI(bool, bool)));
    // Rendering parameters.
    connect(&renderParamsWindow, SIGNAL(changedNodeSize(int)), this, SLOT(updateNodeSize(int)));
    connect(&renderParamsWindow, SIGNAL(changedThermalThreshold(int)), this, SLOT(updateThermalThreshold(int)));
    connect(&renderParamsWindow, SIGNAL(changedRecursionDepth(int)), this, SLOT(updateRecursionDepth(int)));
    connect(&renderParamsWindow, SIGNAL(changedPrimarySamples(int)), this, SLOT(updatePrimarySamples(int)));
    connect(&renderParamsWindow, SIGNAL(changedAOSamples(int)), this, SLOT(updateAOSamples(int)));
    connect(&renderParamsWindow, SIGNAL(changedShadowSamples(int)), this, SLOT(updateShadowSamples(int)));
    connect(&renderParamsWindow, SIGNAL(changedAORadius(int)), this, SLOT(updateAORadius(int)));
    connect(&renderParamsWindow, SIGNAL(changedShadowRadius(int)), this, SLOT(updateShadowRadius(int)));
    connect(ui.glwidget, SIGNAL(changedNodeSizeBounds(int, int, int)),
        &renderParamsWindow, SLOT(loadNodeSizeBounds(int, int, int)));
    // Light.
    connect(&lightWindow, SIGNAL(changedX(int)), this, SLOT(updateLightX(int)));
    connect(&lightWindow, SIGNAL(changedY(int)), this, SLOT(updateLightY(int)));
    connect(&lightWindow, SIGNAL(changedZ(int)), this, SLOT(updateLightZ(int)));
    connect(&lightWindow, SIGNAL(changedKeyValue(int)), this, SLOT(updateKeyValue(int)));
    connect(&lightWindow, SIGNAL(changedWhitePoint(int)), this, SLOT(updateWhitePoint(int)));
}

MainWindow::~MainWindow() {
    delete bvhGroup;
    delete rayGroup;
}

void MainWindow::init() {

    int width, height;
    bool stats, animPause;
    QString bvhMethod, bvhUpdateMethod, staticSceneFilename, sceneFilefilter, screenshotDir;
    QStringList dynamicSceneFilenames;

    // Stats.
    Environment::getInstance()->getBoolValue("Application.stats", stats);
    ui.glwidget->enableStats(stats);

    // Resolution.
    Environment::getInstance()->getIntValue("Resolution.width", width);
    Environment::getInstance()->getIntValue("Resolution.height", height);
    setFixedSize(QSize(width, height));

    // BVH builder.
    Environment::getInstance()->getStringValue("Bvh.method", bvhMethod);
    if (bvhMethod == "lbvh") {
        ui.actionLBVH->setChecked(true);
        ui.glwidget->setLBVH();
    }
    else if (bvhMethod == "hlbvh") {
        ui.actionHLBVH->setChecked(true);
        ui.glwidget->setHLBVH();
    }
    else if (bvhMethod == "sbvh") {
        ui.actionSBVH->setChecked(true);
        ui.glwidget->setSBVH();
    }
    else if (bvhMethod == "ploc") {
        ui.actionPLOC->setChecked(true);
        ui.glwidget->setPLOC();
    }
    else if (bvhMethod == "atr") {
        ui.actionATR->setChecked(true);
        ui.glwidget->setATR();
    }
    else if (bvhMethod == "tr") {
        ui.actionTR->setChecked(true);
        ui.glwidget->setTR();
    }
    else if (bvhMethod == "insertion") {
        ui.actionInsertion->setChecked(true);
        ui.glwidget->setInsertion();
    }

    // Ray type.
    QString rayType;
    Environment::getInstance()->getStringValue("Renderer.rayType", rayType);
    if (rayType == "primary")
        ui.actionPrimary_Rays->setChecked(true);
    else if (rayType == "shadow")
        ui.actionShadow_Rays->setChecked(true);
    else if (rayType == "ao")
        ui.actionAO_Rays->setChecked(true);
    else if (rayType == "path")
        ui.actionPath_Rays->setChecked(true);
    else if (rayType == "pseudocolor")
        ui.actionPseudocolor_Rays->setChecked(true);
    else
        ui.actionThermal_Rays->setChecked(true);
    
    // Update method.
    Environment::getInstance()->getStringValue("Bvh.update", bvhUpdateMethod);
    if (bvhUpdateMethod == "refit") {
        ui.glwidget->setUpdateMethod(RenderWidget::REFIT);
        ui.actionRefit->setChecked(true);
    }
    else {
        ui.glwidget->setUpdateMethod(RenderWidget::REBUILD);
        ui.actionRebuild->setChecked(true);
    }

    // Animation.
    Environment::getInstance()->getBoolValue("Animation.pause", animPause);
    ui.glwidget->setAnimationPause(animPause);

    // Screenshots.
    Environment::getInstance()->getStringValue("Screenshots.directory", screenshotDir);
    if (QDir(screenshotDir).exists()) ui.glwidget->setScreenshotsDirectory(screenshotDir);

    // Scene files.
    Environment::getInstance()->getStringValue("Scene.filename", staticSceneFilename);
    Environment::getInstance()->getStringValue("Scene.filefilter", sceneFilefilter);

    // Scene filenames.
    QFileInfo sceneInfo = QFileInfo(QFile(sceneFilefilter));
    QDir sceneDir = sceneInfo.dir();
    QStringList sceneFilter(sceneInfo.fileName());
    dynamicSceneFilenames = sceneDir.entryList(sceneFilter);
    for (QStringList::iterator i = dynamicSceneFilenames.begin(); i != dynamicSceneFilenames.end(); ++i)
        *i = sceneInfo.absolutePath() + "/" + *i;

    // Open scene.
    ui.glwidget->openScene(staticSceneFilename, dynamicSceneFilenames);

    // Update GUI.
    updateGUI(!staticSceneFilename.isEmpty(), dynamicSceneFilenames.size() > 1);

}

void MainWindow::closeEvent(QCloseEvent * ev) {
    renderParamsWindow.close();
    helpDialog.close();
    aboutDialog.close();
    lightWindow.close();
}

void MainWindow::showAboutDialog() {
    aboutDialog.show();
}

void MainWindow::showHelpDialog() {
    helpDialog.show();
}

void MainWindow::showRenderParamsWindow() {
    renderParamsWindow.loadNodeSizeValue(ui.glwidget->renderer.getNodeSizeThreshold());
    renderParamsWindow.loadThermalThresholdValue(ui.glwidget->renderer.getThermalThreshold());
    renderParamsWindow.loadRecursionDepthValue(ui.glwidget->renderer.getRecursionDepth());
    renderParamsWindow.loadPrimarySamplesValue(ui.glwidget->renderer.getNumberOfPrimarySamples());
    renderParamsWindow.loadAOSamplesValue(ui.glwidget->renderer.getNumberOfAOSamples());
    renderParamsWindow.loadShadowSamplesValue(ui.glwidget->renderer.getNumberOfShadowSamples());
    renderParamsWindow.loadAORadius(int(ui.glwidget->renderer.getAORadius() * AO_RADIUS_SCALE));
    renderParamsWindow.loadShadowRadius(int(ui.glwidget->renderer.getShadowRadius() * SHADOW_RADIUS_SCALE));
    renderParamsWindow.show();
}

void MainWindow::showLightWindow() {
    lightWindow.setX(int(ui.glwidget->light.x * LIGHT_POSITION_SCALE));
    lightWindow.setY(int(ui.glwidget->light.y * LIGHT_POSITION_SCALE));
    lightWindow.setZ(int(ui.glwidget->light.z * LIGHT_POSITION_SCALE));
    lightWindow.setKeyValue(int(ui.glwidget->renderer.getKeyValue() * KEY_VALUE_SCALE));
    lightWindow.setWhitePoint(int(ui.glwidget->renderer.getWhitePoint() * WHITE_POINT_SCALE));
    lightWindow.show();
}

void MainWindow::setATR() {
    ui.actionRebuild->setEnabled(true);
    ui.glwidget->setATR();
}

void MainWindow::setTR() {
    ui.actionRebuild->setEnabled(true);
    ui.glwidget->setTR();
}

void MainWindow::setSBVH() {
    ui.actionRebuild->setEnabled(true);
    ui.glwidget->setSBVH();
}

void MainWindow::setPLOC() {
    ui.actionRebuild->setEnabled(true);
    ui.glwidget->setPLOC();
}

void MainWindow::setLBVH() {
    ui.actionRebuild->setEnabled(true);
    ui.glwidget->setLBVH();
}

void MainWindow::setHLBVH() {
    ui.actionRebuild->setEnabled(true);
    ui.glwidget->setHLBVH();
}

void MainWindow::setInsertion() {
    ui.actionRebuild->setEnabled(true);
    ui.glwidget->setInsertion();
}

void MainWindow::setRefit() {
    ui.actionRefit->setChecked(true);
    ui.glwidget->setUpdateMethod(RenderWidget::REFIT);
}

void MainWindow::setRebuild() {
    ui.actionRebuild->setChecked(true);
    ui.glwidget->setUpdateMethod(RenderWidget::REBUILD);
}

void MainWindow::setPrimaryRays() {
    ui.glwidget->renderer.setRayType(Renderer::PRIMARY_RAYS);
}

void MainWindow::setShadowRays() {
    ui.glwidget->renderer.setRayType(Renderer::SHADOW_RAYS);
}

void MainWindow::setAORays() {
    ui.glwidget->renderer.setRayType(Renderer::AO_RAYS);
}

void MainWindow::setPathRays() {
    ui.glwidget->renderer.setRayType(Renderer::PATH_RAYS);
}

void MainWindow::setPseudocolorRays() {
    ui.glwidget->renderer.setRayType(Renderer::PSEUDOCOLOR_RAYS);
}

void MainWindow::setThermalRays() {
    ui.glwidget->renderer.setRayType(Renderer::THERMAL_RAYS);
}

void MainWindow::openScene() {
    openSceneDialog.show();
}

void MainWindow::closeScene() {
    updateGUI(false, false);
    ui.glwidget->closeScene();
}

void MainWindow::enableStats(bool enable) {
    ui.glwidget->enableStats(enable);
}

void MainWindow::takeScreenshot() {
    ui.glwidget->takeScreenshot();
}

void MainWindow::resetCamera() {
    ui.glwidget->camera.reset();
    ui.glwidget->renderer.resetFrameIndex();
}

void MainWindow::updateGUI(bool staticScene, bool dynamicScene) {
    ui.actionOpen_Scene->setEnabled(true);
    ui.actionClose_Scene->setEnabled(false);
    ui.actionReset_Camera->setEnabled(false);
    ui.actionScreenshot->setEnabled(false);
    ui.actionParameters->setEnabled(false);
    updateGroup->setEnabled(false);
    if (staticScene || dynamicScene) {
        ui.actionOpen_Scene->setEnabled(false);
        ui.actionClose_Scene->setEnabled(true);
        ui.actionReset_Camera->setEnabled(true);
        ui.actionScreenshot->setEnabled(true);
        ui.actionParameters->setEnabled(true);
        if (dynamicScene) updateGroup->setEnabled(true);
    }
}

void MainWindow::updateNodeSize(int nodeSize) {
    ui.glwidget->renderer.setNodeSizeThreshold(nodeSize);
}

void MainWindow::updateThermalThreshold(int thermalThreshold) {
    ui.glwidget->renderer.setThermalThreshold(thermalThreshold);
}

void MainWindow::updateRecursionDepth(int recursionDepth) {
    ui.glwidget->renderer.setRecursionDepth(recursionDepth);
}

void MainWindow::updatePrimarySamples(int primarySamples) {
    ui.glwidget->renderer.setNumberOfPrimarySamples(primarySamples);
}

void MainWindow::updateAOSamples(int aoSamples) {
    ui.glwidget->renderer.setNumberOfAOSamples(aoSamples);
}

void MainWindow::updateShadowSamples(int shadowSamples) {
    ui.glwidget->renderer.setNumberOfShadowSamples(shadowSamples);
}

void MainWindow::updateAORadius(int aoRadius) {
    ui.glwidget->renderer.setAORadius(aoRadius / AO_RADIUS_SCALE);
}

void MainWindow::updateShadowRadius(int shadowRadius) {
    ui.glwidget->renderer.setShadowRadius(shadowRadius / SHADOW_RADIUS_SCALE);
}

void MainWindow::updateLightX(int x) {
    ui.glwidget->light.x = x / LIGHT_POSITION_SCALE;
    ui.glwidget->renderer.resetFrameIndex();
}

void MainWindow::updateLightY(int y) {
    ui.glwidget->light.y = y / LIGHT_POSITION_SCALE;
    ui.glwidget->renderer.resetFrameIndex();
}

void MainWindow::updateLightZ(int z) {
    ui.glwidget->light.z = z / LIGHT_POSITION_SCALE;
    ui.glwidget->renderer.resetFrameIndex();
}

void MainWindow::updateKeyValue(int keyValue) {
    ui.glwidget->renderer.setKeyValue(keyValue / KEY_VALUE_SCALE);
}

void MainWindow::updateWhitePoint(int whitePoint) {
    ui.glwidget->renderer.setWhitePoint(whitePoint / WHITE_POINT_SCALE);
}
