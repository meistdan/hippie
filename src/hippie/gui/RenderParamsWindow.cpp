#include "RenderParamsWindow.h"

void RenderParamsWindow::updateNodeSize(int nodeSize) {
    emit changedNodeSize(nodeSize);
}

void RenderParamsWindow::updateThermalThreshold(int thermalThreshold) {
    emit changedThermalThreshold(thermalThreshold);
}

void RenderParamsWindow::updateRecursionDepth(int recursionDepth) {
    emit changedRecursionDepth(recursionDepth);
}

void RenderParamsWindow::updatePrimarySamples(int primarySamples) {
    emit changedPrimarySamples(primarySamples);
}

void RenderParamsWindow::updateAOSamples(int aoSamples) {
    emit changedAOSamples(aoSamples);
}

void RenderParamsWindow::updateShadowSamples(int shadowSamples) {
    emit changedShadowSamples(shadowSamples);
}

void RenderParamsWindow::updateAORadius(int aoRadius) {
    emit changedAORadius(aoRadius);
}

void RenderParamsWindow::updateShadowRadius(int shadowRadius) {
    emit changedShadowRadius(shadowRadius);
}

RenderParamsWindow::RenderParamsWindow(QWidget *parent) {
    ui.setupUi(this);
    QWidget::move(20, 200);
    connect(ui.sliderNodeSize, SIGNAL(valueChanged(int)), this, SLOT(updateNodeSize(int)));
    connect(ui.sliderThermalThreshold, SIGNAL(valueChanged(int)), this, SLOT(updateThermalThreshold(int)));
    connect(ui.sliderRecursionDepth, SIGNAL(valueChanged(int)), this, SLOT(updateRecursionDepth(int)));
    connect(ui.sliderPrimarySamples, SIGNAL(valueChanged(int)), this, SLOT(updatePrimarySamples(int)));
    connect(ui.sliderAOSamples, SIGNAL(valueChanged(int)), this, SLOT(updateAOSamples(int)));
    connect(ui.sliderShadowSamples, SIGNAL(valueChanged(int)), this, SLOT(updateShadowSamples(int)));
    connect(ui.sliderAORadius, SIGNAL(valueChanged(int)), this, SLOT(updateAORadius(int)));
    connect(ui.sliderShadowRadius, SIGNAL(valueChanged(int)), this, SLOT(updateShadowRadius(int)));
}

RenderParamsWindow::~RenderParamsWindow() {
}

void RenderParamsWindow::loadNodeSizeValue(int nodeSize) {
    ui.sliderNodeSize->setValue(nodeSize);
}

void RenderParamsWindow::loadThermalThresholdValue(int thermalThreshold) {
    ui.sliderThermalThreshold->setValue(thermalThreshold);
}

void RenderParamsWindow::loadRecursionDepthValue(int recursionDepth) {
    ui.sliderRecursionDepth->setValue(recursionDepth);
}

void RenderParamsWindow::loadPrimarySamplesValue(int primarySamples) {
    ui.sliderPrimarySamples->setValue(primarySamples);
}

void RenderParamsWindow::loadAOSamplesValue(int aoSamples) {
    ui.sliderAOSamples->setValue(aoSamples);
}

void RenderParamsWindow::loadShadowSamplesValue(int shadowSamples) {
    ui.sliderShadowSamples->setValue(shadowSamples);
}

void RenderParamsWindow::loadAORadius(int aoRadius) {
    ui.sliderAORadius->setValue(aoRadius);
}

void RenderParamsWindow::loadShadowRadius(int shadowRadius) {
    ui.sliderShadowRadius->setValue(shadowRadius);
}

void RenderParamsWindow::loadNodeSizeBounds(int min, int max, int step) {
    ui.sliderNodeSize->setMinimum(min);
    ui.sliderNodeSize->setMaximum(max);
    ui.sliderNodeSize->setSingleStep(step);
}
