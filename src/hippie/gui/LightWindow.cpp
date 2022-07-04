#include "LightWindow.h"

void LightWindow::updateX(int x) {
    emit changedX(x);
}

void LightWindow::updateY(int y) {
    emit changedY(y);
}

void LightWindow::updateZ(int z) {
    emit changedZ(z);
}

void LightWindow::updateKeyValue(int keyValue) {
    emit changedKeyValue(keyValue);
}

void LightWindow::updateWhitePoint(int whitePoint) {
    emit changedWhitePoint(whitePoint);
}

LightWindow::LightWindow(QWidget *parent) {
    ui.setupUi(this);
    QWidget::move(20, 400);
    connect(ui.sliderX, SIGNAL(valueChanged(int)), this, SLOT(updateX(int)));
    connect(ui.sliderY, SIGNAL(valueChanged(int)), this, SLOT(updateY(int)));
    connect(ui.sliderZ, SIGNAL(valueChanged(int)), this, SLOT(updateZ(int)));
    connect(ui.sliderKeyValue, SIGNAL(valueChanged(int)), this, SLOT(updateKeyValue(int)));
    connect(ui.sliderWhitePoint, SIGNAL(valueChanged(int)), this, SLOT(updateWhitePoint(int)));
}

LightWindow::~LightWindow() {
}

void LightWindow::setX(int x) {
    ui.sliderX->setValue(x);
}

void LightWindow::setY(int y) {
    ui.sliderY->setValue(y);
}

void LightWindow::setZ(int z) {
    ui.sliderZ->setValue(z);
}

void LightWindow::setKeyValue(int keyValue) {
    ui.sliderKeyValue->setValue(keyValue);
}

void LightWindow::setWhitePoint(int whitePoint) {
    ui.sliderWhitePoint->setValue(whitePoint);
}
