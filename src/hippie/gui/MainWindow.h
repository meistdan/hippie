/**
 * \file	MainWindow.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	MainWindow class header file.
 */

#ifndef _MAIN_WINDOW_H_
#define _MAIN_WINDOW_H_

#include <QtWidgets/QMainWindow>
#include "AboutDialog.h"
#include "HelpDialog.h"
#include "LightWindow.h"
#include "OpenSceneDialog.h"
#include "RenderParamsWindow.h"
#include "ui_MainWindow.h"

#define AO_RADIUS_SCALE 40000.0f
#define SHADOW_RADIUS_SCALE 1000000.0f
#define LIGHT_POSITION_SCALE 1000.0f
#define KEY_VALUE_SCALE 500.0f
#define WHITE_POINT_SCALE 500.0f

class MainWindow : public QMainWindow {

    Q_OBJECT

private:

    QActionGroup * bvhGroup;
    QActionGroup * rayGroup;
    QActionGroup * updateGroup;

    Ui::MainWindowClass ui;

    AboutDialog aboutDialog;
    HelpDialog helpDialog;
    OpenSceneDialog openSceneDialog;
    RenderParamsWindow renderParamsWindow;
    LightWindow lightWindow;

    RayBuffer::MortonCodeMethod stringToMortonCodeMethod(const QString & mortonCodeMethod);

public:

    MainWindow(QWidget *parent = 0);
    ~MainWindow(void);

    void init(void);
    virtual void closeEvent(QCloseEvent * ev);

    public slots:

    void showAboutDialog(void);
    void showHelpDialog(void);
    void showRenderParamsWindow(void);
    void showLightWindow(void);

    void setATR(void);
    void setTR(void);
    void setSBVH(void);
    void setPLOC(void);
    void setLBVH(void);
    void setHLBVH(void);
    void setInsertion(void);

    void setRefit(void);
    void setRebuild(void);

    void setPrimaryRays(void);
    void setShadowRays(void);
    void setAORays(void);
    void setPathRays(void);
    void setPseudocolorRays(void);
    void setThermalRays(void);

    void openScene(void);
    void closeScene(void);

    void enableStats(bool enable);
    void takeScreenshot(void);

    void resetCamera(void);

    void updateGUI(bool staticScene, bool dynamicScene);
    void updateNodeSize(int nodeSize);
    void updateThermalThreshold(int thermalThreshold);
    void updateRecursionDepth(int recursionDepth);

    void updatePrimarySamples(int primarySamples);
    void updateAOSamples(int aoSamples);
    void updateShadowSamples(int shadowSamples);
    void updateAORadius(int aoRadius);
    void updateShadowRadius(int shadowRadius);

    void updateLightX(int x);
    void updateLightY(int y);
    void updateLightZ(int z);

    void updateKeyValue(int keyValue);
    void updateWhitePoint(int whitePoint);

};

#endif /* _MAIN_WINDOW_H_ */
