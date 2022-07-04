/**
 * \file	RenderParamsWindow.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	RenderParamsWindow class header file.
 */

#ifndef _RENDER_PARAMS_WINDOW_H_
#define _RENDER_PARAMS_WINDOW_H_

#include <QDialog>
#include "ui_RenderParamsWindow.h"

class RenderParamsWindow : public QDialog {

    Q_OBJECT

private:

    Ui::RenderParamsWindow ui;

    private slots:

    void updateNodeSize(int nodeSize);
    void updateThermalThreshold(int thermalThreshold);
    void updateRecursionDepth(int recursionDepth);
    void updatePrimarySamples(int primarySamples);
    void updateAOSamples(int aoSamples);
    void updateShadowSamples(int shadowSamples);
    void updateAORadius(int aoRadius);
    void updateShadowRadius(int shadowRadius);

public:

    RenderParamsWindow(QWidget *parent = 0);
    ~RenderParamsWindow(void);

    void loadNodeSizeValue(int nodeSize);
    void loadThermalThresholdValue(int thermalThreshold);
    void loadRecursionDepthValue(int recursionDepth);
    void loadPrimarySamplesValue(int primarySamples);
    void loadAOSamplesValue(int aoSamples);
    void loadShadowSamplesValue(int shadowSamples);
    void loadAORadius(int aoRadius);
    void loadShadowRadius(int shadowRadius);

    public slots:

    void loadNodeSizeBounds(int min, int max, int step);

signals:

    void changedNodeSize(int nodeSize);
    void changedThermalThreshold(int thermalThreshold);
    void changedRecursionDepth(int recursionDepth);
    void changedPrimarySamples(int primarySamples);
    void changedAOSamples(int aoSamples);
    void changedDiffuseSamples(int diffuseSamples);
    void changedShadowSamples(int shadowSamples);
    void changedAORadius(int aoRadius);
    void changedShadowRadius(int shadowRadius);

};

#endif /* _RENDER_PARAMS_WINDOW_H_ */
