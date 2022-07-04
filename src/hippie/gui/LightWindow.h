/**
 * \file	LightWindow.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	LightWindow class header file.
 */

#ifndef _LIGHT_WINDOW_H_
#define _LIGHT_WINDOW_H_

#include <QDialog>
#include "ui_LightWindow.h"

class LightWindow : public QDialog {

    Q_OBJECT

private:

    Ui::LightWindow ui;

    private slots:

    void updateX(int x);
    void updateY(int y);
    void updateZ(int z);
    void updateKeyValue(int keyValue);
    void updateWhitePoint(int whitePoint);

public:

    LightWindow(QWidget *parent = 0);
    ~LightWindow(void);

    void setX(int x);
    void setY(int y);
    void setZ(int z);
    void setKeyValue(int keyValue);
    void setWhitePoint(int whitePoint);

signals:

    void changedX(int x);
    void changedY(int y);
    void changedZ(int z);
    void changedKeyValue(int keyValue);
    void changedWhitePoint(int whitePoint);
};

#endif /* _LIGHT_WINDOW_H_ */
