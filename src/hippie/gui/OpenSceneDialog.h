/**
 * \file	OpenSceneDialog.h
 * \author	Daniel Meister
 * \date	2014/09/09
 * \brief	OpenSceneDialog class header file.
 */

#ifndef _OPEN_SCENE_DIALOG_H_
#define _OPEN_SCENE_DIALOG_H_

#include <QDialog>
#include "ui_OpenSceneDialog.h"

class OpenSceneDialog : public QDialog {

    Q_OBJECT

private:

    Ui::OpenSceneDialog ui;

public:

    OpenSceneDialog(QWidget *parent = 0);
    ~OpenSceneDialog(void);

public slots:

    void openScene(void);

signals:

    void openScene(
        const QString & staticFilename,
        const QStringList & dynamicFilenames
    );

    void updateGUI(bool staticScene, bool dynamicScene);

};

#endif /* _OPEN_SCENE_DIALOG_H_ */
