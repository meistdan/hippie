/**
 * \file	OpenSceneDialog.cpp
 * \author	Daniel Meister
 * \date	2014/09/09
 * \brief	OpenSceneDialog class source file.
 */

#include "OpenSceneDialog.h"
#include <QFileDialog>
#include <QDir>

OpenSceneDialog::OpenSceneDialog(QWidget *parent) : QDialog(parent) {
    ui.setupUi(this);
    setFixedSize(size());
}

OpenSceneDialog::~OpenSceneDialog() {
}

void OpenSceneDialog::openScene() {

    QString staticFilename;
    QStringList dynamicFilenames;

    bool staticScene = ui.checkBoxStaticScene->isChecked();
    bool dynamicScene = ui.checkBoxDynamicScene->isChecked();

    if (staticScene || dynamicScene) {

        // Close the dialog.
        close();

        // Static scene.
        if (staticScene) {
            if (QDir("../data/scenes").exists())
                staticFilename = QFileDialog::getOpenFileName(this,
                    "Select static scene file.", "../data/scenes", "Scenes (*.obj *.ply)");
            else
                staticFilename = QFileDialog::getOpenFileName(this,
                    "Select static scene file.", "", "Scenes (*.obj *.ply)");
        }

        // Dynamic scene.
        if (dynamicScene) {
            if (QDir("../data/scenes").exists())
                dynamicFilenames = QFileDialog::getOpenFileNames(this,
                    "Select dynamic scene files.", "../data/scenes", "Scenes (*.obj *.ply)");
            else
                dynamicFilenames = QFileDialog::getOpenFileNames(this,
                    "Select dynamic scene files.", "", "Scenes (*.obj *.ply)");
        }

        // Open scene.
        emit openScene(staticFilename, dynamicFilenames);

        // Inform main window.
        emit updateGUI(!staticFilename.isEmpty(), dynamicFilenames.size() > 1);

    }

}
