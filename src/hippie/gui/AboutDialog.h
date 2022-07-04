/**
 * \file	AboutDialog.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	AboutDialog class header file.
 */

#ifndef _ABOUT_DIALOG_H_
#define _ABOUT_DIALOG_H_

#include <QDialog>
#include "ui_AboutDialog.h"

class AboutDialog : public QDialog {

    Q_OBJECT

private:

    Ui::AboutDialog ui;

public:

    AboutDialog(QWidget *parent = 0);
    ~AboutDialog(void);

};

#endif /* _ABOUT_DIALOG_H_ */
