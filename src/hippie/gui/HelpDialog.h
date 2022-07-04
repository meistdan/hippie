/**
 * \file	HelpDialog.h
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	HelpDialog class header file.
 */

#ifndef _HELP_DIALOG_H_
#define _HELP_DIALOG_H_

#include <QDialog>
#include "ui_HelpDialog.h"

class HelpDialog : public QDialog {

    Q_OBJECT

private:

    Ui::HelpDialog ui;

public:

    HelpDialog(QWidget *parent = 0);
    ~HelpDialog(void);

};

#endif /* _HELP_DIALOG_H_ */
