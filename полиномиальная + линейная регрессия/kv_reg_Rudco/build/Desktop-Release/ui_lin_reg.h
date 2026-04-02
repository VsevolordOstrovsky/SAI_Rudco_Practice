/********************************************************************************
** Form generated from reading UI file 'lin_reg.ui'
**
** Created by: Qt User Interface Compiler version 6.4.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_LIN_REG_H
#define UI_LIN_REG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_lin_reg
{
public:
    QWidget *centralwidget;
    QHBoxLayout *horizontalLayout;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *lin_reg)
    {
        if (lin_reg->objectName().isEmpty())
            lin_reg->setObjectName("lin_reg");
        lin_reg->resize(800, 600);
        centralwidget = new QWidget(lin_reg);
        centralwidget->setObjectName("centralwidget");
        horizontalLayout = new QHBoxLayout(centralwidget);
        horizontalLayout->setObjectName("horizontalLayout");
        lin_reg->setCentralWidget(centralwidget);
        menubar = new QMenuBar(lin_reg);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 800, 20));
        lin_reg->setMenuBar(menubar);
        statusbar = new QStatusBar(lin_reg);
        statusbar->setObjectName("statusbar");
        lin_reg->setStatusBar(statusbar);

        retranslateUi(lin_reg);

        QMetaObject::connectSlotsByName(lin_reg);
    } // setupUi

    void retranslateUi(QMainWindow *lin_reg)
    {
        lin_reg->setWindowTitle(QCoreApplication::translate("lin_reg", "lin_reg", nullptr));
    } // retranslateUi

};

namespace Ui {
    class lin_reg: public Ui_lin_reg {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_LIN_REG_H
