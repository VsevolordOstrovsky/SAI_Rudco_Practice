QT       += core gui printsupport

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

include(./scr/qcustomplot/CUSTOMPLOT.pri)

SOURCES += \
    main.cpp \
    lin_reg.cpp \

HEADERS += \
    lin_reg.h \

FORMS += \
    lin_reg.ui
