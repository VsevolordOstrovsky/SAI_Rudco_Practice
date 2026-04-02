#ifndef LIN_REG_H
#define LIN_REG_H

#include <QMainWindow>
#include "scr/qcustomplot/qcustomplot.h"
#include <QVector>
#include <math.h>

QT_BEGIN_NAMESPACE
namespace Ui { class lin_reg; }
QT_END_NAMESPACE

class lin_reg : public QMainWindow
{
    Q_OBJECT

public:
    lin_reg(QWidget *parent = nullptr);
    ~lin_reg();

private slots:
    void createGraph();
    void onLoadExcelFile();
    void ChangeOption(int index);

private:
    Ui::lin_reg *ui;

    int ChangeOptionFlag = 0;

    QCustomPlot *_graphRegres;
    QVector<double> X, Y, X_red, Y_red;

    void parseExcelFile(const QString &filePath);
    void CalAndShowLinRegg(QVector<double>& X, QVector<double>& Y);
    void CalAndShowKvRegg(QVector<double>& X, QVector<double>& Y);

};

#endif // LIN_REG_H
