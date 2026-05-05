#ifndef NEURALNETWORKWIDGET_H
#define NEURALNETWORKWIDGET_H

#include <QWidget>
#include <QVector>

class NeuralNetworkWidget : public QWidget
{
    Q_OBJECT

public:
    explicit NeuralNetworkWidget(QWidget *parent = nullptr);

    void setLayerSizes(const QVector<int>& sizes);
    void setLayerOutputs(const QVector<QVector<double>>& outputs);
    void setWeights(const QVector<QVector<QVector<double>>>& weights);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QVector<int> layerSizes;
    QVector<QVector<double>> layerOutputs;
    QVector<QVector<QVector<double>>> weights;
};

#endif // NEURALNETWORKWIDGET_H
