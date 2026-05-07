#ifndef NEURALNETWORKWIDGET_H
#define NEURALNETWORKWIDGET_H

#include <QWidget>
#include <QVector>

// Класс для визуализации нейронной сети
class NeuralNetworkWidget : public QWidget
{
    Q_OBJECT

public:
    explicit NeuralNetworkWidget(QWidget *parent = nullptr);

    // Методы для установки данных о сети
    void setLayerSizes(const QVector<int>& sizes);           // Установка количества нейронов в слоях
    void setLayerOutputs(const QVector<QVector<double>>& outputs);  // Установка выходных значений нейронов
    void setWeights(const QVector<QVector<QVector<double>>>& weights); // Установка весов связей

protected:
    void paintEvent(QPaintEvent* event) override;  // Переопределение события отрисовки

private:
    QVector<int> layerSizes;                        // Размеры слоев
    QVector<QVector<double>> layerOutputs;          // Выходные значения нейронов
    QVector<QVector<QVector<double>>> weights;      // Веса связей [слой][нейрон][вес]
};

#endif // NEURALNETWORKWIDGET_H
