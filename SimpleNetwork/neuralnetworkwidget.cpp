#include "neuralnetworkwidget.h"
#include <QPainter>
#include <QtMath>

NeuralNetworkWidget::NeuralNetworkWidget(QWidget *parent)
    : QWidget(parent)
{
    setStyleSheet("background-color: #2a2a2a;");
    setAttribute(Qt::WA_OpaquePaintEvent);
}

void NeuralNetworkWidget::setLayerSizes(const QVector<int>& sizes) {
    layerSizes = sizes;
    update();
}

void NeuralNetworkWidget::setLayerOutputs(const QVector<QVector<double>>& outputs) {
    layerOutputs = outputs;
    update();
}

void NeuralNetworkWidget::setWeights(const QVector<QVector<QVector<double>>>& weightMatrix) {
    weights = weightMatrix;
    update();
}

void NeuralNetworkWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    if (layerSizes.isEmpty()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "Нейросеть не инициализирована");
        return;
    }

    // Получаем размеры области рисования
    int width = this->width() - 20;
    int height = this->height() - 20;

    if (width <= 10 || height <= 10) return;

    int startX = 60;
    int endX = width - 60;
    int layerSpacing = (endX - startX) / (layerSizes.size() - 1);

    if (layerSpacing < 10) return;

    // Цвета для слоев
    QVector<QColor> layerColors = {
        QColor(100, 150, 255),  // Входной слой - синий
        QColor(100, 255, 150),  // Скрытый слой 1 - зеленый
        QColor(255, 200, 100),  // Скрытый слой 2 - оранжевый
        QColor(255, 100, 100)   // Выходной слой - красный
    };

    // Рисуем связи
    QPen connectionPen;
    connectionPen.setCapStyle(Qt::RoundCap);

    for (int layer = 0; layer < layerSizes.size() - 1; layer++) {
        int x1 = startX + layer * layerSpacing;
        int x2 = startX + (layer + 1) * layerSpacing;

        int spacing1 = height / (layerSizes[layer] + 1);
        int spacing2 = height / (layerSizes[layer + 1] + 1);

        for (int i = 0; i < layerSizes[layer]; i++) {
            int y1 = spacing1 * (i + 1);

            for (int j = 0; j < layerSizes[layer + 1]; j++) {
                int y2 = spacing2 * (j + 1);

                // Получаем вес связи
                double weight = 0.0;
                if (layer + 1 < weights.size() && j < weights[layer + 1].size()
                    && i < weights[layer + 1][j].size()) {
                    weight = weights[layer + 1][j][i];
                }

                // Определяем цвет и толщину линии
                if (weight > 0) {
                    int intensity = qMin(255, int(100 + qAbs(weight) * 155));
                    connectionPen.setColor(QColor(0, intensity, 0));
                    connectionPen.setWidth(1 + int(qAbs(weight) * 2));
                } else {
                    int intensity = qMin(255, int(100 + qAbs(weight) * 155));
                    connectionPen.setColor(QColor(intensity, 0, 0));
                    connectionPen.setWidth(1 + int(qAbs(weight) * 2));
                }

                painter.setPen(connectionPen);
                painter.drawLine(x1, y1, x2, y2);

                // Рисуем текст веса
                if (qAbs(weight) > 0.2) {
                    painter.setPen(Qt::white);
                    QFont font = painter.font();
                    font.setPointSize(7);
                    painter.setFont(font);
                    int midX = (x1 + x2) / 2;
                    int midY = (y1 + y2) / 2;
                    painter.drawText(midX - 12, midY - 5,
                                     QString::number(weight, 'f', 2));
                }
            }
        }
    }

    // Рисуем нейроны
    for (int layer = 0; layer < layerSizes.size(); layer++) {
        int x = startX + layer * layerSpacing;
        int spacing = height / (layerSizes[layer] + 1);

        for (int i = 0; i < layerSizes[layer]; i++) {
            int y = spacing * (i + 1);
            int radius = 25;

            // Получаем выходное значение нейрона
            double outputValue = 0.0;
            if (layer < layerOutputs.size() && i < layerOutputs[layer].size()) {
                outputValue = layerOutputs[layer][i];
            }

            // Цвет нейрона зависит от значения выхода
            QColor neuronColor = layerColors[layer % layerColors.size()];
            int intensity = 80 + int(outputValue * 175);
            neuronColor = QColor(
                qMin(255, (neuronColor.red() * intensity) / 255),
                qMin(255, (neuronColor.green() * intensity) / 255),
                qMin(255, (neuronColor.blue() * intensity) / 255)
                );

            painter.setBrush(QBrush(neuronColor));
            painter.setPen(QPen(Qt::black, 2));
            painter.drawEllipse(x - radius/2, y - radius/2, radius, radius);

            // Рисуем выходное значение
            painter.setPen(Qt::white);
            QFont font = painter.font();
            font.setPointSize(8);
            font.setBold(true);
            painter.setFont(font);
            painter.drawText(x - 12, y + 4,
                             QString::number(outputValue, 'f', 3));

            // Название слоя
            if (i == 0) {
                QString layerName;
                switch(layer) {
                case 0: layerName = "Input\n(2)"; break;
                case 1: layerName = "Hidden 1\n(3)"; break;
                case 2: layerName = "Hidden 2\n(2)"; break;
                case 3: layerName = "Output\n(1)"; break;
                }
                painter.setPen(QColor(200, 200, 255));
                font.setPointSize(7);
                font.setBold(false);
                painter.setFont(font);
                painter.drawText(x - 20, y - 30, layerName);
            }
        }
    }
}
