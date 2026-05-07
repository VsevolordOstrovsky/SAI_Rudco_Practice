#include "neuralnetworkwidget.h"
#include <QPainter>
#include <QtMath>

NeuralNetworkWidget::NeuralNetworkWidget(QWidget *parent)
    : QWidget(parent)
{
    // Устанавливаем темный фон для виджета
    setStyleSheet("background-color: #2a2a2a;");
    // Оптимизация отрисовки - виджет полностью непрозрачный
    setAttribute(Qt::WA_OpaquePaintEvent);
}

void NeuralNetworkWidget::setLayerSizes(const QVector<int>& sizes) {
    layerSizes = sizes;
    update();  // Запрашиваем перерисовку
}

void NeuralNetworkWidget::setLayerOutputs(const QVector<QVector<double>>& outputs) {
    layerOutputs = outputs;
    update();  // Запрашиваем перерисовку
}

void NeuralNetworkWidget::setWeights(const QVector<QVector<QVector<double>>>& weightMatrix) {
    weights = weightMatrix;
    update();  // Запрашиваем перерисовку
}

void NeuralNetworkWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);  // Включаем сглаживание

    // Очищаем фон темным цветом
    painter.fillRect(rect(), QColor("#2a2a2a"));

    // Если сеть не инициализирована, выводим сообщение
    if (layerSizes.isEmpty()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "Нейросеть не инициализирована");
        return;
    }

    // Получаем размеры области рисования с отступами
    int width = this->width() - 20;
    int height = this->height() - 20;

    if (width <= 10 || height <= 10) return;

    // Вычисляем начальную и конечную позиции по X для слоев
    int startX = 60;
    int endX = width - 60;
    int layerSpacing = (endX - startX) / (layerSizes.size() - 1);

    if (layerSpacing < 10) return;

    // Цвета для каждого слоя
    QVector<QColor> layerColors = {
        QColor(100, 150, 255),  // Входной слой - синий
        QColor(100, 255, 150),  // Скрытый слой 1 - зеленый
        QColor(255, 200, 100),  // Скрытый слой 2 - оранжевый
        QColor(255, 100, 100)   // Выходной слой - красный
    };

    // ========== РИСУЕМ СВЯЗИ (ЛИНИИ) МЕЖДУ НЕЙРОНАМИ ==========
    QPen connectionPen;
    connectionPen.setCapStyle(Qt::RoundCap);

    // Проходим по всем слоям кроме последнего (для каждого слоя рисуем связи со следующим)
    for (int layer = 0; layer < layerSizes.size() - 1; layer++) {
        // X-координаты текущего и следующего слоя
        int x1 = startX + layer * layerSpacing;
        int x2 = startX + (layer + 1) * layerSpacing;

        // Вычисляем вертикальные интервалы между нейронами
        int spacing1 = height / (layerSizes[layer] + 1);
        int spacing2 = height / (layerSizes[layer + 1] + 1);

        // Для каждого нейрона текущего слоя
        for (int i = 0; i < layerSizes[layer]; i++) {
            int y1 = spacing1 * (i + 1);  // Y-координата нейрона

            // Для каждого нейрона следующего слоя
            for (int j = 0; j < layerSizes[layer + 1]; j++) {
                int y2 = spacing2 * (j + 1);  // Y-координата нейрона

                // Получаем вес связи
                double weight = 0.0;
                if (layer + 1 < weights.size() && j < weights[layer + 1].size()
                    && i < weights[layer + 1][j].size()) {
                    weight = weights[layer + 1][j][i];
                }

                // Определяем цвет и толщину линии в зависимости от веса
                if (weight > 0) {
                    // Положительный вес - зеленый (интенсивность зависит от величины веса)
                    int intensity = qMin(255, int(100 + qAbs(weight) * 155));
                    connectionPen.setColor(QColor(0, intensity, 0));
                    connectionPen.setWidth(1 + int(qAbs(weight) * 2));
                } else {
                    // Отрицательный вес - красный (интенсивность зависит от величины веса)
                    int intensity = qMin(255, int(100 + qAbs(weight) * 155));
                    connectionPen.setColor(QColor(intensity, 0, 0));
                    connectionPen.setWidth(1 + int(qAbs(weight) * 2));
                }

                painter.setPen(connectionPen);
                painter.drawLine(x1, y1, x2, y2);  // Рисуем линию связи

                // Если вес значительный, подписываем его значение
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

    // ========== РИСУЕМ НЕЙРОНЫ ==========
    for (int layer = 0; layer < layerSizes.size(); layer++) {
        int x = startX + layer * layerSpacing;  // X-координата слоя
        int spacing = height / (layerSizes[layer] + 1);  // Вертикальный интервал

        for (int i = 0; i < layerSizes[layer]; i++) {
            int y = spacing * (i + 1);  // Y-координата нейрона
            int radius = 25;  // Радиус круга нейрона

            // Получаем выходное значение нейрона
            double outputValue = 0.0;
            if (layer < layerOutputs.size() && i < layerOutputs[layer].size()) {
                outputValue = layerOutputs[layer][i];
            }

            // Цвет нейрона зависит от значения выхода (чем больше значение, тем ярче)
            QColor neuronColor = layerColors[layer % layerColors.size()];
            int intensity = 80 + int(outputValue * 175);
            neuronColor = QColor(
                qMin(255, (neuronColor.red() * intensity) / 255),
                qMin(255, (neuronColor.green() * intensity) / 255),
                qMin(255, (neuronColor.blue() * intensity) / 255)
                );

            // Рисуем круг нейрона
            painter.setBrush(QBrush(neuronColor));
            painter.setPen(QPen(Qt::black, 2));
            painter.drawEllipse(x - radius/2, y - radius/2, radius, radius);

            // Рисуем текстовое значение выхода нейрона
            painter.setPen(Qt::white);
            QFont font = painter.font();
            font.setPointSize(8);
            font.setBold(true);
            painter.setFont(font);
            painter.drawText(x - 12, y + 4,
                             QString::number(outputValue, 'f', 3));

            // Рисуем название слоя (только для первого нейрона в слое)
            if (i == 0) {
                QString layerName;
                switch(layer) {
                case 0: layerName = "Input\n(2)"; break;      // Входной слой
                case 1: layerName = "Hidden 1\n(3)"; break;   // Скрытый слой 1
                case 2: layerName = "Hidden 2\n(2)"; break;   // Скрытый слой 2
                case 3: layerName = "Output\n(1)"; break;     // Выходной слой
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
