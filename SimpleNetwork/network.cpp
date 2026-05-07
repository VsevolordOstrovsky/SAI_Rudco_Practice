#include "network.h"
#include "ui_network.h"
#include "neuralnetworkwidget.h"
#include <QDoubleValidator>
#include <QRandomGenerator>
#include <QtMath>
#include <QInputDialog>
#include <QMessageBox>
#include <QVBoxLayout>

// ===================== РЕАЛИЗАЦИЯ НЕЙРОНА =====================

Neuron::Neuron() {
    weights.resize(0);
    prevWeightDelta.resize(0);
    bias = 0.0;
    prevBiasDelta = 0.0;
    output = 0.0;
    delta = 0.0;
}

Neuron::Neuron(int numInputs) {
    weights.resize(numInputs);
    prevWeightDelta.resize(numInputs);

    double limit = qSqrt(6.0 / numInputs);
    for (int i = 0; i < numInputs; i++) {
        weights[i] = QRandomGenerator::global()->generateDouble() * 2 * limit - limit;
        prevWeightDelta[i] = 0.0;
    }
    bias = QRandomGenerator::global()->generateDouble() * 2 * limit - limit;
    prevBiasDelta = 0.0;
    output = 0.0;
    delta = 0.0;
}

double Neuron::activationFunction(double x) {
    return 1.0 / (1.0 + qExp(-x));
}

double Neuron::activationFunctionDerivative(double x) {
    return x * (1.0 - x);
}

double Neuron::forward(const QVector<double>& inputs) {
    double sum = bias;
    for (int i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * weights[i];
    }
    output = activationFunction(sum);
    return output;
}

void Neuron::calculateOutputDelta(double target) {
    double error = target - output;
    delta = error * activationFunctionDerivative(output);
}

void Neuron::calculateHiddenDelta(const QVector<Neuron>& nextLayer, int neuronIndex) {
    double error = 0.0;
    for (int i = 0; i < nextLayer.size(); i++) {
        error += nextLayer[i].getWeight(neuronIndex) * nextLayer[i].getDelta();
    }
    delta = error * activationFunctionDerivative(output);
}

void Neuron::updateWeights(const QVector<double>& inputs, double learningRate, double momentum) {
    for (int i = 0; i < weights.size(); i++) {
        double gradient = learningRate * delta * inputs[i];
        double weightDelta = gradient + momentum * prevWeightDelta[i];
        weights[i] += weightDelta;
        prevWeightDelta[i] = weightDelta;
    }
    double biasGradient = learningRate * delta;
    double biasDelta = biasGradient + momentum * prevBiasDelta;
    bias += biasDelta;
    prevBiasDelta = biasDelta;
}

void Neuron::printWeights() const {
    qDebug() << "Bias:" << bias << "Weights:" << weights;
}

// ===================== РЕАЛИЗАЦИЯ НЕЙРОННОЙ СЕТИ =====================

Network::Network(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::Network)
    , currentStepIndex(-1)
    , currentLearningRate(0.1)
    , currentMomentum(0.9)
    , neuralWidget(nullptr)
    , autoNextTimer(nullptr)
    , autoNextStepTimer(nullptr)
    , autoBackTimer(nullptr)
    , autoBackStepTimer(nullptr)
    , isAutoScrolling(false)
{
    ui->setupUi(this);

    // Создаем виджет для отрисовки нейронной сети
    neuralWidget = new NeuralNetworkWidget(this);

    // Встраиваем виджет в контейнер на UI
    if (ui->graphicsWidget) {
        QLayout* layout = new QVBoxLayout(ui->graphicsWidget);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->addWidget(neuralWidget);
        ui->graphicsWidget->setLayout(layout);
    }

    init_Validator();
    init_NeuralNetwork();
    init_connections();

    ui->PTE_ShowLog->setReadOnly(true);
    ui->PTE_ShowLog->setLineWrapMode(QPlainTextEdit::NoWrap);
    ui->PTE_ShowLog->document()->setMaximumBlockCount(10000);

    // ========== НАСТРОЙКА ТАЙМЕРОВ ДЛЯ АВТОЛИСТАНИЯ ==========
    autoNextTimer = new QTimer(this);
    autoBackTimer = new QTimer(this);
    autoNextTimer->setInterval(500);
    autoBackTimer->setInterval(500);
    autoNextTimer->setSingleShot(true);
    autoBackTimer->setSingleShot(true);

    autoNextStepTimer = new QTimer(this);
    autoBackStepTimer = new QTimer(this);
    autoNextStepTimer->setInterval(100);
    autoBackStepTimer->setInterval(100);

    connect(autoNextTimer, &QTimer::timeout, this, &Network::startAutoNext);
    connect(autoBackTimer, &QTimer::timeout, this, &Network::startAutoBack);
    connect(autoNextStepTimer, &QTimer::timeout, this, &Network::autoNextStep);
    connect(autoBackStepTimer, &QTimer::timeout, this, &Network::autoBackStep);

    ui->BTN_Next->setContextMenuPolicy(Qt::CustomContextMenu);
    ui->BTN_Back->setContextMenuPolicy(Qt::CustomContextMenu);

    ui->DSB_SpeedLearn->setValue(0.3);
    ui->DSB_Moment->setValue(0.1);

    ui->LE_Result->setText("0.9");

    updateVisualization();
}

Network::~Network()
{
    delete ui;
}

void Network::init_NeuralNetwork() {
    layerSizes = {2, 3, 2, 1};
    layers.resize(layerSizes.size());
    layerOutputs.resize(layerSizes.size());

    for (int i = 0; i < layerSizes.size(); i++) {
        layers[i].resize(layerSizes[i]);
        layerOutputs[i].resize(layerSizes[i]);

        if (i == 0) {
            for (int j = 0; j < layerSizes[i]; j++) {
                layers[i][j].setOutput(0.0);
            }
        } else {
            for (int j = 0; j < layerSizes[i]; j++) {
                layers[i][j] = Neuron(layerSizes[i-1]);
            }
        }
    }

    addLog("Нейросеть инициализирована", "cyan");
    addLog(QString("Архитектура: %1 -> %2 -> %3 -> %4")
               .arg(layerSizes[0]).arg(layerSizes[1]).arg(layerSizes[2]).arg(layerSizes[3]), "yellow");
}

void Network::init_connections()
{
    connect(ui->pushButton, &QPushButton::clicked, this, &Network::button_action);

    connect(ui->BTN_Next, &QPushButton::pressed, this, &Network::on_BTN_Next_pressed);
    connect(ui->BTN_Next, &QPushButton::released, this, &Network::on_BTN_Next_released);
    connect(ui->BTN_Back, &QPushButton::pressed, this, &Network::on_BTN_Back_pressed);
    connect(ui->BTN_Back, &QPushButton::released, this, &Network::on_BTN_Back_released);

    connect(ui->BTN_Next, &QPushButton::customContextMenuRequested, this, &Network::on_BTN_Next_customContextMenuRequested);
    connect(ui->BTN_Back, &QPushButton::customContextMenuRequested, this, &Network::on_BTN_Back_customContextMenuRequested);
}

void Network::init_Validator()
{
    QDoubleValidator *validator_FirstInput = new QDoubleValidator(-1000.0, 1000.0, 3, ui->LE_FirstInput);
    QDoubleValidator *validator_SecondInput = new QDoubleValidator(-1000.0, 1000.0, 3, ui->LE_SecondInput);
    QDoubleValidator *validator_Result = new QDoubleValidator(0.0, 1.0, 3, ui->LE_Result);

    validator_FirstInput->setNotation(QDoubleValidator::StandardNotation);
    validator_SecondInput->setNotation(QDoubleValidator::StandardNotation);
    validator_Result->setNotation(QDoubleValidator::StandardNotation);

    ui->LE_FirstInput->setValidator(validator_FirstInput);
    ui->LE_SecondInput->setValidator(validator_SecondInput);
    ui->LE_Result->setValidator(validator_Result);
}

void Network::addLog(const QString& message, const QString& color) {
    ui->PTE_ShowLog->appendHtml(QString("<span style='color: %1;'>%2</span>").arg(color).arg(message));
    ui->PTE_ShowLog->ensureCursorVisible();
}

double Network::forward(const QVector<double>& inputs) {
    // Устанавливаем входные значения
    for (int i = 0; i < layerSizes[0]; i++) {
        layerOutputs[0][i] = inputs[i];
        layers[0][i].setOutput(inputs[i]);
    }

    // Прямое распространение по слоям
    for (int layer = 1; layer < layers.size(); layer++) {
        for (int neuron = 0; neuron < layers[layer].size(); neuron++) {
            layerOutputs[layer][neuron] = layers[layer][neuron].forward(layerOutputs[layer-1]);
        }
    }

    // Обновляем визуализацию
    updateVisualization();

    return layerOutputs.last().first();
}

void Network::backward(double target) {
    int lastLayer = layers.size() - 1;

    for (int i = 0; i < layers[lastLayer].size(); i++) {
        layers[lastLayer][i].calculateOutputDelta(target);
    }

    for (int layer = lastLayer - 1; layer > 0; layer--) {
        for (int i = 0; i < layers[layer].size(); i++) {
            layers[layer][i].calculateHiddenDelta(layers[layer+1], i);
        }
    }
}

void Network::updateWeights(double learningRate, double momentum) {
    for (int layer = 1; layer < layers.size(); layer++) {
        for (int neuron = 0; neuron < layers[layer].size(); neuron++) {
            layers[layer][neuron].updateWeights(layerOutputs[layer-1], learningRate, momentum);
        }
    }
}

void Network::restoreWeights(const QVector<QVector<QVector<double>>>& weightsHistory,
                             const QVector<QVector<double>>& biasHistory) {
    for (int layer = 1; layer < layers.size(); layer++) {
        for (int neuron = 0; neuron < layers[layer].size(); neuron++) {
            for (int weight = 0; weight < layerSizes[layer-1]; weight++) {
                layers[layer][neuron].setWeight(weight, weightsHistory[layer][neuron][weight]);
            }
            if (layer < biasHistory.size() && neuron < biasHistory[layer].size()) {
                layers[layer][neuron].setBias(biasHistory[layer][neuron]);
            }
        }
    }
}

void Network::trainStep(double input1, double input2, double target) {
    QVector<double> inputs = {input1, input2};

    // Сохраняем историю весов ДО обновления
    QVector<QVector<QVector<double>>> weightsHistory;
    QVector<QVector<double>> biasHistory;
    weightsHistory.resize(layers.size());
    biasHistory.resize(layers.size());
    for (int layer = 1; layer < layers.size(); layer++) {
        weightsHistory[layer].resize(layers[layer].size());
        biasHistory[layer].resize(layers[layer].size());
        for (int neuron = 0; neuron < layers[layer].size(); neuron++) {
            weightsHistory[layer][neuron].resize(layerSizes[layer-1]);
            for (int weight = 0; weight < layerSizes[layer-1]; weight++) {
                weightsHistory[layer][neuron][weight] = layers[layer][neuron].getWeight(weight);
            }
            biasHistory[layer][neuron] = layers[layer][neuron].getPrevBiasDelta();
        }
    }

    // Прямой проход
    double output = forward(inputs);
    double error = target - output;

    // Снимок выходов всех нейронов сразу после forward
    QVector<QVector<double>> layerOutputsSnapshot = layerOutputs;

    // Обратный проход и обновление весов
    backward(target);
    updateWeights(currentLearningRate, currentMomentum);

    // Сохраняем шаг
    TrainingStep step;
    step.inputs = inputs;
    step.target = target;
    step.weightsHistory = weightsHistory;
    step.biasHistory = biasHistory;
    step.layerOutputsSnapshot = layerOutputsSnapshot;
    step.output = output;
    step.error = qAbs(error);

    trainingSteps.append(step);
    currentStepIndex = trainingSteps.size() - 1;
}

// ОСНОВНОЙ МЕТОД ДЛЯ ПЕРЕКЛЮЧЕНИЯ ШАГОВ
void Network::goToStep(int step) {
    if (trainingSteps.isEmpty()) return;
    if (step < 0 || step >= trainingSteps.size()) return;
    if (step == currentStepIndex) return;

    currentStepIndex = step;
    const TrainingStep& stepData = trainingSteps[step];

    // Жёстко берём сохранённые выходы нейронов — без пересчёта forward()
    layerOutputs = stepData.layerOutputsSnapshot;

    // Строим структуру весов для виджета из снимка шага
    QVector<QVector<QVector<double>>> widgetWeights;
    widgetWeights.resize(layers.size());
    for (int layer = 0; layer < layers.size(); layer++) {
        widgetWeights[layer].resize(layerSizes[layer]);
        for (int neuron = 0; neuron < layerSizes[layer]; neuron++) {
            if (layer == 0) {
                widgetWeights[layer][neuron].resize(0);
            } else {
                widgetWeights[layer][neuron].resize(layerSizes[layer-1]);
                for (int w = 0; w < layerSizes[layer-1]; w++) {
                    widgetWeights[layer][neuron][w] = stepData.weightsHistory[layer][neuron][w];
                }
            }
        }
    }

    // Обновляем виджет напрямую из снимка
    if (neuralWidget) {
        neuralWidget->setLayerSizes(layerSizes);
        neuralWidget->setLayerOutputs(stepData.layerOutputsSnapshot);
        neuralWidget->setWeights(widgetWeights);
        neuralWidget->update();
    }

    // Обновляем текстовую информацию
    updateDisplay();
}

void Network::setCurrentStep(int step) {
    goToStep(step);
}

void Network::button_action()
{
    double val1 = 0.0, val2 = 0.0, target = 0.0;

    if(ui->LE_FirstInput->text().isEmpty()) {
        val1 = 0.0;
    } else {
        val1 = ui->LE_FirstInput->text().replace(",",".").toDouble();
    }

    if(ui->LE_SecondInput->text().isEmpty()) {
        val2 = 0.0;
    } else {
        val2 = ui->LE_SecondInput->text().replace(",",".").toDouble();
    }

    if(ui->LE_Result->text().isEmpty()) {
        target = 0.5;
    } else {
        target = ui->LE_Result->text().replace(",",".").toDouble();
        if (target < 0.0) target = 0.0;
        if (target > 1.0) target = 1.0;
    }

    currentLearningRate = ui->DSB_SpeedLearn->value();
    currentMomentum = ui->DSB_Moment->value();

    addLog("========================================", "gray");
    addLog("НАЧАЛО ОБУЧЕНИЯ", "yellow");
    addLog(QString("Скорость обучения: %1, Момент: %2").arg(currentLearningRate).arg(currentMomentum), "cyan");
    addLog(QString("Входные значения: [%1, %2]").arg(val1).arg(val2), "cyan");
    addLog(QString("Целевое значение (из LE_Result): %1").arg(target), "magenta");

    trainingSteps.clear();

    for (int epoch = 0; epoch < 1000; epoch++) {
        trainStep(val1, val2, target);

        if (epoch % 100 == 0 && epoch > 0) {
            addLog(QString("Шаг %1: Выход = %2, Ошибка = %3, Цель = %4")
                       .arg(epoch)
                       .arg(trainingSteps.last().output, 0, 'f', 6)
                       .arg(trainingSteps.last().error, 0, 'f', 6)
                       .arg(target, 0, 'f', 4), "lightgray");
        }
    }

    const TrainingStep& finalStep = trainingSteps.last();
    addLog("========================================", "gray");
    addLog("ОБУЧЕНИЕ ЗАВЕРШЕНО", "green");
    addLog(QString("Целевое значение: %1").arg(target, 0, 'f', 4), "magenta");
    addLog(QString("Финальный выход сети: %1").arg(finalStep.output, 0, 'f', 6), "lime");
    addLog(QString("Финальная ошибка: %1").arg(finalStep.error, 0, 'f', 6), "lime");

    double accuracy = (1.0 - finalStep.error) * 100;
    if (finalStep.error < 0.1) {
        addLog(QString("✅ Отличный результат! Точность: %1%").arg(accuracy, 0, 'f', 2), "green");
    } else if (finalStep.error < 0.2) {
        addLog(QString("👍 Хороший результат. Точность: %1%").arg(accuracy, 0, 'f', 2), "yellow");
    } else {
        addLog(QString("⚠️ Результат можно улучшить. Точность: %1%").arg(accuracy, 0, 'f', 2), "red");
    }
    addLog("========================================", "gray");

    goToStep(trainingSteps.size() - 1);
}

void Network::on_BTN_Next_clicked() {
    if (trainingSteps.isEmpty()) {
        addLog("Сначала выполните расчет!", "red");
        return;
    }

    if (currentStepIndex < trainingSteps.size() - 1) {
        goToStep(currentStepIndex + 1);
    } else {
        addLog("Это последний шаг обучения", "yellow");
    }
}

void Network::on_BTN_Back_clicked() {
    if (trainingSteps.isEmpty()) {
        addLog("Сначала выполните расчет!", "red");
        return;
    }

    if (currentStepIndex > 0) {
        goToStep(currentStepIndex - 1);
    } else {
        addLog("Это первый шаг обучения", "yellow");
    }
}

void Network::on_BTN_Next_pressed() {
    if (trainingSteps.isEmpty()) return;
    if (!autoNextTimer->isActive() && !autoNextStepTimer->isActive()) {
        autoNextTimer->start();
    }
}

void Network::on_BTN_Next_released() {
    bool wasAutoScrolling = isAutoScrolling;
    autoNextTimer->stop();
    autoNextStepTimer->stop();
    isAutoScrolling = false;

    if (!wasAutoScrolling) {
        on_BTN_Next_clicked();
    }
}

void Network::on_BTN_Back_pressed() {
    if (trainingSteps.isEmpty()) return;
    if (!autoBackTimer->isActive() && !autoBackStepTimer->isActive()) {
        autoBackTimer->start();
    }
}

void Network::on_BTN_Back_released() {
    bool wasAutoScrolling = isAutoScrolling;
    autoBackTimer->stop();
    autoBackStepTimer->stop();
    isAutoScrolling = false;

    if (!wasAutoScrolling) {
        on_BTN_Back_clicked();
    }
}

void Network::startAutoNext() {
    if (!autoNextStepTimer->isActive()) {
        isAutoScrolling = true;
        autoNextStepTimer->start();
    }
}

void Network::startAutoBack() {
    if (!autoBackStepTimer->isActive()) {
        isAutoScrolling = true;
        autoBackStepTimer->start();
    }
}

void Network::autoNextStep() {
    if (currentStepIndex < trainingSteps.size() - 1) {
        goToStep(currentStepIndex + 1);
    } else {
        autoNextStepTimer->stop();
        isAutoScrolling = false;
    }
}

void Network::autoBackStep() {
    if (currentStepIndex > 0) {
        goToStep(currentStepIndex - 1);
    } else {
        autoBackStepTimer->stop();
        isAutoScrolling = false;
    }
}

void Network::showStepDialog() {
    if (trainingSteps.isEmpty()) {
        addLog("Нет шагов для навигации. Сначала выполните расчет!", "red");
        return;
    }

    bool ok;
    int stepNumber = QInputDialog::getInt(this,
                                          "Выбор шага обучения",
                                          QString("Выберите шаг от 1 до %1:\n(Текущий шаг: %2)")
                                              .arg(trainingSteps.size())
                                              .arg(currentStepIndex + 1),
                                          currentStepIndex + 1,
                                          1,
                                          trainingSteps.size(),
                                          1,
                                          &ok);

    if (ok) {
        goToStep(stepNumber - 1);
        addLog(QString("Переход на шаг %1 из %2").arg(stepNumber).arg(trainingSteps.size()), "cyan");
    }
}

void Network::on_BTN_Next_customContextMenuRequested(const QPoint &pos) {
    Q_UNUSED(pos);
    showStepDialog();
}

void Network::on_BTN_Back_customContextMenuRequested(const QPoint &pos) {
    Q_UNUSED(pos);
    showStepDialog();
}

void Network::updateDisplay() {
    if (trainingSteps.isEmpty() || currentStepIndex < 0) return;

    const TrainingStep& step = trainingSteps[currentStepIndex];

    QString info = QString( "<html><body>"
                           "<p style='font-size:11pt;'>"
                           "<span style='color: #ffaa00;'>ШАГ %1 ИЗ %2</span><br>"
                           "<span style='color: #888888;'>─────────────────────</span><br>"
                           "<span style='color: #00ff00;'>Входные значения:</span> [%3, %4]<br>"
                           "<span style='color: #00ff00;'>Целевое значение:</span> %5<br>"
                           "<span style='color: #ffff00;'>Выход сети:</span> %6<br>"
                           "<span style='color: #ff6666;'>Ошибка:</span> %7<br>"
                           "<span style='color: #888888;'>─────────────────────</span><br>"
                           "<span style='color: #888888; font-size:9pt;'>  Управление:</span><br>"
                           "<span style='color: #cccccc; font-size:9pt;'>  • ЛКМ - листание шагов</span><br>"
                           "<span style='color: #cccccc; font-size:9pt;'>  • Зажатие ЛКМ (500мс задержка, затем 100мс интервал) - автолистание</span><br>"
                           "<span style='color: #cccccc; font-size:9pt;'>  • ПКМ - выбор конкретного шага</span><br>"
                           "</p></body></html>")
                       .arg(currentStepIndex + 1)
                       .arg(trainingSteps.size())
                       .arg(step.inputs[0], 0, 'f', 4)
                       .arg(step.inputs[1], 0, 'f', 4)
                       .arg(step.target, 0, 'f', 4)
                       .arg(step.output, 0, 'f', 8)
                       .arg(step.error, 0, 'f', 8);

    ui->L_Resalt->setText(info);
}

void Network::updateVisualization() {
    if (neuralWidget) {
        // Передаем размеры слоев
        neuralWidget->setLayerSizes(layerSizes);

        // Передаем актуальные выходные значения нейронов
        neuralWidget->setLayerOutputs(layerOutputs);

        // Собираем веса для отрисовки связей
        QVector<QVector<QVector<double>>> allWeights;
        allWeights.resize(layers.size());

        for (int layer = 0; layer < layers.size(); layer++) {
            allWeights[layer].resize(layers[layer].size());
            for (int neuron = 0; neuron < layers[layer].size(); neuron++) {
                if (layer == 0) {
                    allWeights[layer][neuron].resize(0);
                } else {
                    allWeights[layer][neuron].resize(layerSizes[layer-1]);
                    for (int weight = 0; weight < layerSizes[layer-1]; weight++) {
                        allWeights[layer][neuron][weight] = layers[layer][neuron].getWeight(weight);
                    }
                }
            }
        }

        neuralWidget->setWeights(allWeights);

        // Принудительная перерисовка
        neuralWidget->update();
    }
}
