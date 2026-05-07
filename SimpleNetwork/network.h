#ifndef NETWORK_H
#define NETWORK_H

#include <QMainWindow>
#include <QVector>
#include <QDebug>
#include <QTimer>

QT_BEGIN_NAMESPACE
namespace Ui {
class Network;
}
QT_END_NAMESPACE

// Forward declaration
class NeuralNetworkWidget;

// ===================== КЛАСС НЕЙРОН =====================
class Neuron {
public:
    Neuron();
    Neuron(int numInputs);

    double forward(const QVector<double>& inputs);
    void calculateOutputDelta(double target);
    void calculateHiddenDelta(const QVector<Neuron>& nextLayer, int neuronIndex);
    void updateWeights(const QVector<double>& inputs, double learningRate, double momentum);

    double getOutput() const { return output; }
    double getDelta() const { return delta; }
    double getWeight(int index) const { return weights[index]; }
    void setOutput(double value) { output = value; }
    void setDelta(double value) { delta = value; }
    void setWeight(int index, double value) { weights[index] = value; }
    void setPrevWeightDelta(int index, double value) { prevWeightDelta[index] = value; }
    double getPrevWeightDelta(int index) const { return prevWeightDelta[index]; }
    double getPrevBiasDelta() const { return prevBiasDelta; }
    void setPrevBiasDelta(double value) { prevBiasDelta = value; }
    void setWeights(const QVector<double>& newWeights) { weights = newWeights; }
    void setBias(double newBias) { bias = newBias; }
    void printWeights() const;

private:
    double activationFunction(double x);
    double activationFunctionDerivative(double x);

    QVector<double> weights;
    QVector<double> prevWeightDelta;
    double bias;
    double prevBiasDelta;
    double output;
    double delta;
};

// ===================== КЛАСС НЕЙРОННАЯ СЕТЬ =====================
class Network : public QMainWindow
{
    Q_OBJECT

public:
    Network(QWidget *parent = nullptr);
    ~Network();

    double forward(const QVector<double>& inputs);
    void backward(double target);
    void updateWeights(double learningRate, double momentum);
    void trainStep(double input1, double input2, double target);
    void setCurrentStep(int step);
    int getTotalSteps() const { return trainingSteps.size(); }
    void restoreWeights(const QVector<QVector<QVector<double>>>& weightsHistory,
                        const QVector<QVector<double>>& biasHistory);
    void goToStep(int step);  // Новый метод для перехода на шаг

private slots:
    void button_action();
    void on_BTN_Next_clicked();
    void on_BTN_Back_clicked();
    void on_BTN_Next_pressed();
    void on_BTN_Next_released();
    void on_BTN_Back_pressed();
    void on_BTN_Back_released();
    void startAutoNext();  // Новый слот
    void startAutoBack();  // Новый слот
    void autoNextStep();
    void autoBackStep();
    void showStepDialog();
    void on_BTN_Next_customContextMenuRequested(const QPoint &pos);
    void on_BTN_Back_customContextMenuRequested(const QPoint &pos);

private:
    Ui::Network *ui;

    void init_Validator();
    void init_connections();
    void init_NeuralNetwork();
    void addLog(const QString& message, const QString& color = "white");
    void updateDisplay();
    void updateVisualization();

    QVector<QVector<Neuron>> layers;
    QVector<int> layerSizes;
    QVector<QVector<double>> layerOutputs;

    NeuralNetworkWidget* neuralWidget;

    struct TrainingStep {
        QVector<double> inputs;
        double target;
        QVector<QVector<QVector<double>>> weightsHistory;
        QVector<QVector<double>> biasHistory;
        QVector<QVector<double>> layerOutputsSnapshot;  // снимок выходов всех нейронов
        double output;
        double error;
    };
    QVector<TrainingStep> trainingSteps;
    int currentStepIndex;
    double currentLearningRate;
    double currentMomentum;

    QTimer* autoNextTimer;
    QTimer* autoNextStepTimer;
    QTimer* autoBackTimer;
    QTimer* autoBackStepTimer;
    bool isAutoScrolling;  // Флаг для отслеживания автолистания
};

#endif // NETWORK_H
