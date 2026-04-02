#include "lin_reg.h"
#include "ui_lin_reg.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>

lin_reg::lin_reg(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::lin_reg), _graphRegres(new QCustomPlot)
{
    ui->setupUi(this);

    // Создаем кнопку для загрузки Excel файла
    QPushButton *loadButton = new QPushButton("Загрузить Excel файл", this);
    loadButton->setFixedSize(150, 30);

    // Выбор формата регрессии
    QComboBox *ChangeBox = new QComboBox();
    ChangeBox->setFixedSize(200,30);
    QStringList list {"Линейная регрессия", "Полиномиальная регрессия"};
    ChangeBox->addItems(list);


    // Создаем тулбар и добавляем на него кнопку
    QToolBar *toolBar = addToolBar("Инструменты");
    toolBar->addWidget(loadButton);
    toolBar->addWidget(ChangeBox);

    // Подключаем сигнал нажатия кнопки
    connect(loadButton, &QPushButton::clicked, this, &lin_reg::onLoadExcelFile);
    connect(ChangeBox, &QComboBox::currentIndexChanged, this, &lin_reg::ChangeOption);

    QWidget *central = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(central);

    layout->addWidget(_graphRegres);

    setCentralWidget(central);
    createGraph();
}

lin_reg::~lin_reg()
{
    delete _graphRegres;
    delete ui;
}

void lin_reg::createGraph()
{
    // Очищаем векторы на случай повторного вызова
    X.clear();
    Y.clear();
    X_red.clear();
    Y_red.clear();

    // Очищаем график
    _graphRegres->clearGraphs();

    // Создаем синусоиду
    double Value = 0;
    for(int x = 0; x <= 5000; x+=1)
    {
        Value = x/100.0;
        X.push_back(Value);
        Y.push_back(sin(Value));
    }

    // // Добавляем график синусоиды
    // _graphRegres->addGraph();
    // _graphRegres->graph(0)->setData(X,Y);
    // _graphRegres->graph(0)->setPen(QPen(Qt::blue));
    // _graphRegres->rescaleAxes(true);

    // Получаем текущие границы осей после rescaleAxes
    double xMin = _graphRegres->xAxis->range().lower;
    double xMax = _graphRegres->xAxis->range().upper;
    double yMin = _graphRegres->yAxis->range().lower;
    double yMax = _graphRegres->yAxis->range().upper;

    // Генерируем случайные точки по всей области графика
    for(int i = 0; i < 50; i++)
    {
        // Генерация X в пределах всей оси X
        double x = QRandomGenerator64::global()->generateDouble() * (xMax - xMin) + xMin;
        X_red.push_back(x);

        // Генерация Y в пределах всей оси Y
        double y = QRandomGenerator64::global()->generateDouble() * (yMax - yMin) + yMin;
        Y_red.push_back(y);
    }

    // Добавляем график случайных точек
    // _graphRegres->addGraph();
    // _graphRegres->graph(1)->setData(X_red, Y_red);
    // _graphRegres->graph(1)->setLineStyle(QCPGraph::lsNone);
    // _graphRegres->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 5));
    // _graphRegres->graph(1)->setPen(QPen(Qt::red));
    // _graphRegres->graph(1)->setBrush(QBrush(Qt::red));

    // Обновляем график
    _graphRegres->replot();
}

void lin_reg::onLoadExcelFile()
{
    // Открываем диалог выбора файла
    QString filePath = QFileDialog::getOpenFileName(
        this,
        "Выберите CSV файл",
        QDir::homePath(),
        "CSV файлы (*.csv);;Все файлы (*.*)"
        );

    if (filePath.isEmpty()) {
        return;
    }

    // Парсим Excel файл
    parseExcelFile(filePath);
}

void lin_reg::CalAndShowLinRegg(QVector<double>& X, QVector<double>& Y)
{
    if (X.size() != Y.size() || X.isEmpty()) {
        qDebug() << "Ошибка: векторы пусты или имеют разный размер";
        return;
    }

    double sr_x = 0, sr_y = 0, a_0 = 0, a_1 = 0;
    double sr_xy = 0, sr_x2 = 0;

    const int size = X.size();

    for(int i = 0; i < size; i++)
    {
        sr_x += X.at(i);
        sr_y += Y.at(i);
        sr_xy += X.at(i) * Y.at(i);
        sr_x2 += pow(X.at(i), 2);
    }

    sr_x /= size;
    sr_y /= size;
    sr_xy /= size;
    sr_x2 /= size;

    double denominator = sr_x2 - pow(sr_x, 2);
    if (fabs(denominator) > 1e-10) {
        a_0 = (sr_xy - sr_x * sr_y) / denominator;
    } else {
        a_0 = 0.0;
        qDebug() << "Предупреждение: знаменатель близок к нулю";
    }

    a_1 = sr_y - a_0 * sr_x;

    qDebug() << "Средние значения: x =" << sr_x << ", y =" << sr_y;
    qDebug() << "Коэффициенты: a_0 =" << a_0 << ", a_1 =" << a_1;

    QVector<double> GrX, GrY;

    double minX = *std::min_element(X.begin(), X.end());
    double maxX = *std::max_element(X.begin(), X.end());

    GrX << minX << maxX;
    GrY << a_0 * minX + a_1 << a_0 * maxX + a_1;

    _graphRegres->addGraph();
    int graphIndex = _graphRegres->graphCount() - 1;
    _graphRegres->graph(graphIndex)->setData(GrX, GrY);
    _graphRegres->graph(graphIndex)->setPen(QPen(Qt::red, 2));
    _graphRegres->rescaleAxes(true);
    _graphRegres->replot();

    qDebug() << "Линия регрессии построена от" << minX << "до" << maxX;
}

void lin_reg::parseExcelFile(const QString &filePath)
{
    QVector<double> excelX, excelY;

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "Не удалось открыть файл";
        return;
    }

    QTextStream in(&file);
    int row = 0;

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        row++;

        if (line.isEmpty())
            continue;

        QStringList parts;
        if (line.contains(";"))
            parts = line.split(";");
        else
            parts = line.split(",");

        if (parts.size() < 2) {
            qDebug() << "Мало данных в строке" << row;
            continue;
        }

        QString xStr = parts[0].trimmed();
        QString yStr = parts[1].trimmed();

        xStr.replace(",", ".");
        yStr.replace(",", ".");

        bool okX, okY;
        double x = xStr.toDouble(&okX);
        double y = yStr.toDouble(&okY);

        if (okX && okY) {
            excelX.append(x);
            excelY.append(y);
        } else {
            qDebug() << "Ошибка в строке" << row << ":" << line;
        }
    }

    file.close();



    // Рисуем точки
    _graphRegres->clearPlottables();
    _graphRegres->addGraph();
    int graphIndex = _graphRegres->graphCount() - 1;

    _graphRegres->graph(graphIndex)->setData(excelX, excelY);
    _graphRegres->graph(graphIndex)->setLineStyle(QCPGraph::lsNone);
    _graphRegres->graph(graphIndex)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 6));
    _graphRegres->graph(graphIndex)->setPen(QPen(Qt::green));

    _graphRegres->rescaleAxes(true);
    _graphRegres->replot();


    switch (ChangeOptionFlag) {
    case 0:
        CalAndShowLinRegg(excelX, excelY);
        break;
    case 1:
        CalAndShowKvRegg(excelX, excelY);
        break;
    default:
        CalAndShowLinRegg(excelX, excelY);
        break;
    }

}

void lin_reg::CalAndShowKvRegg(QVector<double>& X, QVector<double>& Y)
{
    if (X.size() != Y.size() || X.size() < 3) {
        qDebug() << "Недостаточно данных для квадратичной регрессии";
        return;
    }

    double sum_x = 0.0, sum_y = 0.0;
    double sum_xy = 0.0, sum_x2y = 0.0;
    double sum_x2 = 0.0, sum_x3 = 0.0, sum_x4 = 0.0;

    int n = X.size();

    // Вычисление сумм
    for(int i = 0; i < n; i++)
    {
        double xi = X.at(i);
        double yi = Y.at(i);
        double xi2 = xi * xi;
        double xi3 = xi2 * xi;
        double xi4 = xi3 * xi;

        sum_x += xi;
        sum_y += yi;
        sum_x2 += xi2;
        sum_x3 += xi3;
        sum_x4 += xi4;
        sum_xy += xi * yi;
        sum_x2y += xi2 * yi;
    }

    double det = n * (sum_x2 * sum_x4 - sum_x3 * sum_x3) -
                 sum_x * (sum_x * sum_x4 - sum_x3 * sum_x2) +
                 sum_x2 * (sum_x * sum_x3 - sum_x2 * sum_x2);

    // Определители для каждого коэффициента
    double det_a2 = sum_y * (sum_x2 * sum_x4 - sum_x3 * sum_x3) -
                    sum_x * (sum_xy * sum_x4 - sum_x3 * sum_x2y) +
                    sum_x2 * (sum_xy * sum_x3 - sum_x2 * sum_x2y);

    double det_a1 = n * (sum_xy * sum_x4 - sum_x3 * sum_x2y) -
                    sum_y * (sum_x * sum_x4 - sum_x3 * sum_x2) +
                    sum_x2 * (sum_x * sum_x2y - sum_xy * sum_x2);

    double det_a0 = n * (sum_x2 * sum_x2y - sum_xy * sum_x3) -
                    sum_x * (sum_x * sum_x2y - sum_xy * sum_x2) +
                    sum_y * (sum_x * sum_x3 - sum_x2 * sum_x2);

    double a0 = det_a0 / det;
    double a1 = det_a1 / det;
    double a2 = det_a2 / det;



    // Построение графика
    QVector<double> GrX, GrY;


    // Если данных мало, используем исходные точки
    for(int i = 0; i < n; i++)
    {
        GrX << X.at(i);
        GrY << a0 * X.at(i) * X.at(i) + a1 * X.at(i) + a2;
    }


    // Добавление графика
    if (_graphRegres) {
        _graphRegres->addGraph();
        int graphIndex = _graphRegres->graphCount() - 1;
        _graphRegres->graph(graphIndex)->setData(GrX, GrY);
        _graphRegres->graph(graphIndex)->setPen(QPen(Qt::blue, 2));
        _graphRegres->graph(graphIndex)->setName("Квадратичная регрессия");

        // Настройка осей
        _graphRegres->xAxis->setLabel("X");
        _graphRegres->yAxis->setLabel("Y");
        _graphRegres->rescaleAxes(true);
        _graphRegres->replot();
    } else {
        qDebug() << "Ошибка: _graphRegres не инициализирован";
    }
}


void lin_reg::ChangeOption(int index)
{
    ChangeOptionFlag = index;
}
