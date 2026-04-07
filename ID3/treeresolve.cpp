#include "treeresolve.h"
#include "ui_treeresolve.h"

#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QVBoxLayout>
#include <math.h>
#include <QGraphicsTextItem>

// ================= UI =================

TreeResolve::TreeResolve(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::TreeResolve)
{
    ui->setupUi(this);
    connect(ui->pushButton, &QPushButton::clicked, this, &TreeResolve::OnLoadExcelFile);
}

TreeResolve::~TreeResolve()
{
    delete ui;
}

// ================= CSV =================

void TreeResolve::OnLoadExcelFile()
{
    QString filePath = QFileDialog::getOpenFileName(this, "CSV", "", "*.csv");
    if (filePath.isEmpty()) return;

    ParseCSVFile(filePath);
    BuildTree();
}

void TreeResolve::ParseCSVFile(const QString &filePath)
{
    QFile file(filePath);
    file.open(QIODevice::ReadOnly);

    QTextStream in(&file);
    GlobalTable.clear();

    while (!in.atEnd())
        GlobalTable.append(in.readLine().split(","));

    file.close();
}

// ================= МАТЕМАТИКА =================

double TreeResolve::CalculateEntropy(const QVector<QStringList>& data)
{
    double yes = 0, no = 0;

    for (auto row : data)
        (row.last() == "Yes") ? yes++ : no++;

    double total = yes + no;
    if (total == 0) return 0;

    double p1 = yes / total;
    double p2 = no / total;

    if (p1 == 0 || p2 == 0) return 0;

    return E(p1, p2);
}

double TreeResolve::CalculateInformationGain(const QVector<QStringList>& data, int attrIndex, double parentEntropy)
{
    QMap<QString, QVector<QStringList>> groups;

    for (auto row : data)
        groups[row[attrIndex]].append(row);

    double IG = parentEntropy;

    for (auto g : groups)
    {
        double ent = CalculateEntropy(g);
        IG -= ((double)g.size() / data.size()) * ent;
    }

    return IG;
}

// ================= ДЕРЕВО =================

TreeNode* TreeResolve::BuildDecisionTree(
    const QVector<QStringList>& data,
    const QStringList& attributes,
    QSet<int> usedAttributes)
{
    TreeNode* node = new TreeNode();

    // Проверка на одинаковый класс
    QString first = data[0].last();
    bool same = true;

    for (auto r : data)
        if (r.last() != first) { same = false; break; }

    if (same) {
        node->isLeaf = true;
        node->decision = first;
        return node;
    }

    // Энтропия
    double parentEntropy = CalculateEntropy(data);

    int bestAttr = -1;
    double bestIG = -1;

    for (int i = 1; i < attributes.size() - 1; i++)
    {
        if (usedAttributes.contains(i)) continue;

        double ig = CalculateInformationGain(data, i, parentEntropy);

        if (ig > bestIG) {
            bestIG = ig;
            bestAttr = i;
        }
    }

    if (bestAttr == -1)
    {
        node->isLeaf = true;
        node->decision = GetMajorityClass(data);
        return node;
    }

    node->attribute = attributes[bestAttr];

    QSet<QString> values;
    for (auto row : data)
        values.insert(row[bestAttr]);

    usedAttributes.insert(bestAttr);

    for (auto val : values)
    {
        auto subset = FilterData(data, bestAttr, val);

        if (subset.isEmpty())
        {
            TreeNode* leaf = new TreeNode();
            leaf->isLeaf = true;
            leaf->decision = GetMajorityClass(data);
            node->children[val] = leaf;
        }
        else
        {
            node->children[val] =
                BuildDecisionTree(subset, attributes, usedAttributes);
        }
    }

    return node;
}

// ================= HELPERS =================

QVector<QStringList> TreeResolve::FilterData(
    const QVector<QStringList>& data,
    int attrIndex,
    const QString& value)
{
    QVector<QStringList> out;

    for (auto r : data)
        if (r[attrIndex] == value)
            out.append(r);

    return out;
}

QString TreeResolve::GetMajorityClass(const QVector<QStringList>& data)
{
    int y = 0, n = 0;

    for (auto r : data)
        (r.last() == "Yes") ? y++ : n++;

    return (y >= n) ? "Yes" : "No";
}

// ================= ВИЗУАЛ =================

TreeVisualization::TreeVisualization(TreeNode* root, QWidget *parent)
    : QDialog(parent), root(root)
{
    resize(1000, 700);

    scene = new QGraphicsScene(this);
    view = new QGraphicsView(scene, this);

    QVBoxLayout* l = new QVBoxLayout(this);
    l->addWidget(view);

    DrawTree();
}

int TreeVisualization::CalculateWidth(TreeNode* node)
{
    if (!node || node->isLeaf) return 1;

    int sum = 0;
    for (auto c : node->children)
        sum += CalculateWidth(c);

    return sum;
}

void TreeVisualization::DrawTree()
{
    int w = CalculateWidth(root) * 120;
    DrawNode(root, 0, 0, w);
}

void TreeVisualization::DrawNode(TreeNode* node, qreal x, qreal y, qreal width)
{
    if (!node) return;

    QBrush brush = node->isLeaf ?
                       (node->decision == "Yes" ? Qt::green : Qt::red)
                                : Qt::lightGray;

    scene->addEllipse(x-30, y-30, 60, 60, QPen(Qt::black), brush);

    auto t = scene->addText(node->isLeaf ? node->decision : node->attribute);
    t->setPos(x - t->boundingRect().width()/2,
              y - t->boundingRect().height()/2);

    if (node->isLeaf) return;

    qreal curX = x - width/2;

    for (auto it = node->children.begin(); it != node->children.end(); ++it)
    {
        int w = CalculateWidth(it.value()) * 120;
        qreal childX = curX + w/2;
        qreal childY = y + 120;

        DrawEdge(x, y+30, childX, childY-30, it.key());
        DrawNode(it.value(), childX, childY, w);

        curX += w;
    }
}

void TreeVisualization::DrawEdge(qreal x1, qreal y1, qreal x2, qreal y2, const QString& label)
{
    scene->addLine(x1, y1, x2, y2);

    auto t = scene->addText(label);
    t->setPos((x1+x2)/2, (y1+y2)/2);
}

// ================= START =================

void TreeResolve::BuildTree()
{
    QStringList headers = GlobalTable[0];

    QVector<QStringList> data;
    for (int i = 1; i < GlobalTable.size(); i++)
        data.append(GlobalTable[i]);

    TreeNode* root = BuildDecisionTree(data, headers, {});

    TreeVisualization* v = new TreeVisualization(root);
    v->exec();
}
