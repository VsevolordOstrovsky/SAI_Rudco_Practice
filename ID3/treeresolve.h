#ifndef TREERESOLVE_H
#define TREERESOLVE_H

#define E(A,B) (-(A * log2(A) + B * log2(B)))

#include <QMainWindow>
#include <QVector>
#include <QStringList>
#include <QMap>
#include <QSet>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class TreeResolve; }
QT_END_NAMESPACE

struct TreeNode {
    QString attribute;
    QString value;
    bool isLeaf;
    QString decision;
    QMap<QString, TreeNode*> children;

    TreeNode() : isLeaf(false) {}
    ~TreeNode() { qDeleteAll(children); }
};

class TreeVisualization : public QDialog
{
    Q_OBJECT

public:
    TreeVisualization(TreeNode* root, QWidget *parent = nullptr);

private:
    void DrawTree();
    void DrawNode(TreeNode* node, qreal x, qreal y, qreal width);
    void DrawEdge(qreal x1, qreal y1, qreal x2, qreal y2, const QString& label);
    int CalculateWidth(TreeNode* node);

    QGraphicsScene* scene;
    QGraphicsView* view;
    TreeNode* root;
};

class TreeResolve : public QMainWindow
{
    Q_OBJECT

public:
    TreeResolve(QWidget *parent = nullptr);
    ~TreeResolve();

private slots:
    void OnLoadExcelFile();

private:
    Ui::TreeResolve *ui;

    QVector<QStringList> GlobalTable;

    void ParseCSVFile(const QString &filePath);
    void BuildTree();

    TreeNode* BuildDecisionTree(
        const QVector<QStringList>& data,
        const QStringList& attributes,
        QSet<int> usedAttributes
        );

    double CalculateEntropy(const QVector<QStringList>& data);
    double CalculateInformationGain(const QVector<QStringList>& data, int attrIndex, double parentEntropy);
    QVector<QStringList> FilterData(const QVector<QStringList>& data, int attrIndex, const QString& value);
    QString GetMajorityClass(const QVector<QStringList>& data);
};

#endif
