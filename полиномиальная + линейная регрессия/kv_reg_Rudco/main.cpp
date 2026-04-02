#include "lin_reg.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    lin_reg w;
    w.show();
    return a.exec();
}
