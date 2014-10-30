#include <stdio.h>
#include "transform.h"
#include "qcustomplot.h"
#include <QApplication>

std::string data_path = "/Users/scope/slam_data/rgbd_dataset_freiburg1_rpy/";

std::string cur = "1305031229.528748.png";
std::string cur_d ="1305031229.564442.png";
std::string prev = "1305031229.528748.png";

cv::Mat rgb_image(std::string name) {
  return cv::imread(data_path+"rgb/"+name);
}

cv::Mat d_image(std::string name) {
  return cv::imread(data_path+"depth/"+name);
}

void basic_plot() {
  QCustomPlot *customPlot = new QCustomPlot();
  // generate some data:
  QVector<double> x(101), y(101); // initialize with entries 0..100
  for (int i=0; i<101; ++i)
  {
    x[i] = i/50.0 - 1; // x goes from -1 to 1
    y[i] = x[i]*x[i]; // let's plot a quadratic function
  }
  // create graph and assign data to it:
  customPlot->addGraph();
  customPlot->graph(0)->setData(x, y);
  // give the axes some labels:
  customPlot->xAxis->setLabel("x");
  customPlot->yAxis->setLabel("y");
  // set axes ranges, so we see all data:
  customPlot->xAxis->setRange(-1, 1);
  customPlot->yAxis->setRange(0, 1);
  customPlot->replot();
  customPlot->show();
  customPlot->resize(200, 200);
}

int main(int argc, char **argv) {
  QApplication app (argc, argv);
  basic_plot();


  cv::imshow("im", rgb_image(cur));
  cv::imshow("im2", d_image(cur_d));
  cv::waitKey(0);
  //app.exec();

  return 0;
}
