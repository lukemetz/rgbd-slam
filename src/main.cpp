#include <stdio.h>
#include "transform.h"
#include "qcustomplot.h"
#include <QApplication>
#include <armadillo>
#include <vector>
#include "data.h"


double cur = 1305031229.528748;
double prev = 1305031229.596600;
//std::string prev = "1305031232.093725.png";


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
  // set axes ranges, so we see all data:
  customPlot->xAxis->setRange(-1, 1);
  customPlot->yAxis->setRange(0, 1);
  customPlot->replot();
  customPlot->show();
  customPlot->resize(200, 200);
}

int main(int argc, char **argv) {
  //get_times("rgb");
  double time = 1305031229.596600;
  get_closest_depth(time);
  std::cout << get_times_vec("rgb")[0] << "Times vec" << std::endl;
  QApplication app (argc, argv);
  //basic_plot();


  /*
  cv::imshow("im", rgb_image(cur));
  cv::imshow("im2", d_image(cur_d));
  cv::waitKey(0);
  */

  cv::Mat prev_rgb = rgb_image(prev);
  cv::Mat cur_rgb = rgb_image(cur);

  cv::Mat prev_depth = depth_image_near(prev);
  cv::Mat cur_depth = depth_image_near(cur);
  /*
  std::vector<double> depths = get_times_vec("rgb");
  auto depths_out = arma::mat(depths.size(), 1);
  auto i = 0;
  for (double d : depths) {
    cv::Mat cur_depth = depth_image_near(d);
    //std::cout << cur_depth << std::endl;
    std::cout << to_3d_pos(cur_depth, Point2i(400, 300)) << std::endl;
    depths_out.at(i++, 0) = to_3d_pos(cur_depth, Point2i(320, 240)).z;
  } */

  //depths_out.save("plot.csv", arma::csv_ascii);

  //std::cout << to_3d_pos(cur_depth, Point2i(401, 300)) << std::endl;
  //std::cout << to_3d_pos(cur_depth, Point2i(402, 300)) << std::endl;
  //std::cout << to_3d_pos(cur_depth, Point2i(403, 300)) << std::endl;
  //imshow("this", cur_depth);
  //waitKey(0);

  std::vector<double> imgs = get_times_vec("rgb");
  for (int k=0; k < 7; ++k) {
    cv::Mat prev_rgb = rgb_image(imgs[k]);
    cv::Mat prev_depth = depth_image_near(imgs[k]);
    cv::Mat cur_rgb = rgb_image(imgs[k+1]);
    cv::Mat cur_depth = depth_image_near(imgs[k+1]);
    auto affine = transform_from_images(prev_rgb, prev_depth, cur_rgb, cur_depth);
    std::cout << affine << std::endl << std::endl;

    cv::Mat next_rgb = rgb_image(imgs[k+3]);
    cv::Mat next_depth = depth_image_near(imgs[k+3]);
    auto affine2 = transform_from_images(prev_rgb, prev_depth, next_rgb, next_depth);
    std::cout << affine2 << std::endl << std::endl;
    std::cout << "==" << std::endl;


  }


  //app.exec();

  return 0;
}
