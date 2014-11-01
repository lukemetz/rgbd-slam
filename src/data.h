#pragma one

#include <stdio.h>
#include "transform.h"
#include "qcustomplot.h"
#include <QApplication>
#include <armadillo>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;

static std::string data_path = "/Users/scope/slam_data/rgbd_dataset_freiburg1_rpy/";

inline arma::mat get_times(std::string prefix) {
  auto csv_name = data_path + prefix + ".csv";
  arma::mat mat;
  mat.load(csv_name, arma::csv_ascii);
  return mat;
}

inline std::vector<double> get_times_vec(std::string prefix) {
  arma::mat mat = get_times(prefix);
  arma::vec vector = mat.col(0);
  return arma::conv_to<std::vector<double>>::from(vector);
}

inline double get_closest_depth(double time) {
  auto mat = get_times("depth");
  auto resid = abs(time - mat);
  arma::uword min_index = -1;
  resid.min(min_index);
  return mat[min_index];
}

inline cv::Mat rgb_image(double time) {
  return cv::imread(data_path+"rgb/"+std::to_string(time)+".png");
}

inline cv::Mat depth_image(double time) {
  return cv::imread(data_path+"depth/"+std::to_string(time)+".png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
}

inline cv::Mat depth_image_near(double time) {
  double t = get_closest_depth(time);
  return depth_image(t);
}


Point3f to_3d_pos(cv::Mat depth, Point2i point);
