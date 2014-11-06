#include "transform.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "data.h"

using namespace cv;

cv::Mat transform_from_images(cv::Mat prev, cv::Mat prev_depth, cv::Mat cur, cv::Mat cur_depth) {
  cv::cvtColor(prev, prev, COLOR_BGR2GRAY);
  cv::cvtColor(cur, cur, COLOR_BGR2GRAY);
  cv::imshow("thing", prev);

  // TODO these both have parameters should be hyperized
  auto corner_threshold = 0.035;
  auto ratio_threshold = 1.0;
  SiftFeatureDetector detector;
  //OrbFeatureDetector detector;

  SiftDescriptorExtractor extractor;
  //OrbDescriptorExtractor extractor;
  //SurfDescriptorExtractor extractor; //Worse

  // Brute Force matcher
  // TODO look into faster
  //BruteForceMatcher<L2<float> > matcher;
  BFMatcher matcher;

  std::vector<KeyPoint> prev_keypoints, cur_keypoints;
  detector.detect(prev, prev_keypoints);
  detector.detect(cur, cur_keypoints);

  Mat prev_descriptors, cur_descriptors;
  extractor.compute(prev, prev_keypoints, prev_descriptors);
  extractor.compute(cur, cur_keypoints, cur_descriptors);

  //std::vector<DMatch> matches;
  std::vector<std::vector<DMatch>> matches;
  //matcher.match(cur_descriptors, prev_descriptors, matches);
  matcher.knnMatch(prev_descriptors, cur_descriptors, matches, 2);

  std::vector<DMatch> flatten_matches;

  for (auto vec_match : matches) {
    auto m = vec_match[0];
    auto n = vec_match[1];
    if (m.distance < ratio_threshold * n.distance &&
        prev_keypoints[m.queryIdx].response > corner_threshold &&
        cur_keypoints[m.trainIdx].response > corner_threshold) {
          flatten_matches.push_back(m);
    }
  }


  namedWindow("matches", 1);
  Mat img_matches;

  drawMatches(prev, prev_keypoints, cur, cur_keypoints, flatten_matches, img_matches);
  //drawMatches(cur, cur_keypoints, prev, prev_keypoints, matches, img_matches);
  imshow("matches", img_matches);

  std::vector<Point3f> prev_points;
  std::vector<Point3f> cur_points;
  for (auto match : flatten_matches) {

    auto prev_keypoint = prev_keypoints[match.queryIdx];
    auto cur_keypoint = cur_keypoints[match.trainIdx];
    auto prev_pt = to_3d_pos(prev_depth, prev_keypoint.pt);
    auto cur_pt = to_3d_pos(cur_depth, cur_keypoint.pt);

    if (prev_pt.z <= 0.1 || cur_pt.z <= 0.1) {
      continue;
    }
    std::cout << prev_keypoint.pt << "__" << cur_keypoint.pt << std::endl;
    std::cout << prev_pt << "__" << cur_pt << std::endl;

    prev_points.push_back(prev_pt);
    cur_points.push_back(cur_pt);
  }

  std::vector<uchar> inliers;
  cv::Mat aff(3,4, CV_32F);
  auto ransac_threshold = 1;
  //auto ransac_threshold = 0.1;
  auto confidence = 0.99;
  estimateAffine3D(prev_points, cur_points, aff, inliers, ransac_threshold, confidence);

  auto test = Mat(prev_points[0], CV_32F);
  std::cout << "preconcat" << std::endl;

  cv::Mat mat;
  vconcat(test, Mat::ones(1,1, CV_32F), mat);
  test = mat;

  std::cout << test << "test" << std::endl;
  cv::transpose(aff, aff);
  cv::transpose(test, test);

  std::cout << aff.size() << std::endl;
  std::cout << test.size() << std::endl;

  cv::Mat out;
  transform(test, out, aff);

  std::cout << "prev" << out << std::endl;

  waitKey(0);

  return aff;
}
