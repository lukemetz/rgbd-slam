#include "transform.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;

cv::Mat transform_from_images(cv::Mat prev, cv::Mat cur) {
  cv::imshow("thing", prev);

  // TODO these both have parameters
  GoodFeaturesToTrackDetector detector;
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

  std::vector<DMatch> matches;
  matcher.match(cur_descriptors, prev_descriptors, matches);

  std::cout << prev_keypoints.size() << std::endl;
  std::cout << cur_keypoints.size() << std::endl;
  std::cout << matches.size() << std::endl;


  std::vector<DMatch> trimmed_matches(matches.begin(), matches.begin()+60);
  //std::cout << trimmed_matches.size() << "Size" << std::endl;

  namedWindow("matches", 1);
  Mat img_matches;

  //drawMatches(prev, prev_keypoints, cur, cur_keypoints, matches, img_matches);

  drawMatches(cur, cur_keypoints, prev, prev_keypoints, trimmed_matches, img_matches);
  //drawMatches(cur, cur_keypoints, prev, prev_keypoints, matches, img_matches);
  imshow("matches", img_matches);

  // Sort into sift distance / how good the matches are
  std::sort(matches.begin(), matches.end());

  for (int i=0; i < 10; i++) {
    std::cout << "distance" <<  matches[i].distance << std::endl;
    std::cout << "   " <<  matches[i].trainIdx << ", "
      << matches[i].queryIdx << ", "
      << matches[i].imgIdx
      << std::endl;
  }

  waitKey(0);


  return cur;
}
