#include "data.h"

double fx = 525.0; // focal length x
double fy = 525.0; // focal length y
double cx = 319.5; // optical center x
double cy = 239.5; // optical center y
double factor = 5000.0;
//double factor = 1.0;

Point3f to_3d_pos(cv::Mat depth, Point2i point) {
    double Z = depth.at<unsigned short>(point.x, point.y) / factor;
    double X = (point.x - cx) * Z / fx;
    double Y = (point.y - cy) * Z / fy;
    return Point3f(X, Y, Z);
}
