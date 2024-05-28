#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

map<string, Size> init_constants();
Mat convertBinaryToBool(const Mat &mask);
Mat adjustLuminance(const Mat &gray, double factor);
double getMeanStatistic(const Mat &gray, const Mat &mask);
double meanLuminanceRatio(const Mat &grayA, const Mat &grayB, const Mat &mask);
Mat getMask(const Mat &img);
Mat getOverlapRegionMask(const Mat &imA, const Mat &imB);
vector<Point> getOutmostPolygonBoundary(const Mat &img);
pair<Mat, Mat> getWeightMaskMatrix(const Mat &imA, const Mat &imB, double distThreshold = 5);
Mat makeWhiteBalance(const Mat &image);
// void process_image(const vector<string> &images, const string &output_path, const vector<FisheyeCameraModel> &camera_models);
// void process_video(const vector<string> &input_videos, const string &output_path, const vector<FisheyeCameraModel> &camera_models);

#endif // UTILITIES_HPP
