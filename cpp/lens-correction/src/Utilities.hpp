#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <opencv2/opencv.hpp>

std::map<std::string, cv::Size> init_constants();
cv::Mat convertBinaryToBool(const cv::Mat &mask);
cv::Mat adjustLuminance(const cv::Mat &gray, double factor);
double getMeanStatistic(const cv::Mat &gray, const cv::Mat &mask);
double meanLuminanceRatio(const cv::Mat &grayA, const cv::Mat &grayB, const cv::Mat &mask);
cv::Mat getMask(const cv::Mat &img);
cv::Mat getOverlapRegionMask(const cv::Mat &imA, const cv::Mat &imB);
std::vector<cv::Point> getOutmostPolygonBoundary(const cv::Mat &img);
std::pair<cv::Mat, cv::Mat> getWeightMaskMatrix(const cv::Mat &imA, const cv::Mat &imB, double distThreshold = 5);
cv::Mat makeWhiteBalance(const cv::Mat &image);
// void process_image(const std::vector<std::string> &images, const std::string &output_path, const std::vector<FisheyeCameraModel> &camera_models);
// void process_video(const std::vector<std::string> &input_videos, const std::string &output_path, const std::vector<FisheyeCameraModel> &camera_models);

#endif // UTILITIES_HPP
