#ifndef FISHEYE_CAMERA_MODEL_HPP
#define FISHEYE_CAMERA_MODEL_HPP

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <map>
#include <vector>

class FisheyeCameraModel
{
public:
    FisheyeCameraModel(const std::string &camera_param_file, const std::string &camera_name);
    cv::Mat readMatrixSimple(const YAML::Node &node);
    cv::Point2f readPoint2fSimple(const YAML::Node &node);
    void load_camera_params();
    void update_undistort_maps();
    FisheyeCameraModel &set_scale_and_shift(cv::Point2f scale_xy = {1.0, 1.0}, cv::Point2f shift_xy = {0, 0});
    cv::Mat undistort(const cv::Mat &image) const;
    cv::Mat project(const cv::Mat &image) const;
    cv::Mat flip(const cv::Mat &image) const;
    void save_data() const;

private:
    std::string camera_file;
    std::string camera_name;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    cv::Mat resolution;
    cv::Mat project_matrix;
    cv::Size project_shape;
    cv::Point2f scale_xy;
    cv::Point2f shift_xy;
    cv::Mat undistort_map1;
    cv::Mat undistort_map2;
};

#endif // FISHEYE_CAMERA_MODEL_HPP
