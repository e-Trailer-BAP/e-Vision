#ifndef FISHEYE_CAMERA_MODEL_HPP
#define FISHEYE_CAMERA_MODEL_HPP

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <map>
#include <vector>

using namespace std;
using namespace cv;

class FisheyeCameraModel
{
public:
    FisheyeCameraModel(const string &camera_param_file, const string &camera_name);
    Mat readMatrixSimple(const YAML::Node &node);
    Point2f readPoint2fSimple(const YAML::Node &node);
    void load_camera_params();
    void update_undistort_maps();
    void set_scale_and_shift(Point2f scale_xy = {1.0, 1.0}, Point2f shift_xy = {0, 0});
    Mat undistort(const Mat &image) const;
    Mat project(const Mat &image) const;
    Mat flip(const Mat &image) const;
    void save_data() const;

private:
    string camera_file;
    string camera_name;
    Mat camera_matrix;
    Mat dist_coeffs;
    Mat resolution;
    Mat project_matrix;
    Size project_shape;
    Point2f scale_xy;
    Point2f shift_xy;
    Mat undistort_map1;
    Mat undistort_map2;
};

#endif // FISHEYE_CAMERA_MODEL_HPP
