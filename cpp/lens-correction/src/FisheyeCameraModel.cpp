#include "FisheyeCameraModel.hpp"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <filesystem>
#include "utilities.hpp"

using namespace std;
using namespace cv;

FisheyeCameraModel::FisheyeCameraModel(const string &camera_param_file, const string &camera_name)
{
    if (!filesystem::exists(camera_param_file))
    {
        throw runtime_error("Cannot find camera param file");
    }

    this->camera_file = camera_param_file;
    this->camera_name = camera_name;
    this->scale_xy = Point2f(1.0, 1.0);
    this->shift_xy = Point2f(0, 0);
    this->project_shape = init_constants()[this->camera_name];
    this->load_camera_params();
}

Mat FisheyeCameraModel::readMatrixSimple(const YAML::Node &node)
{
    int rows = node["rows"].as<int>();
    int cols = node["cols"].as<int>();
    vector<double> data = node["data"].as<vector<double>>();
    Mat mat(rows, cols, CV_64F, data.data());
    return mat.clone(); // Ensure the data is properly managed
}

Point2f FisheyeCameraModel::readPoint2fSimple(const YAML::Node &node)
{
    vector<float> data = node["data"].as<vector<float>>();
    if (data.size() != 2)
    {
        throw runtime_error("Expected a 2-element sequence for Point2f");
    }
    return Point2f(data[0], data[1]);
}

void FisheyeCameraModel::load_camera_params()
{
    YAML::Node config = YAML::LoadFile(camera_file);

    try
    {
        camera_matrix = readMatrixSimple(config["camera_matrix"]);
        dist_coeffs = readMatrixSimple(config["dist_coeffs"]);
        resolution = readMatrixSimple(config["resolution"]);
        resolution = resolution.reshape(1, 2); // Ensure resolution is 1x2

        scale_xy = readPoint2fSimple(config["scale_xy"]);
        shift_xy = readPoint2fSimple(config["shift_xy"]);

        project_matrix = readMatrixSimple(config["project_matrix"]);

        // Ensure the undistortion maps are updated
        this->update_undistort_maps();
    }
    catch (const YAML::Exception &e)
    {
        cerr << "Error reading YAML file: " << e.what() << endl;
        throw runtime_error("Failed to load camera parameters from file: " + camera_file);
    }
}

void FisheyeCameraModel::update_undistort_maps()
{
    Mat new_matrix = camera_matrix.clone();
    new_matrix.at<double>(0, 0) *= scale_xy.x;
    new_matrix.at<double>(1, 1) *= scale_xy.y;
    new_matrix.at<double>(0, 2) += shift_xy.x;
    new_matrix.at<double>(1, 2) += shift_xy.y;

    int width = static_cast<int>(resolution.at<double>(0, 0));
    int height = static_cast<int>(resolution.at<double>(0, 1));

    Mat ud1, ud2;
    fisheye::initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        Mat::eye(3, 3, CV_64F),
        new_matrix,
        Size(width, height),
        CV_16SC2,
        ud1,
        ud2);
    undistort_map1 = ud1.clone();
    undistort_map2 = ud2.clone();
}

void FisheyeCameraModel::set_scale_and_shift(Point2f scale_xy, Point2f shift_xy)
{
    this->scale_xy = scale_xy;
    this->shift_xy = shift_xy;
    this->update_undistort_maps();
}

Mat FisheyeCameraModel::undistort(const Mat &image) const
{
    Mat result;
    remap(image, result, undistort_map1, undistort_map2, INTER_LINEAR, BORDER_CONSTANT);
    return result;
}

Mat FisheyeCameraModel::project(const Mat &image) const
{
    Mat result;
    warpPerspective(image, result, project_matrix, project_shape);
    return result;
}

Mat FisheyeCameraModel::flip(const Mat &image) const
{
    if (camera_name == "front")
    {
        return image.clone();
    }
    else if (camera_name == "back")
    {
        Mat flipped;
        cv::flip(image, flipped, -1); // Flip both horizontally and vertically
        return flipped;
    }
    else if (camera_name == "left")
    {
        Mat transposed;
        transpose(image, transposed);
        cv::flip(transposed, transposed, 0); // Flip vertically
        return transposed;
    }
    else
    {
        Mat transposed;
        transpose(image, transposed);
        cv::flip(transposed, transposed, 1); // Flip horizontally
        return transposed;
    }
}

void FisheyeCameraModel::save_data() const
{
    FileStorage fs(camera_file, FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "dist_coeffs" << dist_coeffs;
    fs << "resolution" << resolution;
    fs << "project_matrix" << project_matrix;
    fs << "scale_xy" << scale_xy;
    fs << "shift_xy" << shift_xy;
    fs.release();
}
