#include "FisheyeCameraModel.hpp"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <filesystem>

// FisheyeCameraModel::FisheyeCameraModel(const std::string &camera_param_file, const std::string &camera_name)
FisheyeCameraModel::FisheyeCameraModel(const std::string &camera_param_file, const std::string &camera_name)
{
    if (!std::filesystem::exists(camera_param_file))
    {
        throw std::runtime_error("Cannot find camera param file");
    }

    if (std::find(camera_names.begin(), camera_names.end(), camera_name) == camera_names.end())
    {
        throw std::runtime_error("Unknown camera name: " + camera_name);
    }

    this->camera_file = camera_param_file;
    this->camera_name = camera_name;
    this->scale_xy = cv::Point2f(1.0, 1.0);
    this->shift_xy = cv::Point2f(0, 0);
    this->project_shape = project_shapes[this->camera_name];
    this->load_camera_params();
}

cv::Mat readMatrixSimple(const YAML::Node &node)
{
    int rows = node["rows"].as<int>();
    int cols = node["cols"].as<int>();
    std::vector<double> data = node["data"].as<std::vector<double>>();
    cv::Mat mat(rows, cols, CV_64F, data.data());
    return mat.clone(); // Ensure the data is properly managed
}

cv::Point2f readPoint2fSimple(const YAML::Node &node)
{
    std::vector<float> data = node["data"].as<std::vector<float>>();
    if (data.size() != 2)
    {
        throw std::runtime_error("Expected a 2-element sequence for Point2f");
    }
    return cv::Point2f(data[0], data[1]);
}

void load_camera_params()
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
        std::cerr << "Error reading YAML file: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load camera parameters from file: " + camera_file);
    }
}

void update_undistort_maps()
{
    cv::Mat new_matrix = camera_matrix.clone();
    new_matrix.at<double>(0, 0) *= scale_xy.x;
    new_matrix.at<double>(1, 1) *= scale_xy.y;
    new_matrix.at<double>(0, 2) += shift_xy.x;
    new_matrix.at<double>(1, 2) += shift_xy.y;

    int width = static_cast<int>(resolution.at<double>(0, 0));
    int height = static_cast<int>(resolution.at<double>(0, 1));

    cv::Mat ud1, ud2;
    cv::fisheye::initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        cv::Mat::eye(3, 3, CV_64F),
        new_matrix,
        cv::Size(width, height),
        CV_16SC2,
        ud1,
        ud2);
    undistort_map1 = ud1.clone();
    undistort_map2 = ud2.clone();
}

FisheyeCameraModel &set_scale_and_shift(cv::Point2f scale_xy = {1.0, 1.0}, cv::Point2f shift_xy = {0, 0})
{
    this->scale_xy = scale_xy;
    this->shift_xy = shift_xy;
    this->update_undistort_maps();
    return *this;
}

cv::Mat undistort(const cv::Mat &image) const
{
    cv::Mat result;
    cv::remap(image, result, undistort_map1, undistort_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return result;
}

cv::Mat project(const cv::Mat &image) const
{
    cv::Mat result;
    cv::warpPerspective(image, result, project_matrix, project_shape);
    return result;
}

cv::Mat flip(const cv::Mat &image) const
{
    if (camera_name == "front")
    {
        return image.clone();
    }
    else if (camera_name == "back")
    {
        cv::Mat flipped;
        cv::flip(image, flipped, -1); // Flip both horizontally and vertically
        return flipped;
    }
    else if (camera_name == "left")
    {
        cv::Mat transposed;
        cv::transpose(image, transposed);
        cv::flip(transposed, transposed, 0); // Flip vertically
        return transposed;
    }
    else
    {
        cv::Mat transposed;
        cv::transpose(image, transposed);
        cv::flip(transposed, transposed, 1); // Flip horizontally
        return transposed;
    }
}

void save_data() const
{
    cv::FileStorage fs(camera_file, cv::FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "dist_coeffs" << dist_coeffs;
    fs << "resolution" << resolution;
    fs << "project_matrix" << project_matrix;
    fs << "scale_xy" << scale_xy;
    fs << "shift_xy" << shift_xy;
    fs.release();
}
