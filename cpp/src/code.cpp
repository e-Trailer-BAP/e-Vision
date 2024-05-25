#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <filesystem>

// Use OpenCV namespaces to avoid qualifying every OpenCV function call with "cv::"
using namespace cv;
using namespace std;

// Declare global variables and constants
std::vector<std::string> camera_names = {"front", "back", "left", "right"};
std::string data_path = "C:/Users/Infer/Documents/Git/BAP/e-Vision/data"; // Set your data path here
// std::string output_path = "/path/to/your/output"; // Set your output path here

// --------------------------------------------------------------------
// (shift_width, shift_height): how far away the birdview looks outside
// of the calibration pattern in horizontal and vertical directions
const int shift_w = 300;
const int shift_h = 300;

// Size of the gap between the calibration pattern and the car
// in horizontal and vertical directions
const int inn_shift_w = 20;
const int inn_shift_h = 50;

// Total width/height of the stitched image
const int total_w = 600 + 2 * shift_w;
const int total_h = 1000 + 2 * shift_h;

// Four corners of the rectangular region occupied by the car
// top-left (x_left, y_top), bottom-right (x_right, y_bottom)
const int xl = shift_w + 180 + inn_shift_w;
const int xr = total_w - xl;
const int yt = shift_h + 200 + inn_shift_h;
const int yb = total_h - yt;
// --------------------------------------------------------------------

std::map<std::string, cv::Size> project_shapes = {
    {"front", cv::Size(total_w, yt)},
    {"back", cv::Size(total_w, yt)},
    {"left", cv::Size(total_h, xl)},
    {"right", cv::Size(total_h, xl)}};

std::map<std::string, std::vector<cv::Point>> project_keypoints = {
    {"front", {cv::Point(shift_w + 120, shift_h), cv::Point(shift_w + 480, shift_h), cv::Point(shift_w + 120, shift_h + 160), cv::Point(shift_w + 480, shift_h + 160)}},

    {"back", {cv::Point(shift_w + 120, shift_h), cv::Point(shift_w + 480, shift_h), cv::Point(shift_w + 120, shift_h + 160), cv::Point(shift_w + 480, shift_h + 160)}},

    {"left", {cv::Point(shift_h + 280, shift_w), cv::Point(shift_h + 840, shift_w), cv::Point(shift_h + 280, shift_w + 160), cv::Point(shift_h + 840, shift_w + 160)}},

    {"right", {cv::Point(shift_h + 160, shift_w), cv::Point(shift_h + 720, shift_w), cv::Point(shift_h + 160, shift_w + 160), cv::Point(shift_h + 720, shift_w + 160)}}};

std::string car_image_path = data_path + "/images/car.png";
cv::Mat car_image = cv::imread(car_image_path);
void load_car_image()
{
    std::string car_image_path = data_path + "/images/car.png";
    car_image = cv::imread(car_image_path);

    if (!car_image.empty())
    {
        cv::resize(car_image, car_image, cv::Size(xr - xl, yb - yt));
    }
    else
    {
        std::cerr << "Error: Unable to load car image: " << car_image_path << std::endl;
    }
}

class FisheyeCameraModel
{
public:
    FisheyeCameraModel(const std::string &camera_param_file, const std::string &camera_name)
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
        int width = resolution.at<int>(0);
        int height = resolution.at<int>(1);

        cv::fisheye::initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            cv::Mat::eye(3, 3, CV_64F),
            new_matrix,
            cv::Size(width, height),
            CV_16SC2,
            undistort_map1,
            undistort_map2);
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

Mat convertBinaryToBool(const Mat &mask)
{
    Mat boolMask;
    mask.convertTo(boolMask, CV_32F, 1.0 / 255.0);
    return boolMask;
}

Mat adjustLuminance(const Mat &gray, double factor)
{
    Mat adjusted;
    gray.convertTo(adjusted, -1, factor);
    return adjusted;
}

double getMeanStatistic(const Mat &gray, const Mat &mask)
{
    Scalar sum = cv::sum(gray.mul(mask));
    return sum[0];
}

double meanLuminanceRatio(const Mat &grayA, const Mat &grayB, const Mat &mask)
{
    return getMeanStatistic(grayA, mask) / getMeanStatistic(grayB, mask);
}

Mat getMask(const Mat &img)
{
    Mat gray, mask;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY);
    return mask;
}

Mat getOverlapRegionMask(const Mat &imA, const Mat &imB)
{
    Mat overlap, mask;
    cv::bitwise_and(imA, imB, overlap);
    mask = getMask(overlap);
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)), cv::Point(-1, -1), 2);
    return mask;
}

vector<Point> getOutmostPolygonBoundary(const Mat &img)
{
    Mat mask = getMask(img);
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)), cv::Point(-1, -1), 2);
    vector<vector<Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    vector<Point> largestContour = *max_element(contours.begin(), contours.end(), [](const vector<Point> &a, const vector<Point> &b)
                                                { return cv::contourArea(a) < cv::contourArea(b); });

    vector<Point> polygon;
    cv::approxPolyDP(largestContour, polygon, 0.009 * cv::arcLength(largestContour, true), true);

    return polygon;
}

pair<Mat, Mat> getWeightMaskMatrix(const Mat &imA, const Mat &imB, double distThreshold = 5)
{
    Mat overlapMask = getOverlapRegionMask(imA, imB);
    Mat overlapMaskInv;
    cv::bitwise_not(overlapMask, overlapMaskInv);

    Mat imA_diff, imB_diff;
    cv::bitwise_and(imA, imA, imA_diff, overlapMaskInv);
    cv::bitwise_and(imB, imB, imB_diff, overlapMaskInv);

    Mat G = getMask(imA);
    G.convertTo(G, CV_32F, 1.0 / 255.0);

    vector<Point> polyA = getOutmostPolygonBoundary(imA_diff);
    vector<Point> polyB = getOutmostPolygonBoundary(imB_diff);

    for (int y = 0; y < overlapMask.rows; ++y)
    {
        for (int x = 0; x < overlapMask.cols; ++x)
        {
            if (overlapMask.at<uchar>(y, x) == 255)
            {
                Point2f pt(x, y);
                double distToB = cv::pointPolygonTest(polyB, pt, true);
                if (distToB < distThreshold)
                {
                    double distToA = cv::pointPolygonTest(polyA, pt, true);
                    distToB *= distToB;
                    distToA *= distToA;
                    G.at<float>(y, x) = distToB / (distToA + distToB);
                }
            }
        }
    }

    return make_pair(G, overlapMask);
}

Mat makeWhiteBalance(const Mat &image)
{
    vector<Mat> channels;
    cv::split(image, channels);
    Scalar means = cv::mean(image);
    double K = (means[0] + means[1] + means[2]) / 3.0;

    channels[0] = adjustLuminance(channels[0], K / means[0]);
    channels[1] = adjustLuminance(channels[1], K / means[1]);
    channels[2] = adjustLuminance(channels[2], K / means[2]);

    Mat balancedImage;
    cv::merge(channels, balancedImage);
    return balancedImage;
}

class BirdView
{
public:
    BirdView()
    {
        this->image = cv::Mat::zeros(total_h, total_w, CV_8UC3);
        this->car_image = ::car_image;
    }

    void update_frames(const std::vector<cv::Mat> &images)
    {
        this->frames = images;
    }

    cv::Mat merge(const cv::Mat &imA, const cv::Mat &imB, int k)
    {
        cv::Mat G = this->weights[k];
        cv::Mat blended, merged;
        cv::add(imA.mul(G), imB.mul(1 - G), blended);
        blended.convertTo(merged, CV_8UC3);
        return merged;
    }

    cv::Mat FL() const { return this->image(cv::Rect(0, 0, xl, yt)); }
    cv::Mat F() const { return this->image(cv::Rect(xl, 0, xr - xl, yt)); }
    cv::Mat FR() const { return this->image(cv::Rect(xr, 0, this->image.cols - xr, yt)); }
    cv::Mat BL() const { return this->image(cv::Rect(0, yb, xl, this->image.rows - yb)); }
    cv::Mat B() const { return this->image(cv::Rect(xl, yb, xr - xl, this->image.rows - yb)); }
    cv::Mat BR() const { return this->image(cv::Rect(xr, yb, this->image.cols - xr, this->image.rows - yb)); }
    cv::Mat L() const { return this->image(cv::Rect(0, yt, xl, yb - yt)); }
    cv::Mat R() const { return this->image(cv::Rect(xr, yt, this->image.cols - xr, yb - yt)); }
    cv::Mat C() const { return this->image(cv::Rect(xl, yt, xr - xl, yb - yt)); }

    void stitch_all_parts()
    {
        const auto &front = frames[0];
        const auto &back = frames[1];
        const auto &left = frames[2];
        const auto &right = frames[3];

        front(cv::Rect(xl, 0, xr - xl, yt)).copyTo(this->F());
        back(cv::Rect(xl, yb, xr - xl, back.rows - yb)).copyTo(this->B());
        left(cv::Rect(0, yt, xl, yb - yt)).copyTo(this->L());
        right(cv::Rect(xr, yt, right.cols - xr, yb - yt)).copyTo(this->R());

        this->merge(front(cv::Rect(0, 0, xl, yt)), left(cv::Rect(0, 0, xl, yt)), 0).copyTo(this->FL());
        this->merge(front(cv::Rect(xr, 0, front.cols - xr, yt)), right(cv::Rect(0, 0, xl, yt)), 1).copyTo(this->FR());
        this->merge(back(cv::Rect(0, yb, xl, back.rows - yb)), left(cv::Rect(yb, 0, xl, left.rows - yb)), 2).copyTo(this->BL());
        this->merge(back(cv::Rect(xr, yb, back.cols - xr, back.rows - yb)), right(cv::Rect(yb, 0, xl, right.rows - yb)), 3).copyTo(this->BR());
    }

    void copy_car_image()
    {
        this->car_image.copyTo(this->C());
    }

    BirdView &make_luminance_balance()
    {
        auto tune = [](double x)
        {
            return x >= 1 ? x * std::exp((1 - x) * 0.5) : x * std::exp((1 - x) * 0.8);
        };

        const auto &front = frames[0];
        const auto &back = frames[1];
        const auto &left = frames[2];
        const auto &right = frames[3];

        std::vector<cv::Mat> front_channels, back_channels, left_channels, right_channels;
        cv::split(front, front_channels);
        cv::split(back, back_channels);
        cv::split(left, left_channels);
        cv::split(right, right_channels);

        auto a1 = meanLuminanceRatio(RII(right_channels[0]), FII(front_channels[0]), masks[1]);
        auto a2 = meanLuminanceRatio(RII(right_channels[1]), FII(front_channels[1]), masks[1]);
        auto a3 = meanLuminanceRatio(RII(right_channels[2]), FII(front_channels[2]), masks[1]);

        auto b1 = meanLuminanceRatio(BIV(back_channels[0]), RIV(right_channels[0]), masks[3]);
        auto b2 = meanLuminanceRatio(BIV(back_channels[1]), RIV(right_channels[1]), masks[3]);
        auto b3 = meanLuminanceRatio(BIV(back_channels[2]), RIV(right_channels[2]), masks[3]);

        auto c1 = meanLuminanceRatio(LIII(left_channels[0]), BIII(back_channels[0]), masks[2]);
        auto c2 = meanLuminanceRatio(LIII(left_channels[1]), BIII(back_channels[1]), masks[2]);
        auto c3 = meanLuminanceRatio(LIII(left_channels[2]), BIII(back_channels[2]), masks[2]);

        auto d1 = meanLuminanceRatio(FI(front_channels[0]), LI(left_channels[0]), masks[0]);
        auto d2 = meanLuminanceRatio(FI(front_channels[1]), LI(left_channels[1]), masks[0]);
        auto d3 = meanLuminanceRatio(FI(front_channels[2]), LI(left_channels[2]), masks[0]);

        double t1 = std::pow(a1 * b1 * c1 * d1, 0.25);
        double t2 = std::pow(a2 * b2 * c2 * d2, 0.25);
        double t3 = std::pow(a3 * b3 * c3 * d3, 0.25);

        double x1 = t1 / std::sqrt(d1 / a1);
        double x2 = t2 / std::sqrt(d2 / a2);
        double x3 = t3 / std::sqrt(d3 / a3);

        x1 = tune(x1);
        x2 = tune(x2);
        x3 = tune(x3);

        front_channels[0] = adjustLuminance(front_channels[0], x1);
        front_channels[1] = adjustLuminance(front_channels[1], x2);
        front_channels[2] = adjustLuminance(front_channels[2], x3);

        double y1 = t1 / std::sqrt(b1 / c1);
        double y2 = t2 / std::sqrt(b2 / c2);
        double y3 = t3 / std::sqrt(b3 / c3);

        y1 = tune(y1);
        y2 = tune(y2);
        y3 = tune(y3);

        back_channels[0] = adjustLuminance(back_channels[0], y1);
        back_channels[1] = adjustLuminance(back_channels[1], y2);
        back_channels[2] = adjustLuminance(back_channels[2], y3);

        double z1 = t1 / std::sqrt(c1 / d1);
        double z2 = t2 / std::sqrt(c2 / d2);
        double z3 = t3 / std::sqrt(c3 / d3);

        z1 = tune(z1);
        z2 = tune(z2);
        z3 = tune(z3);

        left_channels[0] = adjustLuminance(left_channels[0], z1);
        left_channels[1] = adjustLuminance(left_channels[1], z2);
        left_channels[2] = adjustLuminance(left_channels[2], z3);

        double w1 = t1 / std::sqrt(a1 / b1);
        double w2 = t2 / std::sqrt(a2 / b2);
        double w3 = t3 / std::sqrt(a3 / b3);

        w1 = tune(w1);
        w2 = tune(w2);
        w3 = tune(w3);

        right_channels[0] = adjustLuminance(right_channels[0], w1);
        right_channels[1] = adjustLuminance(right_channels[1], w2);
        right_channels[2] = adjustLuminance(right_channels[2], w3);

        cv::merge(front_channels, frames[0]);
        cv::merge(back_channels, frames[1]);
        cv::merge(left_channels, frames[2]);
        cv::merge(right_channels, frames[3]);

        return *this;
    }

    std::tuple<cv::Mat, cv::Mat> get_weights_and_masks(const std::vector<cv::Mat> &images)
    {
        const auto &front = images[0];
        const auto &back = images[1];
        const auto &left = images[2];
        const auto &right = images[3];

        auto [G0, M0] = getWeightMaskMatrix(FI(front), LI(left));
        auto [G1, M1] = getWeightMaskMatrix(FII(front), RII(right));
        auto [G2, M2] = getWeightMaskMatrix(BIII(back), LIII(left));
        auto [G3, M3] = getWeightMaskMatrix(BIV(back), RIV(right));

        weights = {cv::Mat(G0), cv::Mat(G1), cv::Mat(G2), cv::Mat(G3)};
        masks = {cv::Mat(M0) / 255, cv::Mat(M1) / 255, cv::Mat(M2) / 255, cv::Mat(M3) / 255};

        return {cv::Mat::zeros(1, 4, CV_8UC1), cv::Mat::zeros(1, 4, CV_8UC1)}; // Adjust as needed
    }

    void make_white_balance()
    {
        this->image = ::makeWhiteBalance(this->image);
    }

    cv::Mat getImage() const
    {
        return this->image;
    }

private:
    cv::Mat image;
    cv::Mat car_image;
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> weights;
    std::vector<cv::Mat> masks;

    cv::Mat FI(const cv::Mat &img) const { return img(cv::Rect(0, 0, xl, img.rows)); }
    cv::Mat FII(const cv::Mat &img) const { return img(cv::Rect(xr, 0, img.cols - xr, img.rows)); }
    cv::Mat FM(const cv::Mat &img) const { return img(cv::Rect(xl, 0, xr - xl, img.rows)); }
    cv::Mat BIII(const cv::Mat &img) const { return img(cv::Rect(0, 0, xl, img.rows)); }
    cv::Mat BIV(const cv::Mat &img) const { return img(cv::Rect(xr, 0, img.cols - xr, img.rows)); }
    cv::Mat BM(const cv::Mat &img) const { return img(cv::Rect(xl, 0, xr - xl, img.rows)); }
    cv::Mat LI(const cv::Mat &img) const { return img(cv::Rect(0, 0, img.cols, yt)); }
    cv::Mat LIII(const cv::Mat &img) const { return img(cv::Rect(0, yb, img.cols, img.rows - yb)); }
    cv::Mat LM(const cv::Mat &img) const { return img(cv::Rect(0, yt, img.cols, yb - yt)); }
    cv::Mat RII(const cv::Mat &img) const { return img(cv::Rect(0, 0, img.cols, yt)); }
    cv::Mat RIV(const cv::Mat &img) const { return img(cv::Rect(0, yb, img.cols, img.rows - yb)); }
    cv::Mat RM(const cv::Mat &img) const { return img(cv::Rect(0, yt, img.cols, yb - yt)); }
};

void main_function()
{
    // Load the car image
    load_car_image();

    // Initialize camera names and paths to images and YAML files
    std::vector<std::string> names = camera_names;
    std::vector<std::string> images;
    std::vector<std::string> yamls;

    for (const auto &name : names)
    {
        images.push_back(data_path + "/images/" + name + ".png");
        yamls.push_back(data_path + "/yaml/" + name + ".yaml");
    }

    // Initialize camera models
    std::vector<FisheyeCameraModel> camera_models;
    for (size_t i = 0; i < names.size(); ++i)
    {
        camera_models.emplace_back(yamls[i], names[i]);
    }

    // Process images using the camera models
    std::vector<cv::Mat> projected;
    for (size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat img = cv::imread(images[i]);
        if (img.empty())
        {
            throw std::runtime_error("Could not open or find the image: " + images[i]);
        }
        img = camera_models[i].undistort(img);
        img = camera_models[i].project(img);
        img = camera_models[i].flip(img);
        projected.push_back(img);
    }

    // Initialize BirdView and perform operations
    BirdView birdview;
    auto [Gmat, Mmat] = birdview.get_weights_and_masks(projected);
    birdview.update_frames(projected);

    birdview.make_luminance_balance();
    birdview.stitch_all_parts();
    birdview.copy_car_image();
    birdview.make_white_balance();

    std::cout << "Bird's eye view stitched image after added white balance" << std::endl;

    // Display the resulting birdview image
    cv::imshow("BirdView", birdview.getImage());
    cv::waitKey(0);

    // // Save weights and masks images to the output path
    // cv::imwrite(output_path + "/weights.png", Gmat * 255);
    // cv::imwrite(output_path + "/masks.png", Mmat);
}

int main(int argc, char **argv)
{
    try
    {
        main_function();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
