#include "BirdView.hpp"
#include "Utilities.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

BirdView::BirdView()
{
    map<string, Size> project_shapes = init_constants();
    setParams(project_shapes);
    this->image = Mat::zeros(total_h, total_w, CV_8UC3);
}

void BirdView::update_frames(const vector<Mat> &new_frames)
{
    // Set frames
    this->frames = new_frames;
}

Mat BirdView::merge(const Mat &imA, const Mat &imB, int k)
{
    // Ensure k is a valid index
    if (k < 0 || k >= weights.size())
    {
        throw out_of_range("Invalid index for weights.");
    }

    Mat G = weights[k];

    // Ensure that the input images have the same size and type
    CV_Assert(imA.size() == imB.size());
    CV_Assert(imA.channels() == imB.channels());

    // If G is single channel and imA/imB are multi-channel, replicate G across channels
    if (G.channels() == 1 && imA.channels() > 1)
    {
        Mat channels[3];
        for (int i = 0; i < 3; ++i)
        {
            channels[i] = G;
        }
        cv::merge(channels, 3, G);
    }

    // Convert images and G to double for accurate calculations
    Mat imA_double, imB_double, G_double;
    imA.convertTo(imA_double, CV_64F);
    imB.convertTo(imB_double, CV_64F);
    G.convertTo(G_double, CV_64F);

    // Perform the weighted merge
    Mat merged = imA_double.mul(G_double) + imB_double.mul(Scalar(1.0, 1.0, 1.0) - G_double);

    // Convert back to uint8
    merged.convertTo(merged, CV_8U);

    return merged;
}

void BirdView::stitch_all_parts()
{
    const auto &front = frames[0];
    const auto &back = frames[1];
    const auto &left = frames[2];
    const auto &right = frames[3];

    front(Rect(xl, 0, xr - xl, yt)).copyTo(this->F());
    back(Rect(xl, 0, xr - xl, yt)).copyTo(this->B());
    left(Rect(0, yt, xl, yb - yt)).copyTo(this->L());
    right(Rect(0, yt, xl, yb - yt)).copyTo(this->R());

    this->merge(front(Rect(0, 0, xl, yt)), left(Rect(0, 0, xl, yt)), 0).copyTo(this->FL());
    this->merge(front(Rect(xr, 0, xl, yt)), right(Rect(0, 0, xl, yt)), 1).copyTo(this->FR());
    this->merge(back(Rect(0, 0, xl, yt)), left(Rect(0, yb, xl, yt)), 2).copyTo(this->BL());
    this->merge(back(Rect(xr, 0, xl, yt)), right(Rect(0, yb, xl, yt)), 3).copyTo(this->BR());
}
void BirdView::load_car_image(const string &data_path)
{
    // Set car image
    string car_image_path = data_path + "/images/car.png";
    this->car_image = imread(car_image_path);
    if (!this->car_image.empty())
    {
        resize(this->car_image, this->car_image, Size(xr - xl, yb - yt));
    }
    else
    {
        cerr << "Error: Unable to load car image: " << car_image_path << endl;
    }
    return;
}

void BirdView::copy_car_image()
{
    this->car_image.copyTo(this->C());
}

// set bird's eye view projection params
void BirdView::setParams(map<string, Size> project_shapes)
{
    this->total_w = project_shapes["front"].width;
    this->total_h = project_shapes["left"].width;
    this->yt = project_shapes["front"].height;
    this->yb = total_h - yt;
    this->xl = project_shapes["left"].height;
    this->xr = total_w - xl;
}

void BirdView::make_luminance_balance()
{
    auto tune = [](double x)
    {
        return x >= 1 ? x * exp((1 - x) * 0.5) : x * exp((1 - x) * 0.8);
    };

    const auto &front = frames[0];
    const auto &back = frames[1];
    const auto &left = frames[2];
    const auto &right = frames[3];

    vector<Mat> front_channels, back_channels, left_channels, right_channels;
    split(front, front_channels);
    split(back, back_channels);
    split(left, left_channels);
    split(right, right_channels);

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

    double t1 = pow(a1 * b1 * c1 * d1, 0.25);
    double t2 = pow(a2 * b2 * c2 * d2, 0.25);
    double t3 = pow(a3 * b3 * c3 * d3, 0.25);

    double x1 = t1 / sqrt(d1 / a1);
    double x2 = t2 / sqrt(d2 / a2);
    double x3 = t3 / sqrt(d3 / a3);

    x1 = tune(x1);
    x2 = tune(x2);
    x3 = tune(x3);

    front_channels[0] = adjustLuminance(front_channels[0], x1);
    front_channels[1] = adjustLuminance(front_channels[1], x2);
    front_channels[2] = adjustLuminance(front_channels[2], x3);

    double y1 = t1 / sqrt(b1 / c1);
    double y2 = t2 / sqrt(b2 / c2);
    double y3 = t3 / sqrt(b3 / c3);

    y1 = tune(y1);
    y2 = tune(y2);
    y3 = tune(y3);

    back_channels[0] = adjustLuminance(back_channels[0], y1);
    back_channels[1] = adjustLuminance(back_channels[1], y2);
    back_channels[2] = adjustLuminance(back_channels[2], y3);

    double z1 = t1 / sqrt(c1 / d1);
    double z2 = t2 / sqrt(c2 / d2);
    double z3 = t3 / sqrt(c3 / d3);

    z1 = tune(z1);
    z2 = tune(z2);
    z3 = tune(z3);

    left_channels[0] = adjustLuminance(left_channels[0], z1);
    left_channels[1] = adjustLuminance(left_channels[1], z2);
    left_channels[2] = adjustLuminance(left_channels[2], z3);

    double w1 = t1 / sqrt(a1 / b1);
    double w2 = t2 / sqrt(a2 / b2);
    double w3 = t3 / sqrt(a3 / b3);

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
}

tuple<Mat, Mat> BirdView::get_weights_and_masks(const vector<Mat> &images)
{
    const auto &front = images[0];
    const auto &back = images[1];
    const auto &left = images[2];
    const auto &right = images[3];

    auto [G0, M0] = getWeightMaskMatrix(FI(front), LI(left));
    auto [G1, M1] = getWeightMaskMatrix(FII(front), RII(right));
    auto [G2, M2] = getWeightMaskMatrix(BIII(back), LIII(left));
    auto [G3, M3] = getWeightMaskMatrix(BIV(back), RIV(right));

    weights = {Mat(G0), Mat(G1), Mat(G2), Mat(G3)};
    masks = {Mat(M0) / 255, Mat(M1) / 255, Mat(M2) / 255, Mat(M3) / 255};

    return {Mat::zeros(1, 4, CV_8UC1), Mat::zeros(1, 4, CV_8UC1)}; // Adjust as needed
}

void BirdView::make_white_balance()
{
    this->image = ::makeWhiteBalance(this->image);
}

Mat BirdView::getImage() const
{
    return this->image;
}

// Getting image partitions
// Front image - Left corner
Mat BirdView::FI(const Mat &img) const
{
    return img(Rect(0, 0, xl, yt));
}
// Front image - Right corner
Mat BirdView::FII(const Mat &img) const
{
    return img(Rect(xr, 0, xl, yt));
}
// Front image - Middle
Mat BirdView::FM(const Mat &img) const
{
    return img(Rect(xl, 0, xr - xl, yt));
}

// Back image - Left corner
Mat BirdView::BIII(const Mat &img) const
{
    return img(Rect(0, 0, xl, yt));
}
// Back image - Right corner
Mat BirdView::BIV(const Mat &img) const
{
    return img(Rect(xr, 0, xl, yt));
}
// Back image - Middle
Mat BirdView::BM(const Mat &img) const
{
    return img(Rect(xl, 0, xr - xl, yt));
}

// Left image - Top corner
Mat BirdView::LI(const Mat &img) const
{
    return img(Rect(0, 0, xl, yt));
}
// Left image - Bottom corner
Mat BirdView::LIII(const Mat &img) const
{
    return img(Rect(0, yb, xl, yt));
}
// Left image - Middle
Mat BirdView::LM(const Mat &img) const
{
    return img(Rect(0, yt, xl, yb - yt));
}

// Right image - Top corner
Mat BirdView::RII(const Mat &img) const
{
    return img(Rect(0, 0, xl, yt));
}
// Right image - Bottom corner
Mat BirdView::RIV(const Mat &img) const
{
    return img(Rect(0, yb, xl, yt));
}
// Right image - Middle
Mat BirdView::RM(const Mat &img) const
{
    return img(Rect(0, yt, xl, yb - yt));
}