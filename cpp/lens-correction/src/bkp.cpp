#include "BirdView.hpp"
#include "Utilities.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

BirdView::BirdView()
{
    std::map<std::string, cv::Size> project_shapes = init_constants();
    this->total_w = project_shapes["front"].width;
    this->total_h = project_shapes["left"].width;
    this->yt = project_shapes["front"].height;
    this->yb = total_h - yt;
    this->xl = project_shapes["left"].height;
    this->xr = total_w - xl;
    // this->image = cv::Mat::zeros(total_h, total_w, CV_8UC3);
    // load_car_image("../../../data");
}

void update_frames(const std::vector<cv::Mat> &images)
{
    this->frames = images;
}

cv::Mat merge(const cv::Mat &imA, const cv::Mat &imB, int k)
{
    // Ensure k is a valid index
    if (k < 0 || k >= weights.size())
    {
        throw std::out_of_range("Invalid index for weights.");
    }

    cv::Mat G = weights[k];

    // Ensure that the input images have the same size and type
    CV_Assert(imA.size() == imB.size());
    CV_Assert(imA.channels() == imB.channels());

    // If G is single channel and imA/imB are multi-channel, replicate G across channels
    if (G.channels() == 1 && imA.channels() > 1)
    {
        cv::Mat channels[3];
        for (int i = 0; i < 3; ++i)
        {
            channels[i] = G;
        }
        cv::merge(channels, 3, G);
    }

    // Convert images and G to double for accurate calculations
    cv::Mat imA_double, imB_double, G_double;
    imA.convertTo(imA_double, CV_64F);
    imB.convertTo(imB_double, CV_64F);
    G.convertTo(G_double, CV_64F);

    // Perform the weighted merge
    cv::Mat merged = imA_double.mul(G_double) + imB_double.mul(cv::Scalar(1.0, 1.0, 1.0) - G_double);

    // Convert back to uint8
    merged.convertTo(merged, CV_8U);

    return merged;
}

cv::Mat FL() { return this->image(cv::Rect(0, 0, xl, yt)); }
cv::Mat F() { return this->image(cv::Rect(xl, 0, xr - xl, yt)); }
cv::Mat FR() { return this->image(cv::Rect(xr, 0, xl, yt)); }
cv::Mat BL() { return this->image(cv::Rect(0, yb, xl, yt)); }
cv::Mat B() { return this->image(cv::Rect(xl, yb, xr - xl, yt)); }
cv::Mat BR() { return this->image(cv::Rect(xr, yb, xl, yt)); }
cv::Mat L() { return this->image(cv::Rect(0, yt, xl, yb - yt)); }
cv::Mat R() { return this->image(cv::Rect(xr, yt, xl, yb - yt)); }
cv::Mat C() { return this->image(cv::Rect(xl, yt, xr - xl, yb - yt)); }

void stitch_all_parts()
{
    const auto &front = frames[0];
    const auto &back = frames[1];
    const auto &left = frames[2];
    const auto &right = frames[3];

    front(cv::Rect(xl, 0, xr - xl, yt)).copyTo(this->F());
    back(cv::Rect(xl, 0, xr - xl, yt)).copyTo(this->B());
    left(cv::Rect(0, yt, xl, yb - yt)).copyTo(this->L());
    right(cv::Rect(0, yt, xl, yb - yt)).copyTo(this->R());

    this->merge(front(cv::Rect(0, 0, xl, yt)), left(cv::Rect(0, 0, xl, yt)), 0).copyTo(this->FL());
    this->merge(front(cv::Rect(xr, 0, xl, yt)), right(cv::Rect(0, 0, xl, yt)), 1).copyTo(this->FR());
    this->merge(back(cv::Rect(0, 0, xl, yt)), left(cv::Rect(0, yb, xl, yt)), 2).copyTo(this->BL());
    this->merge(back(cv::Rect(xr, 0, xl, yt)), right(cv::Rect(0, yb, xl, yt)), 3).copyTo(this->BR());
}

void load_car_image(const std::string &data_path)
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
    return;
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
cv::Mat FI(const cv::Mat &img) const { return img(cv::Rect(0, 0, xl, yt)); }
cv::Mat FII(const cv::Mat &img) const { return img(cv::Rect(xr, 0, xl, yt)); }
cv::Mat FM(const cv::Mat &img) const { return img(cv::Rect(xl, 0, xr - xl, yt)); }
cv::Mat BIII(const cv::Mat &img) const { return img(cv::Rect(0, 0, xl, yt)); }
cv::Mat BIV(const cv::Mat &img) const { return img(cv::Rect(xr, 0, xl, yt)); }
cv::Mat BM(const cv::Mat &img) const { return img(cv::Rect(xl, 0, xr - xl, yt)); }
cv::Mat LI(const cv::Mat &img) const { return img(cv::Rect(0, 0, xl, yt)); }
cv::Mat LIII(const cv::Mat &img) const { return img(cv::Rect(0, yb, xl, yt)); }
cv::Mat LM(const cv::Mat &img) const { return img(cv::Rect(0, yt, xl, yb - yt)); }
cv::Mat RII(const cv::Mat &img) const { return img(cv::Rect(0, 0, xl, yt)); }
cv::Mat RIV(const cv::Mat &img) const { return img(cv::Rect(0, yb, xl, yt)); }
cv::Mat RM(const cv::Mat &img) const { return img(cv::Rect(0, yt, xl, yb - yt)); }