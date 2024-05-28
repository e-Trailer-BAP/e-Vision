#ifndef BIRDVIEW_HPP
#define BIRDVIEW_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class BirdView
{
public:
    BirdView();
    void update_frames(const std::vector<cv::Mat> &images);
    cv::Mat merge(const cv::Mat &imA, const cv::Mat &imB, int k);
    cv::Mat FL() { return this->image(cv::Rect(0, 0, xl, yt)); }
    cv::Mat F() { return this->image(cv::Rect(xl, 0, xr - xl, yt)); }
    cv::Mat FR() { return this->image(cv::Rect(xr, 0, xl, yt)); }
    cv::Mat BL() { return this->image(cv::Rect(0, yb, xl, yt)); }
    cv::Mat B() { return this->image(cv::Rect(xl, yb, xr - xl, yt)); }
    cv::Mat BR() { return this->image(cv::Rect(xr, yb, xl, yt)); }
    cv::Mat L() { return this->image(cv::Rect(0, yt, xl, yb - yt)); }
    cv::Mat R() { return this->image(cv::Rect(xr, yt, xl, yb - yt)); }
    cv::Mat C() { return this->image(cv::Rect(xl, yt, xr - xl, yb - yt)); }

    void stitch_all_parts();
    void load_car_image(const std::string &data_path);
    void copy_car_image();
    void setParams(std::map<std::string, cv::Size> project_shapes);
    void make_luminance_balance();
    std::tuple<cv::Mat, cv::Mat> get_weights_and_masks(const std::vector<cv::Mat> &images);
    void make_white_balance();
    cv::Mat getImage() const;

private:
    cv::Mat image;
    cv::Mat car_image;
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> weights;
    std::vector<cv::Mat> masks;
    int total_w;
    int total_h;
    int xl;
    int xr;
    int yt;
    int yb;
    cv::Mat FI(const cv::Mat &img) const;
    cv::Mat FII(const cv::Mat &img) const;
    cv::Mat FM(const cv::Mat &img) const;
    cv::Mat BIII(const cv::Mat &img) const;
    cv::Mat BIV(const cv::Mat &img) const;
    cv::Mat BM(const cv::Mat &img) const;
    cv::Mat LI(const cv::Mat &img) const;
    cv::Mat LIII(const cv::Mat &img) const;
    cv::Mat LM(const cv::Mat &img) const;
    cv::Mat RII(const cv::Mat &img) const;
    cv::Mat RIV(const cv::Mat &img) const;
    cv::Mat RM(const cv::Mat &img) const;
};

#endif // BIRDVIEW_HPP
