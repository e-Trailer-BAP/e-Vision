#ifndef BIRDVIEW_HPP
#define BIRDVIEW_HPP

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class BirdView
{
public:
    BirdView();
    void update_frames(const vector<Mat> &images);
    Mat merge(const Mat &imA, const Mat &imB, int k);
    Mat FL() { return this->image(Rect(0, 0, xl, yt)); }
    Mat F() { return this->image(Rect(xl, 0, xr - xl, yt)); }
    Mat FR() { return this->image(Rect(xr, 0, xl, yt)); }
    Mat BL() { return this->image(Rect(0, yb, xl, yt)); }
    Mat B() { return this->image(Rect(xl, yb, xr - xl, yt)); }
    Mat BR() { return this->image(Rect(xr, yb, xl, yt)); }
    Mat L() { return this->image(Rect(0, yt, xl, yb - yt)); }
    Mat R() { return this->image(Rect(xr, yt, xl, yb - yt)); }
    Mat C() { return this->image(Rect(xl, yt, xr - xl, yb - yt)); }

    void stitch_all_parts();
    void load_car_image(const string &data_path);
    void copy_car_image();
    void setParams(map<string, Size> project_shapes);
    void make_luminance_balance();
    tuple<Mat, Mat> get_weights_and_masks(const vector<Mat> &images);
    void make_white_balance();
    Mat getImage() const;

private:
    Mat image;
    Mat car_image;
    vector<Mat> frames;
    vector<Mat> weights;
    vector<Mat> masks;
    int total_w;
    int total_h;
    int xl;
    int xr;
    int yt;
    int yb;
    Mat FI(const Mat &img) const;
    Mat FII(const Mat &img) const;
    Mat FM(const Mat &img) const;
    Mat BIII(const Mat &img) const;
    Mat BIV(const Mat &img) const;
    Mat BM(const Mat &img) const;
    Mat LI(const Mat &img) const;
    Mat LIII(const Mat &img) const;
    Mat LM(const Mat &img) const;
    Mat RII(const Mat &img) const;
    Mat RIV(const Mat &img) const;
    Mat RM(const Mat &img) const;
};

#endif // BIRDVIEW_HPP
