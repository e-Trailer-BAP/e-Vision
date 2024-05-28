#include "Utilities.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

map<string, Size> init_constants()
{
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

    map<string, Size> project_shapes = {
        {"front", Size(total_w, yt)},
        {"back", Size(total_w, yt)},
        {"left", Size(total_h, xl)},
        {"right", Size(total_h, xl)}};

    map<string, vector<Point>> project_keypoints = {
        {"front", {Point(shift_w + 120, shift_h), Point(shift_w + 480, shift_h), Point(shift_w + 120, shift_h + 160), Point(shift_w + 480, shift_h + 160)}},

        {"back", {Point(shift_w + 120, shift_h), Point(shift_w + 480, shift_h), Point(shift_w + 120, shift_h + 160), Point(shift_w + 480, shift_h + 160)}},

        {"left", {Point(shift_h + 280, shift_w), Point(shift_h + 840, shift_w), Point(shift_h + 280, shift_w + 160), Point(shift_h + 840, shift_w + 160)}},

        {"right", {Point(shift_h + 160, shift_w), Point(shift_h + 720, shift_w), Point(shift_h + 160, shift_w + 160), Point(shift_h + 720, shift_w + 160)}}};

    return project_shapes;
}

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
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, mask, 0, 255, THRESH_BINARY);
    return mask;
}

Mat getOverlapRegionMask(const Mat &imA, const Mat &imB)
{
    Mat overlap, mask;
    bitwise_and(imA, imB, overlap);
    mask = getMask(overlap);
    dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(2, 2)), Point(-1, -1), 2);
    return mask;
}

vector<Point> getOutmostPolygonBoundary(const Mat &img)
{
    Mat mask = getMask(img);
    dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(2, 2)), Point(-1, -1), 2);
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Point> largestContour = *max_element(contours.begin(), contours.end(), [](const vector<Point> &a, const vector<Point> &b)
                                                { return contourArea(a) < contourArea(b); });

    vector<Point> polygon;
    approxPolyDP(largestContour, polygon, 0.009 * arcLength(largestContour, true), true);

    return polygon;
}

pair<Mat, Mat> getWeightMaskMatrix(const Mat &imA, const Mat &imB, double distThreshold)
{
    Mat overlapMask = getOverlapRegionMask(imA, imB);
    Mat overlapMaskInv;
    bitwise_not(overlapMask, overlapMaskInv);

    Mat imA_diff, imB_diff;
    bitwise_and(imA, imA, imA_diff, overlapMaskInv);
    bitwise_and(imB, imB, imB_diff, overlapMaskInv);

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
                double distToB = pointPolygonTest(polyB, pt, true);
                if (distToB < distThreshold)
                {
                    double distToA = pointPolygonTest(polyA, pt, true);
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
    vector<cv::Mat> channels;
    split(image, channels);
    Scalar means = mean(image);
    double K = (means[0] + means[1] + means[2]) / 3.0;

    channels[0] = adjustLuminance(channels[0], K / means[0]);
    channels[1] = adjustLuminance(channels[1], K / means[1]);
    channels[2] = adjustLuminance(channels[2], K / means[2]);

    Mat balancedImage;
    merge(channels, balancedImage);
    return balancedImage;
}
