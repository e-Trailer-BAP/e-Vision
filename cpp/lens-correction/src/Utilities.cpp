#include "Utilities.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

std::map<std::string, cv::Size> init_constants()
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

    return project_shapes;
}

cv::Mat convertBinaryToBool(const cv::Mat &mask)
{
    cv::Mat boolMask;
    mask.convertTo(boolMask, CV_32F, 1.0 / 255.0);
    return boolMask;
}

cv::Mat adjustLuminance(const cv::Mat &gray, double factor)
{
    cv::Mat adjusted;
    gray.convertTo(adjusted, -1, factor);
    return adjusted;
}

double getMeanStatistic(const cv::Mat &gray, const cv::Mat &mask)
{
    cv::Scalar sum = cv::sum(gray.mul(mask));
    return sum[0];
}

double meanLuminanceRatio(const cv::Mat &grayA, const cv::Mat &grayB, const cv::Mat &mask)
{
    return getMeanStatistic(grayA, mask) / getMeanStatistic(grayB, mask);
}

cv::Mat getMask(const cv::Mat &img)
{
    cv::Mat gray, mask;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY);
    return mask;
}

cv::Mat getOverlapRegionMask(const cv::Mat &imA, const cv::Mat &imB)
{
    cv::Mat overlap, mask;
    cv::bitwise_and(imA, imB, overlap);
    mask = getMask(overlap);
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)), cv::Point(-1, -1), 2);
    return mask;
}

std::vector<Point> getOutmostPolygonBoundary(const cv::Mat &img)
{
    cv::Mat mask = getMask(img);
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)), cv::Point(-1, -1), 2);
    vector<vector<Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Point> largestContour = *max_element(contours.begin(), contours.end(), [](const std::vector<Point> &a, const std::vector<Point> &b)
                                                     { return cv::contourArea(a) < cv::contourArea(b); });

    std::vector<Point> polygon;
    cv::approxPolyDP(largestContour, polygon, 0.009 * cv::arcLength(largestContour, true), true);

    return polygon;
}

pair<cv::Mat, cv::Mat> getWeightMaskMatrix(const cv::Mat &imA, const cv::Mat &imB, double distThreshold)
{
    cv::Mat overlapMask = getOverlapRegionMask(imA, imB);
    cv::Mat overlapMaskInv;
    cv::bitwise_not(overlapMask, overlapMaskInv);

    cv::Mat imA_diff, imB_diff;
    cv::bitwise_and(imA, imA, imA_diff, overlapMaskInv);
    cv::bitwise_and(imB, imB, imB_diff, overlapMaskInv);

    cv::Mat G = getMask(imA);
    G.convertTo(G, CV_32F, 1.0 / 255.0);

    std::vector<Point> polyA = getOutmostPolygonBoundary(imA_diff);
    std::vector<Point> polyB = getOutmostPolygonBoundary(imB_diff);

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

cv::Mat makeWhiteBalance(const cv::Mat &image)
{
    vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Scalar means = cv::mean(image);
    double K = (means[0] + means[1] + means[2]) / 3.0;

    channels[0] = adjustLuminance(channels[0], K / means[0]);
    channels[1] = adjustLuminance(channels[1], K / means[1]);
    channels[2] = adjustLuminance(channels[2], K / means[2]);

    cv::Mat balancedImage;
    cv::merge(channels, balancedImage);
    return balancedImage;
}
