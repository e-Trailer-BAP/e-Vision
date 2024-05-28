#include "FisheyeCameraModel.hpp"
#include "BirdView.hpp"
#include "Utilities.hpp"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <filesystem>

using namespace cv;
using namespace std;

void process_image(const std::vector<std::string> &images, const std::string &output_path, const std::vector<FisheyeCameraModel> &camera_models)
{
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

    // Save image
    cv::imwrite(output_path + "/birdview.png", birdview.getImage());
}

void process_video(const std::vector<std::string> &input_videos, const std::string &output_path, const std::vector<FisheyeCameraModel> &camera_models)
{
    // Open video files
    std::vector<cv::VideoCapture> caps;
    for (const auto &video : input_videos)
    {
        caps.emplace_back(video);
    }
    if (!std::all_of(caps.begin(), caps.end(), [](cv::VideoCapture &cap)
                     { return cap.isOpened(); }))
    {
        std::cerr << "Error: One or more video files couldn't be opened." << std::endl;
        return;
    }

    // Get video properties
    int width = static_cast<int>(caps[0].get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(caps[0].get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(caps[0].get(cv::CAP_PROP_FPS));
    int frame_count = static_cast<int>(caps[0].get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << frame_count << std::endl;

    // Define the codec and create VideoWriter object
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter out(output_path + "/birds_eye_view.mp4", fourcc, fps, cv::Size(1200, 1600));

    BirdView birdview;

    for (int i = 0; i < frame_count; ++i)
    {
        std::vector<cv::Mat> frames;
        for (auto &cap : caps)
        {
            cv::Mat frame;
            if (!cap.read(frame))
            {
                std::cerr << "Error: Couldn't read frame " << i << "." << std::endl;
                break;
            }
            frames.push_back(frame);
        }

        if (frames.size() != camera_models.size())
        {
            std::cerr << "Error: Number of frames doesn't match number of camera models." << std::endl;
            break;
        }

        // Process each frame
        std::vector<cv::Mat> projected;
        for (size_t j = 0; j < frames.size(); ++j)
        {
            cv::Mat img = camera_models[j].undistort(frames[j]);
            img = camera_models[j].project(img);
            img = camera_models[j].flip(img);
            projected.push_back(img);
        }

        // Stitch frames into a bird's eye view
        if (i == 0)
        {
            cout << "weights" << endl;
            auto [Gmat, Mmat] = birdview.get_weights_and_masks(projected);
        }

        birdview.update_frames(projected);
        birdview.make_luminance_balance();
        birdview.stitch_all_parts();
        birdview.copy_car_image();
        // birdview.make_luminance_balance();
        // birdview.make_white_balance();

        int birdview_height = birdview.getImage().rows;
        int birdview_width = birdview.getImage().cols;

        if ((birdview_height != 1600) || (birdview_width != 1200))
        {
            std::cout << "frame " << i << " is false" << std::endl;
        }
        else
        {
            if (i == 1 || i == 10 || i == 100 || i == 200 || i == frame_count - 1)
            {
                std::cout << "True" << std::endl;
            }
        }

        // Write the frame to the output video
        out.write(birdview.getImage());
    }

    // Release video files and writer
    for (auto &cap : caps)
    {
        cap.release();
    }
    out.release();
    std::cout << "Output video saved to " << output_path + "/birds_eye_view.mp4" << std::endl;
}

int main()
{
    std::string mode = "video";
    // Declare global variables and constants
    std::vector<std::string> camera_names = {"front", "back", "left", "right"};
    std::vector<std::string> yamls;
    std::vector<std::string> images;
    std::vector<std::string> videos;
    std::string data_path = "../../../data";          // Set your data path here
    std::string output_path = "../../../data/output"; // Set your output path here
    // load_car_image(data_path);

    for (const auto &name : camera_names)
    {
        yamls.push_back(data_path + "/yaml/" + name + ".yaml");
        if (mode == "video")
        {
            videos.push_back(data_path + "/videos/" + name + ".mp4");
        }
        else
        {
            images.push_back(data_path + "/images/" + name + ".png");
        }
    }

    std::vector<FisheyeCameraModel> camera_models;
    for (size_t i = 0; i < camera_names.size(); ++i)
    {
        camera_models.emplace_back(yamls[i], camera_names[i]);
    }

    if (mode == "video")
    {
        process_video(videos, output_path, camera_models);
    }
    else
    {
        process_image(images, output_path, camera_models);
    }
    return 0;
}
