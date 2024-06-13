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

void process_image(const vector<string> &images, const string &data_path, const string &output_path, const vector<FisheyeCameraModel> &camera_models)
{
    // Process images using the camera models
    vector<Mat> projected;
    for (size_t i = 0; i < images.size(); ++i)
    {
        Mat img = imread(images[i]);
        if (img.empty())
        {
            throw runtime_error("Could not open or find the image: " + images[i]);
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
    birdview.load_car_image(data_path);
    birdview.copy_car_image();
    birdview.make_white_balance();

    cout << "Bird's eye view stitched image after added white balance" << endl;

    // Display the resulting birdview image
    imshow("BirdView", birdview.getImage());
    waitKey(0);

    // Save image
    imwrite(output_path + "/birdview.png", birdview.getImage());
}

void process_video(const vector<string> &input_videos, const string &data_path, const string &output_path, const vector<FisheyeCameraModel> &camera_models)
{
    // Open video files
    vector<VideoCapture> caps;
    for (const auto &video : input_videos)
    {
        caps.emplace_back(video);
    }
    if (!all_of(caps.begin(), caps.end(), [](VideoCapture &cap)
                { return cap.isOpened(); }))
    {
        cerr << "Error: One or more video files couldn't be opened." << endl;
        return;
    }

    // Get video properties
    int width = static_cast<int>(caps[0].get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(caps[0].get(CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(caps[0].get(CAP_PROP_FPS));
    int frame_count = static_cast<int>(caps[0].get(CAP_PROP_FRAME_COUNT));
    cout << frame_count << endl;

    // Define the codec and create VideoWriter object
    int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
    VideoWriter out(output_path + "/birds_eye_view.mp4", fourcc, fps, Size(1200, 1600));

    BirdView birdview;

    for (int i = 0; i < frame_count; ++i)
    {
        vector<Mat> frames;
        for (auto &cap : caps)
        {
            Mat frame;
            if (!cap.read(frame))
            {
                cerr << "Error: Couldn't read frame " << i << "." << endl;
                break;
            }
            frames.push_back(frame);
        }

        if (frames.size() != camera_models.size())
        {
            cerr << "Error: Number of frames doesn't match number of camera models." << endl;
            break;
        }

        // Process each frame
        vector<Mat> projected;
        for (size_t j = 0; j < frames.size(); ++j)
        {
            Mat img = camera_models[j].undistort(frames[j]);
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
        birdview.load_car_image(data_path);
        birdview.copy_car_image();
        // birdview.make_luminance_balance();
        // birdview.make_white_balance();

        int birdview_height = birdview.getImage().rows;
        int birdview_width = birdview.getImage().cols;

        if ((birdview_height != 1600) || (birdview_width != 1200))
        {
            cout << "frame " << i << " is false" << endl;
        }
        else
        {
            if (i == 1 || i == 10 || i == 100 || i == 200 || i == frame_count - 1)
            {
                cout << "True" << endl;
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
    cout << "Output video saved to " << output_path + "/birds_eye_view.mp4" << endl;
}

void process_stream(const FisheyeCameraModel &camera_model)
{
    // Open the OBS Virtual Webcam (usually at index 1, adjust if necessary)
    cv::VideoCapture cap(3);

    // Check if the webcam is opened correctly
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video stream from OBS Virtual Webcam" << std::endl;
    }

    cv::Mat frame;
    while (true)
    {
        // Capture frame-by-frame
        cap >> frame;

        // cout << "Frame Size" << frame.size() << endl;

        // If the frame is empty, break immediately
        if (frame.empty())
        {
            std::cerr << "Error: Failed to capture image" << std::endl;
            break;
        }

        // Resize the frame to your desired resolution
        cv::Size desired_size(960, 640); // Your desired resolution
        cv::resize(frame, frame, desired_size);

        frame = camera_model.undistort(frame);
        frame = camera_model.project(frame);
        frame = camera_model.flip(frame);

        // Display the resulting frame
        cv::imshow("OBS Virtual Webcam", frame);

        // Break the loop on 'q' key press
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    // When everything is done, release the capture
    cap.release();

    // Close all OpenCV windows
    cv::destroyAllWindows();
}

int main()
{
    string mode = "image";
    vector<string> camera_names = {"front", "back", "left", "right"};
    vector<string> yamls;
    vector<string> images;
    vector<string> videos;
    string data_path = "../../../data";          // Set your data path here
    string output_path = "../../../data/output"; // Set your output path here

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

    vector<FisheyeCameraModel> camera_models;
    for (size_t i = 0; i < camera_names.size(); ++i)
    {
        camera_models.emplace_back(yamls[i], camera_names[i]);
    }

    const FisheyeCameraModel &selected_camera_model = camera_models[0]; // Replace 'index' with the desired index

    if (mode == "video")
    {
        process_video(videos, data_path, output_path, camera_models);
    }
    else if (mode == "stream")
    {
        process_stream(selected_camera_model);
    }
    else
    {
        process_image(images, data_path, output_path, camera_models);
    }
    return 0;
}
