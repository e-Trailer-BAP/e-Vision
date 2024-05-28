#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open the OBS Virtual Webcam (usually at index 1, adjust if necessary)
    cv::VideoCapture cap(3);

    // Check if the webcam is opened correctly
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream from OBS Virtual Webcam" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty()) {
            std::cerr << "Error: Failed to capture image" << std::endl;
            break;
        }

        // Display the resulting frame
        cv::imshow("OBS Virtual Webcam", frame);

        // Break the loop on 'q' key press
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // When everything is done, release the capture
    cap.release();

    // Close all OpenCV windows
    cv::destroyAllWindows();

    return 0;
}

