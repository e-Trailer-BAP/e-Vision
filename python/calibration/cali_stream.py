import numpy as np
import cv2

def undistort_frame(frame):
    # Define calibration parameters
    DIM=(640, 480)
    K = np.array([[3.1806314102579597e+02, 0.0, 2.9736019394763400e+02],
                  [0.0, 3.2607691921210949e+02, 2.6130699784181263e+02],
                  [0.0, 0.0, 1.0]])
    D = np.array([[4.1891934256866173e-02],
                  [8.4793721172734135e-02],
                  [-3.3489960182649564e-01],
                  [2.5174986499066182e-01]])

    # Undistort the frame
    h, w = frame.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted_frame

def process_stream():
    # Open the OBS Virtual Webcam (usually at index 1, adjust if necessary)
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video stream from OBS Virtual Webcam")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame is empty, break immediately
        if not ret:
            print("Error: Failed to capture image")
            break

        # Undistort the frame
        undistorted_frame = undistort_frame(frame)

        # Display the resulting frame
        cv2.imshow("OBS Virtual Webcam", undistorted_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Run the process_stream function
process_stream()
