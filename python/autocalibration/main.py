import utils
import cv2

def process_stream():
    # Open the OBS Virtual Webcam (usually at index 1, adjust if necessary)
    cap = cv2.VideoCapture(1)

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
        centers, image = utils.find_solid_pink_polygons(frame, 20,20,50,20,20,40)
        print(centers)
        # Display the resulting frame
        cv2.imshow("OBS Virtual Webcam", image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break
        else:
            continue

    # When everything is done, release the capture
    cap.release()

    # Close all OpenCV windows
    # cv2.destroyAllWindows()
if __name__ == "__main__":
    process_stream()