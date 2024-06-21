import cv2
import time
import os

# Define frame size and frame rate
frame_width = 1280
frame_height = 960
frame_rate = 50.0

# Define output path
output_path = 'data/prototype/video'

# Ensure the output path exists
os.makedirs(output_path, exist_ok=True)

# Define video capture objects for four video streams
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(3)
cap4 = cv2.VideoCapture(4)

# Set the frame size and frame rate for each capture object
caps = [cap1, cap2, cap3, cap4]
for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)

# Define the codec and create VideoWriter objects for each stream
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
out1 = cv2.VideoWriter(os.path.join(output_path, 'iii-1.mp4'), fourcc, frame_rate, (frame_width, frame_height))
out2 = cv2.VideoWriter(os.path.join(output_path, 'iii-2.mp4'), fourcc, frame_rate, (frame_width, frame_height))
out3 = cv2.VideoWriter(os.path.join(output_path, 'iii-3.mp4'), fourcc, frame_rate, (frame_width, frame_height))
out4 = cv2.VideoWriter(os.path.join(output_path, 'iii-4.mp4'), fourcc, frame_rate, (frame_width, frame_height))

# Start time for recording duration
start_time = time.time()

while True:
    # Capture frame-by-frame from each video source
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    # Check if frames are captured correctly
    if not (ret1 and ret2 and ret3 and ret4):
        print("Failed to capture frames from all sources")
        break

    # Write the frames to their respective files
    out1.write(frame1)
    out2.write(frame2)
    out3.write(frame3)
    out4.write(frame4)

    # Display the frames (optional)
    cv2.imshow('Frame 1', frame1)
    cv2.imshow('Frame 2', frame2)
    cv2.imshow('Frame 3', frame3)
    cv2.imshow('Frame 4', frame4)

    # Exit if 'q' key is pressed or recording time exceeds 2 minutes
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time.time() - start_time > 120:  # 2 minutes = 120 seconds
        print("Recording reached 2 minutes, stopping...")
        break

# Release everything if job is finished
cap1.release()
cap2.release()
cap3.release()
cap4.release()
out1.release()
out2.release()
out3.release()
out4.release()
cv2.destroyAllWindows()
