import cv2
import numpy as np
import os 

def generate_output_path(input_path, suffix):
    # Split the input path into components
    path_parts = input_path.split(os.sep)
    file_name = path_parts[-1]
    
    # Extract file name without extension and the extension
    file_base_name, file_extension = os.path.splitext(file_name)
    
    # Construct the new file name
    new_file_name = f"{file_base_name}-{suffix}{file_extension}"
    
    return new_file_name


def undistort(image, cam):
    if cam == 'pi1':
        K = np.array([[6.4072733351644865e+02, 0.0, 6.2202256509683798e+02],
                    [0.0, 5.9651279438297843e+02, 4.7118575007151452e+02], 
                    [0.0, 0.0, 1.0]])

        D = np.array([[-2.6324817536674022e-03],
                    [-1.0976012529572739e-01],
                    [2.8343617874044086e-01],
                    [-2.2794885503769136e-01]])
    elif cam == 'pi2':
        K = np.array([[6.5749994037647730e+02, 0.0, 6.2419695408122357e+02],
                    [0.0, 6.1708962360492478e+02, 4.7881887927969512e+02], 
                    [0.0, 0.0, 1.0]])

        D = np.array([[-4.3197926701982574e-03],
                    [-8.1253917657262059e-02],
                    [1.6049401021107843e-01],
                    [-1.0204227119642488e-01]])
    elif cam == 'pi3':
        K = np.array([[6.5543949692786850e+02, 0.0, 6.7480181103111045e+02],
            [0.0, 6.0948650232925490e+02, 5.0131595321689986e+02], 
            [0.0, 0.0, 1.0]])

        D = np.array([[-3.9239424430358966e-02],
                    [6.8904791288211240e-02],
                    [-1.1847369719144193e-01],
                    [6.3224077225049163e-02]])
    elif cam == 'pi4':
        K = np.array([[6.4929051801406797e+02, 0.0, 6.1230522039856157e+02],
                [0.0, 6.0556935671870428e+02, 5.0560432887567748e+02], 
                [0.0, 0.0, 1.0]])

        D = np.array([[5.3835887857414068e-03],
                    [-1.6313573556439998e-01],
                    [3.9320691822960374e-01],
                    [-3.0628063565889174e-01]])
        

    scale_xy = (1, 1)
    shift_xy = (0, 0)
    if cam == "pi4":
        scale_xy = (1, 1)
        shift_xy = (0, 150)
    new_matrix = K.copy()
    new_matrix[0, 0] *= scale_xy[0]
    new_matrix[1, 1] *= scale_xy[1]
    new_matrix[0, 2] += shift_xy[0]
    new_matrix[1, 2] += shift_xy[1]

    # Undistort the frame
    h, w = image.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_matrix, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted


# Define the projection matrix
front_matrix = np.array([[ 3.95755148e-01, 3.26879287e-01, 2.50089679e+02],
                              [-4.72947346e-02, 7.60778605e-01, 1.13441096e+02],
                              [-9.47868785e-05, 6.68086220e-04, 1.00000000e+00]])
back_matrix =   np.array([[ 6.95258031e-01,  1.24826254e+00,  3.86058649e+01],
                         [-6.26353104e-02, 1.93544660e+00, -1.86710803e+02],
                         [-2.79806698e-04,  2.56496753e-03,  1.00000000e+00]])
left_matrix = np.array([[ 9.63948487e-01,  1.75181228e+00, -1.50561072e+02],
                        [-6.67561120e-02,  2.54388246e+00, -3.76566406e+02],
                        [-1.98156138e-04,  3.49163457e-03,  1.00000000e+00]])
right_matrix = np.array([[ 9.21375240e-01,  1.50527043e+00, -1.61969383e+02],
                         [-3.28698973e-02,  2.21622361e+00, -2.69334567e+02],
                         [-7.34255427e-05,  3.00736345e-03,  1.00000000e+00]])

for dir in ['front', 'back', 'left', 'right']:
        

    if dir=='front':
        projection_matrix=front_matrix
        pi = 'pi4'
        record_num=3
    elif dir=='back':
        projection_matrix=back_matrix
        pi = 'pi1'
        record_num=1
    elif dir=='right':
        projection_matrix=right_matrix
        pi = 'pi3'
        record_num=4
    elif dir=='left':
        projection_matrix=left_matrix
        pi = 'pi2'
        record_num=2

    # Load the video file
    working_dir = 'data/prototype/testing/'
    input_video_path = working_dir+f't-{record_num}.mp4'
    output_video_path = generate_output_path(input_video_path, 'output')

    cap = cv2.VideoCapture(input_video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the desired output frame size
    output_frame_width = 1000
    output_frame_height = 500
    output_frame_size = (output_frame_width, output_frame_height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, output_frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistort_frame=undistort(frame, pi)
        # Apply the projection matrix to the frame
        transformed_frame = cv2.warpPerspective(undistort_frame, projection_matrix, output_frame_size)

        # Write the transformed frame to the output video
        out.write(transformed_frame)

        # Display the frame (optional)
        # cv2.imshow('Transformed Frame', transformed_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release everything if job is finished
    cap.release()   
    out.release()
    cv2.destroyAllWindows()
