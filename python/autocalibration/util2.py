import cv2
import numpy as np
import os

def find_pink_squares(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("The image file was not found.")
    
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    H=150
    R=50
    lower_pink=np.array([H-R, 255, 255])
    upper_pink=np.array([H+R, 255, 255])
    
    # Create a mask for pink color
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # We expect the approximated polygon to have 4 points (corners of the square)
        if len(approx) == 4:
            # Draw the contour (for visualization purposes)
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            
            
            # Calculate the centroid of the square
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
                
                # Draw the centroid
                cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)
    
    # Display the image with the centroids marked
    cv2.imshow("Image with Centroids", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return centroids

def find_solid_pink_polygons(imagepath, cam, HRL, SRL, VRL, HRU, SRU, VRU):
    # Load the image
    image = cv2.imread(imagepath)
    if image is None:
        raise FileNotFoundError("The image file was not found.")
    imagenew = undistort(image,imagepath, cam)
    
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(imagenew, cv2.COLOR_BGR2HSV)
    
    H=150
    S=220
    V=220
    # HR=30
    # SR=10
    # VR=30
    lower_pink=np.array([H-HRL, S-SRL, V-VRL])
    upper_pink=np.array([H+HRU, S+SRU, V+VRU])
    
    # Create a mask for pink color
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the polygon has at least 3 points
        if len(approx) >= 3:
            # Draw the contour in orange (for visualization purposes)
            cv2.drawContours(imagenew, [approx], -1, (0, 255, 0), 2)
            
            # Calculate the centroid of the polygon
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
                print("Centroid:", cX, cY)
                # Draw the centroid
                cv2.circle(imagenew, (cX, cY), 5, (0, 255, 0), -1)  # Green for centroids
    
    # Display the image with the contours and centroids marked
    cv2.imshow("Image with Contours and Centroids", imagenew)
    cv2.imwrite(generate_output_path(imagepath,"detection"), imagenew)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return centroids, image

# # Example usage:
# image_path = 'data/images/square_detection-3.png'
# centers = find_pink_squares(image_path)
# # centers = find_solid_pink_polygons(image_path)
# print("Centroids of the squares:", centers)
def undistort(image, path, cam):
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
    new_matrix = K.copy()
    new_matrix[0, 0] *= scale_xy[0]
    new_matrix[1, 1] *= scale_xy[1]
    new_matrix[0, 2] += shift_xy[0]
    new_matrix[1, 2] += shift_xy[1]

    # Undistort the frame
    h, w = image.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_matrix, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(generate_output_path(path, "undistort"), undistorted)
    return undistorted

def generate_output_path(input_path, suffix):
    # Split the input path into components
    path_parts = input_path.split(os.sep)
    file_name = path_parts[-1]
    
    # Extract file name without extension and the extension
    file_base_name, file_extension = os.path.splitext(file_name)
    
    # Construct the new file name
    new_file_name = f"{file_base_name}-{suffix}{file_extension}"
    
    return new_file_name


def hsvrgb(R, G, B):
    # Define the RGB color code
    # R, G, B = rgb_color_code
    
    # Create a NumPy array with the RGB color code
    rgb = np.uint8([[[R, G, B]]])
    
    # Convert the RGB color code to the HSV color space
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    
    # Extract the HSV components
    H, S, V = hsv[0][0]
    print("HSV values({}, {}, {}):".format(H, S, V))
    return H, S, V

hsvrgb(219, 30, 218)
hsvrgb(226, 136, 234)
hsvrgb(228, 131, 234)
# find_pink_squares("C:/Users/Infer/Documents/TU Delft/BAP/e-Vision/data/images/test.jpg")
find_solid_pink_polygons('data/prototype/front.png', "pi4", 30, 160, 20, 12, 40, 40)
# generate_output_path('data/prototype/back.png')
