import cv2
import numpy as np

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

def find_solid_pink_polygons(image, HRL, SRL, VRL, HRU, SRU, VRU):
    # Load the image
    # image = cv2.imread(imagepath)
    # if image is None:
    #     raise FileNotFoundError("The image file was not found.")
    
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    H=150
    S=144
    V=229
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
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            
            # Calculate the centroid of the polygon
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
                print("Centroid:", cX, cY)
                # Draw the centroid
                cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)  # Green for centroids
    
    # Display the image with the contours and centroids marked
    # cv2.imshow("Image with Contours and Centroids", image)
    # cv2.imwrite("C:/Users/Infer/Documents/TU Delft/BAP/e-Vision/data/images/cv-final.png", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return centroids, image

# # Example usage:
# image_path = 'data/images/square_detection-3.png'
# centers = find_pink_squares(image_path)
# # centers = find_solid_pink_polygons(image_path)
# print("Centroids of the squares:", centers)

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

hsvrgb(220, 33, 216)
# # find_pink_squares("C:/Users/Infer/Documents/TU Delft/BAP/e-Vision/data/images/test.jpg")
# find_solid_pink_polygons('../../data/images/undistort-final.png', 00, 60, 90, 50, 00, 40)
