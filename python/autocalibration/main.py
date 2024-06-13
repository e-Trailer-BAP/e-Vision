import cv2
import numpy as np

def find_pink_squares(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("The image file was not found.")
    
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_pink=np.array([130, 255, 255])
    upper_pink=np.array([170, 255, 255])
    
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
        if len(approx) >= 4:
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

def find_solid_pink_polygons(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("The image file was not found.")
    
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range of solid pink color in HSV
    lower_pink=np.array([130, 255, 255])
    upper_pink=np.array([170, 255, 255])
    
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
                
                # Draw the centroid
                cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)  # Green for centroids
    
    # Display the image with the contours and centroids marked
    cv2.imshow("Image with Contours and Centroids", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return centroids

# Example usage:
image_path = 'C:/Users/Infer/Documents/Git/BAP/e-Vision/data/images/square_detection-3.png'
centers = find_pink_squares(image_path)
# centers = find_solid_pink_polygons(image_path)
print("Centroids of the squares:", centers)
