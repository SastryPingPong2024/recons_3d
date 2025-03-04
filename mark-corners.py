import cv2
import numpy as np

def mark_corners(image_path, threshold=0.01):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect corners using Harris Corner Detection
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, blockSize=2, ksize=17, k=0.04)
    
    # Dilate result for better visibility
    corners = cv2.dilate(corners, None)
    
    # Mark corners that exceed the threshold
    img[corners > threshold * corners.max()] = [0, 0, 255]  # Red color
    
    # Show and save the output image
    # cv2.imshow('Corners', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Optionally, save the image with marked corners
    output_path = 'corners_marked.png'
    cv2.imwrite(output_path, img)
    print(f'Corners marked and saved to {output_path}')

# Example usage
mark_corners('images/img1.png', threshold=0.003)