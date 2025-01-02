# Simple blob detection in Python
# requires the file blob_detection.py
# pip install opencv-python
# pip install numpy
# Kevin Walker 02 Jan 2025
# ported from an old Processing sketch: https://github.com/increasinglyunclear/processing/blob/main/blobs_stars

import cv2
import numpy as np
from blob_detection import BlobDetection

def create_test_image(width, height):
    print("Creating test image...")
    # Create black background
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Draw 20 random white circles
    for i in range(20):
        center = (
            np.random.randint(0, width),
            np.random.randint(0, height)
        )
        radius = np.random.randint(5, 25)
        cv2.circle(img, center, radius, 255, -1)
        print(f"Drew circle {i+1} at {center} with radius {radius}")
    
    return img

def main():
    try:
        print("Starting blob detection program...")
        
        # Create image
        width, height = 640, 480
        img = create_test_image(width, height)
        print(f"Created image with shape: {img.shape}")
        
        # Initialize blob detection
        detector = BlobDetection(width, height)
        detector.set_pos_discrimination(False)
        detector.set_threshold(0.38)
        print("Processing image for blob detection...")
        detector.compute_blobs(img)
        
        num_blobs = detector.get_blob_nb()
        print(f"Found {num_blobs} blobs")
        
        # Create color image for visualization
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw blobs and edges
        for n in range(detector.get_blob_nb()):
            blob = detector.get_blob(n)
            if blob is not None:
                # Draw edges in green
                for m in range(blob.get_edge_nb()):
                    eA = blob.get_edge_vertex_a(m)
                    eB = blob.get_edge_vertex_b(m)
                    if eA is not None and eB is not None:
                        pt1 = (int(eA.x * width), int(eA.y * height))
                        pt2 = (int(eB.x * width), int(eB.y * height))
                        cv2.line(display_img, pt1, pt2, (0, 255, 0), 2)
                
                # Draw blob rectangles in red
                x = int(blob.x_min * width)
                y = int(blob.y_min * height)
                w = int(blob.w * width)
                h = int(blob.h * height)
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        print("Displaying result...")
        cv2.imshow('Blob Detection', display_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
