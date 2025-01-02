import cv2
import numpy as np

class EdgeVertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Blob:
    def __init__(self):
        self.x_min = 0
        self.y_min = 0
        self.w = 0
        self.h = 0
        self.edges = []
    
    def get_edge_nb(self):
        return len(self.edges)
    
    def get_edge_vertex_a(self, index):
        return self.edges[index][0]
    
    def get_edge_vertex_b(self, index):
        return self.edges[index][1]

class BlobDetection:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.blobs = []
        self.threshold = 0.5
        self.pos_discrimination = True
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def set_pos_discrimination(self, pos_discrimination):
        self.pos_discrimination = pos_discrimination
    
    def get_blob_nb(self):
        return len(self.blobs)
    
    def get_blob(self, index):
        return self.blobs[index] if index < len(self.blobs) else None
    
    def compute_blobs(self, img):
        # Convert to binary image
        _, binary = cv2.threshold(img, int(self.threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Clear previous blobs
        self.blobs = []
        
        # Process each contour
        for contour in contours:
            blob = Blob()
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            blob.x_min = x / self.width
            blob.y_min = y / self.height
            blob.w = w / self.width
            blob.h = h / self.height
            
            # Create edges from contour
            for i in range(len(contour) - 1):
                pt1 = contour[i][0]
                pt2 = contour[i + 1][0]
                edge_a = EdgeVertex(pt1[0] / self.width, pt1[1] / self.height)
                edge_b = EdgeVertex(pt2[0] / self.width, pt2[1] / self.height)
                blob.edges.append((edge_a, edge_b))
            
            # Connect last point to first point
            pt1 = contour[-1][0]
            pt2 = contour[0][0]
            edge_a = EdgeVertex(pt1[0] / self.width, pt1[1] / self.height)
            edge_b = EdgeVertex(pt2[0] / self.width, pt2[1] / self.height)
            blob.edges.append((edge_a, edge_b))
            
            self.blobs.append(blob) 