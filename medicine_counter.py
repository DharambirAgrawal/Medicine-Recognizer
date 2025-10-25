#!/usr/bin/env python3
"""
Medicine Counter

A class for counting medicines in real-time video feed using object detection.
Can continuously count and track medicine objects in the camera view.
"""

import cv2
import numpy as np
from collections import deque, defaultdict
import time


class MedicineCounter:
    """
    Real-time medicine counter using object detection and tracking.
    Counts the number of medicine objects in the camera view continuously.
    """
    
    def __init__(self, min_area=500, max_area=50000, detection_threshold=0.6):
        """
        Initialize the Medicine Counter.
        
        Args:
            min_area (int): Minimum contour area to consider as medicine
            max_area (int): Maximum contour area to consider as medicine
            detection_threshold (float): Confidence threshold for detection (0.0-1.0)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.detection_threshold = detection_threshold
        
        # Tracking
        self.medicine_count = 0
        self.count_history = deque(maxlen=30)  # Last 30 frames for smoothing
        self.detected_objects = []
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # For color-based detection
        self.color_ranges = {
            'pills': [
                # White/light colored pills
                ([0, 0, 150], [180, 80, 255]),
                # Orange pills
                ([5, 100, 100], [25, 255, 255]),
                # Blue pills
                ([90, 50, 50], [130, 255, 255]),
            ]
        }
        
        # Statistics
        self.total_counted = 0
        self.counting_active = False
        self.last_count_time = time.time()
        
        print("Medicine Counter initialized")
    
    def reset_count(self):
        """Reset the medicine count."""
        self.medicine_count = 0
        self.count_history.clear()
        self.detected_objects.clear()
        self.total_counted = 0
        print("Counter reset")
    
    def count_medicines(self, frame):
        """
        Count medicines in the given frame.
        
        Args:
            frame: Input frame from camera (BGR format)
            
        Returns:
            dict: Dictionary containing:
                - count: Number of medicines detected
                - annotated_frame: Frame with detection visualization
                - detections: List of detected object bounding boxes
                - confidence: Average detection confidence
        """
        if frame is None or frame.size == 0:
            return {
                'count': 0,
                'annotated_frame': None,
                'detections': [],
                'confidence': 0.0
            }
        
        # Create a copy for annotation
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Method 1: Contour-based detection
        detections_contour = self._detect_by_contours(frame)
        
        # Method 2: Color-based detection
        detections_color = self._detect_by_color(frame)
        
        # Method 3: Blob detection
        detections_blob = self._detect_by_blobs(frame)
        
        # Combine and filter detections
        all_detections = detections_contour + detections_color + detections_blob
        filtered_detections = self._filter_overlapping_detections(all_detections)
        
        # Update count
        current_count = len(filtered_detections)
        self.count_history.append(current_count)
        
        # Smooth the count (use median to avoid jitter)
        if len(self.count_history) > 0:
            smoothed_count = int(np.median(list(self.count_history)))
        else:
            smoothed_count = current_count
        
        self.medicine_count = smoothed_count
        self.detected_objects = filtered_detections
        
        # Calculate average confidence
        avg_confidence = np.mean([d['confidence'] for d in filtered_detections]) if filtered_detections else 0.0
        
        # Annotate frame
        annotated_frame = self._annotate_frame(annotated_frame, filtered_detections, smoothed_count)
        
        return {
            'count': smoothed_count,
            'annotated_frame': annotated_frame,
            'detections': filtered_detections,
            'confidence': avg_confidence
        }
    
    def _detect_by_contours(self, frame):
        """Detect medicine objects using contour analysis."""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (pills are usually circular or rectangular)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 3.5:
                continue
            
            # Calculate circularity (pills tend to be circular)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Calculate confidence based on features
            confidence = min(1.0, (circularity + 0.5) / 1.5)
            
            if confidence >= self.detection_threshold:
                detections.append({
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'area': area,
                    'confidence': confidence,
                    'method': 'contour'
                })
        
        return detections
    
    def _detect_by_color(self, frame):
        """Detect medicine objects using color segmentation."""
        detections = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create combined mask for all pill colors
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.color_ranges['pills']:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence (color detection is quite reliable)
            confidence = 0.8
            
            detections.append({
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'area': area,
                'confidence': confidence,
                'method': 'color'
            })
        
        return detections
    
    def _detect_by_blobs(self, frame):
        """Detect medicine objects using blob detection."""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area
        params.filterByArea = True
        params.minArea = self.min_area
        params.maxArea = self.max_area
        
        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.7
        
        # Filter by inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.3
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            w = h = size
            
            # Adjust to bounding box
            x1 = max(0, x - w//2)
            y1 = max(0, y - h//2)
            
            detections.append({
                'bbox': (x1, y1, w, h),
                'center': (x, y),
                'area': size * size,
                'confidence': 0.85,
                'method': 'blob'
            })
        
        return detections
    
    def _filter_overlapping_detections(self, detections):
        """Filter out overlapping detections using Non-Maximum Suppression."""
        if not detections:
            return []
        
        # Convert to format for NMS
        boxes = []
        scores = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(det['confidence'])
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=self.detection_threshold,
            nms_threshold=0.4
        )
        
        # Filter detections
        filtered = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                filtered.append(detections[i])
        
        return filtered
    
    def _annotate_frame(self, frame, detections, count):
        """Annotate frame with detection results."""
        h, w = frame.shape[:2]
        
        # Draw detection boxes
        for i, det in enumerate(detections):
            x, y, w_box, h_box = det['bbox']
            confidence = det['confidence']
            method = det['method']
            
            # Choose color based on method
            if method == 'contour':
                color = (0, 255, 0)  # Green
            elif method == 'color':
                color = (255, 0, 0)  # Blue
            else:  # blob
                color = (0, 165, 255)  # Orange
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            
            # Draw label
            label = f"#{i+1} ({confidence:.0%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                frame,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
            # Draw center point
            cx, cy = det['center']
            cv2.circle(frame, (cx, cy), 4, color, -1)
        
        # Draw count banner at top
        banner_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Count text
        count_text = f"Medicine Count: {count}"
        font_scale = 1.5
        thickness = 3
        text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        text_x = (w - text_size[0]) // 2
        text_y = 50
        
        cv2.putText(
            frame,
            count_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (0, 255, 0),
            thickness
        )
        
        # Status text
        status_text = f"Detections: {len(detections)} | Active"
        cv2.putText(
            frame,
            status_text,
            (10, banner_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )
        
        # Instructions at bottom
        instruction_text = "Place medicines in view to count"
        cv2.putText(
            frame,
            instruction_text,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return frame
    
    def get_statistics(self):
        """Get counting statistics."""
        return {
            'current_count': self.medicine_count,
            'total_counted': self.total_counted,
            'detected_objects': len(self.detected_objects),
            'is_active': self.counting_active
        }
    
    def start_counting(self):
        """Start the counting process."""
        self.counting_active = True
        print("Counting started")
    
    def stop_counting(self):
        """Stop the counting process."""
        self.counting_active = False
        print("Counting stopped")


def main():
    """Test the medicine counter with webcam."""
    print("Medicine Counter - Test Mode")
    print("Press 'q' to quit, 'r' to reset count, 's' to show statistics")
    
    # Initialize counter
    counter = MedicineCounter()
    counter.start_counting()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nCamera started. Counting medicines...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Count medicines
        result = counter.count_medicines(frame)
        
        # Display result
        if result['annotated_frame'] is not None:
            cv2.imshow('Medicine Counter', result['annotated_frame'])
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.reset_count()
            print("\nCount reset!")
        elif key == ord('s'):
            stats = counter.get_statistics()
            print("\n--- Statistics ---")
            print(f"Current Count: {stats['current_count']}")
            print(f"Total Counted: {stats['total_counted']}")
            print(f"Detected Objects: {stats['detected_objects']}")
            print(f"Status: {'Active' if stats['is_active'] else 'Inactive'}")
            print("------------------")
    
    # Cleanup
    counter.stop_counting()
    cap.release()
    cv2.destroyAllWindows()
    print("\nCounter stopped. Goodbye!")


if __name__ == "__main__":
    main()
