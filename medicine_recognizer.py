# #!/usr/bin/env python3
# """
# Medicine Recognizer

# Advanced implementation using OpenCV DNN, optimized algorithms,
# and efficient object detection for real-time medicine recognition.
# """

# import cv2
# import numpy as np
# import time
# import os
# from datetime import datetime
# from collections import defaultdict
# import threading
# from queue import Queue


# class MedicineRecognizer:
#     """
#     High-performance medicine recognition using optimized OpenCV algorithms.
#     Uses efficient object segmentation and feature matching for real-time performance.
#     """

#     def __init__(self, reference_folder="medicine_references", 
#                  confidence_threshold=0.60,
#                  use_cuda=False):
#         """
#         Initialize the Medicine Recognizer.
        
#         Args:
#             reference_folder (str): Folder containing reference medicine images
#             confidence_threshold (float): Threshold for detection confidence (0.0-1.0)
#             use_cuda (bool): Whether to use CUDA for GPU acceleration
#         """
#         self.reference_folder = reference_folder
#         self.confidence_threshold = confidence_threshold
#         self.medicine_db = {}
        
#         # Use ORB for speed (much faster than SIFT)
#         self.feature_detector = cv2.ORB_create(
#             nfeatures=500,  # Reduced for speed
#             scaleFactor=1.2,
#             nlevels=8,
#             edgeThreshold=15,
#             firstLevel=0,
#             WTA_K=2,
#             scoreType=cv2.ORB_FAST_SCORE,
#             patchSize=31,
#             fastThreshold=20
#         )
        
#         # BFMatcher with Hamming distance for ORB
#         self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
#         # Color histogram settings (reduced for speed)
#         self.hist_bins = [6, 6, 6]  # Reduced bins for faster comparison
#         self.hist_range = [0, 256, 0, 256, 0, 256]
        
#         # Processing queue for async operations
#         self.processing_queue = Queue(maxsize=2)
        
#         # Create reference folder if it doesn't exist
#         if not os.path.exists(reference_folder):
#             os.makedirs(reference_folder)
#             print(f"Created reference folder: {reference_folder}")
        
#         # Load reference medicines
#         self._load_references()
        
#         # Setup GPU acceleration if requested and available
#         if use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0:
#             print("CUDA acceleration enabled")
#             self.use_cuda = True
#         else:
#             if use_cuda:
#                 print("CUDA requested but not available, using CPU instead")
#             self.use_cuda = False
        
#         # Cache for extracted objects to avoid reprocessing
#         self.extraction_cache = {}
#         self.cache_size = 10
    
#     def _load_references(self):
#         """Load reference medicine images and compute their features (optimized)."""
#         print("Loading medicine references...")
        
#         if not os.path.exists(self.reference_folder) or not os.listdir(self.reference_folder):
#             print("No reference images found. Use add_reference() to add medicine references.")
#             return
        
#         for filename in os.listdir(self.reference_folder):
#             if filename.endswith(('.jpg', '.jpeg', '.png')) and 'original' not in filename:
#                 # Extract medicine name from filename
#                 medicine_name = "_".join(filename.split("_")[:-1]) if "_" in filename else os.path.splitext(filename)[0]
#                 image_path = os.path.join(self.reference_folder, filename)
                
#                 # Load image
#                 ref_img = cv2.imread(image_path)
#                 if ref_img is None:
#                     continue
                
#                 # Extract features
#                 features = self._extract_medicine_features(ref_img)
                
#                 if features['descriptors'] is not None:
#                     # Store or update reference data
#                     if medicine_name not in self.medicine_db:
#                         self.medicine_db[medicine_name] = {
#                             'samples': [],
#                             'avg_color_hist': None
#                         }
                    
#                     # Add this sample
#                     self.medicine_db[medicine_name]['samples'].append({
#                         'features': features,
#                         'image': ref_img
#                     })
                    
#                     # Update average color histogram
#                     if self.medicine_db[medicine_name]['avg_color_hist'] is None:
#                         self.medicine_db[medicine_name]['avg_color_hist'] = features['color_hist']
#                     else:
#                         n = len(self.medicine_db[medicine_name]['samples'])
#                         avg = self.medicine_db[medicine_name]['avg_color_hist']
#                         self.medicine_db[medicine_name]['avg_color_hist'] = (avg * (n - 1) + features['color_hist']) / n
                    
#                     kp_count = len(features['keypoints']) if features['keypoints'] else 0
#                     print(f"Loaded: {medicine_name} ({kp_count} features)")
        
#         total_samples = sum(len(v['samples']) for v in self.medicine_db.values())
#         print(f"✓ Loaded {len(self.medicine_db)} medicines with {total_samples} samples")
    
#     def _extract_medicine_features(self, image):
#         """
#         Extract features from medicine image (optimized for speed).
        
#         Args:
#             image: Medicine image (BGR)
        
#         Returns:
#             dict: Extracted features
#         """
#         # Fast medicine extraction
#         medicine_img, mask, bbox = self._extract_medicine_object(image)
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(medicine_img, cv2.COLOR_BGR2GRAY)
        
#         # Extract ORB features only (much faster than SIFT)
#         keypoints, descriptors = self.feature_detector.detectAndCompute(gray, mask)
        
#         # Extract color histogram (only from medicine region)
#         color_hist = self._compute_color_histogram(medicine_img, mask)

#         # Lightweight shape summary for optional matching
#         shape_features = self._compute_shape_features(medicine_img, mask)
        
#         return {
#             'keypoints': keypoints,
#             'descriptors': descriptors,
#             'color_hist': color_hist,
#             'shape_features': shape_features,
#             'extracted_image': medicine_img,
#             'mask': mask,
#             'bbox': bbox
#         }
    
#     def _extract_medicine_object(self, image):
#         """
#         Fast medicine extraction using adaptive thresholding and contours.
#         Similar to number plate detection - finds largest rectangular object.
        
#         Args:
#             image: Input image with medicine
        
#         Returns:
#             tuple: (extracted_medicine_image, mask, bbox)
#         """
#         h, w = image.shape[:2]
        
#         # Resize for faster processing
#         scale = 1.0
#         if max(h, w) > 800:
#             scale = 800.0 / max(h, w)
#             image_resized = cv2.resize(image, None, fx=scale, fy=scale)
#         else:
#             image_resized = image.copy()
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
#         # Apply bilateral filter to reduce noise while keeping edges
#         filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
#         # Adaptive thresholding to handle lighting variations
#         thresh = cv2.adaptiveThreshold(
#             filtered, 255, 
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#             cv2.THRESH_BINARY_INV, 11, 2
#         )
        
#         # Morphological operations to clean up
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#         morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
#         morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
#         # Find contours
#         contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if not contours:
#             # If no contours, use center region
#             margin = 0.1
#             x1, y1 = int(w * margin), int(h * margin)
#             x2, y2 = int(w * (1 - margin)), int(h * (1 - margin))
#             mask = np.zeros(image.shape[:2], np.uint8)
#             cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
#             extracted = cv2.bitwise_and(image, image, mask=mask)
#             return extracted, mask, (x1, y1, x2 - x1, y2 - y1)
        
#         # Find the best contour (likely medicine)
#         # Filter by area, aspect ratio, and position
#         img_area = w * h
#         valid_contours = []
        
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
            
#             # Must be between 5% and 90% of image
#             if area < img_area * 0.05 or area > img_area * 0.90:
#                 continue
            
#             # Get bounding rectangle
#             x, y, cw, ch = cv2.boundingRect(cnt)
            
#             # Scale back to original size
#             if scale != 1.0:
#                 x, y = int(x / scale), int(y / scale)
#                 cw, ch = int(cw / scale), int(ch / scale)
            
#             # Check aspect ratio (medicines are usually not too elongated)
#             aspect_ratio = float(cw) / ch if ch > 0 else 0
#             if aspect_ratio < 0.2 or aspect_ratio > 5:
#                 continue
            
#             # Prefer contours closer to center
#             cx, cy = x + cw // 2, y + ch // 2
#             center_x, center_y = w // 2, h // 2
#             dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
#             valid_contours.append({
#                 'contour': cnt,
#                 'area': area,
#                 'bbox': (x, y, cw, ch),
#                 'dist': dist_to_center
#             })
        
#         if not valid_contours:
#             # Fallback: use center region
#             margin = 0.1
#             x1, y1 = int(w * margin), int(h * margin)
#             x2, y2 = int(w * (1 - margin)), int(h * (1 - margin))
#             mask = np.zeros(image.shape[:2], np.uint8)
#             cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
#             extracted = cv2.bitwise_and(image, image, mask=mask)
#             return extracted, mask, (x1, y1, x2 - x1, y2 - y1)
        
#         # Choose best contour: largest area with bonus for being centered
#         best = max(valid_contours, key=lambda c: c['area'] - c['dist'] * 0.1)
#         x, y, cw, ch = best['bbox']
        
#         # Create mask from contour
#         mask = np.zeros(image.shape[:2], np.uint8)
        
#         # Scale contour back if needed
#         if scale != 1.0:
#             scaled_cnt = (best['contour'] / scale).astype(np.int32)
#         else:
#             scaled_cnt = best['contour']
        
#         cv2.drawContours(mask, [scaled_cnt], -1, 255, -1)
        
#         # Dilate mask slightly to include edges
#         kernel = np.ones((3, 3), np.uint8)
#         mask = cv2.dilate(mask, kernel, iterations=1)
        
#         # Extract medicine
#         extracted = cv2.bitwise_and(image, image, mask=mask)
        
#         # Crop to bounding box with some padding
#         pad = 10
#         x1 = max(0, x - pad)
#         y1 = max(0, y - pad)
#         x2 = min(w, x + cw + pad)
#         y2 = min(h, y + ch + pad)
        
#         extracted = extracted[y1:y2, x1:x2]
#         mask = mask[y1:y2, x1:x2]
        
#         return extracted, mask, (x1, y1, x2 - x1, y2 - y1)
    
#     def _compute_color_histogram(self, image, mask=None):
#         """Compute color histogram (fast version)."""
#         hist = cv2.calcHist(
#             [image], 
#             [0, 1, 2], 
#             mask, 
#             self.hist_bins, 
#             self.hist_range
#         )
#         hist = cv2.normalize(hist, hist).flatten()
#         return hist

#     def _compute_shape_features(self, image, mask):
#         """Compute basic shape descriptors from the extracted mask."""
#         if mask is None:
#             return None

#         try:
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         except cv2.error:
#             return None

#         if not contours:
#             return None

#         contour = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(contour)
#         if area <= 0:
#             return None

#         x, y, w, h = cv2.boundingRect(contour)
#         rect_area = max(w * h, 1)
#         aspect_ratio = float(w) / h if h > 0 else 0.0
#         extent = float(area) / rect_area

#         hull = cv2.convexHull(contour)
#         hull_area = cv2.contourArea(hull) if hull is not None and len(hull) >= 3 else area
#         solidity = float(area) / hull_area if hull_area > 0 else 0.0

#         moments = cv2.moments(contour)
#         hu_moments = cv2.HuMoments(moments).flatten()
#         # Log transform stabilises magnitudes for comparison
#         with np.errstate(divide='ignore'):
#             hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-12)

#         return {
#             'aspect_ratio': aspect_ratio,
#             'extent': extent,
#             'solidity': solidity,
#             'hu_moments': hu_log
#         }

#     def _update_color_histogram(self, medicine_name, new_hist):
#         """Update the running average color histogram for a medicine."""
#         if new_hist is None:
#             return

#         entry = self.medicine_db.get(medicine_name)
#         if not entry:
#             return

#         samples = entry.get('samples', [])
#         sample_count = len(samples)
#         if sample_count <= 0:
#             entry['avg_color_hist'] = new_hist
#             return

#         prev_avg = entry.get('avg_color_hist')
#         if prev_avg is None:
#             entry['avg_color_hist'] = new_hist
#             return

#         entry['avg_color_hist'] = ((prev_avg * (sample_count - 1)) + new_hist) / sample_count

#     def add_reference(self, medicine_name, image=None, camera_index=0):
#         """
#         Add a new medicine reference with visual preview and automatic extraction.
        
#         Args:
#             medicine_name (str): Name of the medicine
#             image (numpy.ndarray, optional): Image of the medicine or None to capture from camera
#             camera_index (int): Camera index to use for capture if image is None
        
#         Returns:
#             bool: True if reference was added successfully
#         """
#         # Sanitize medicine name for filename
#         safe_name = "".join(c if c.isalnum() else "_" for c in medicine_name)
        
#         # If no image provided, capture from camera with live preview
#         if image is None:
#             print(f"Capturing reference for {medicine_name}...")
#             cap = cv2.VideoCapture(camera_index)
#             if not cap.isOpened():
#                 print("Error: Could not open camera")
#                 return False
            
#             # Set camera properties
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
#             captured_image = None
#             countdown = 5
#             last_second = time.time()
            
#             print("Position medicine in the center of the frame...")
#             print("Press SPACE to capture early, or wait for countdown")
            
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("Error: Could not read from camera")
#                     cap.release()
#                     cv2.destroyAllWindows()
#                     return False
                
#                 # Create display frame
#                 display_frame = frame.copy()
#                 h, w = display_frame.shape[:2]
                
#                 # Draw guide rectangle (where medicine should be)
#                 margin = 0.15
#                 rect_x1 = int(w * margin)
#                 rect_y1 = int(h * margin)
#                 rect_x2 = int(w * (1 - margin))
#                 rect_y2 = int(h * (1 - margin))
                
#                 # Draw semi-transparent overlay
#                 overlay = display_frame.copy()
#                 cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 3)
#                 cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
#                 # Extract medicine in real-time to show preview
#                 try:
#                     extracted, mask, bbox = self._extract_medicine_object(frame)
                    
#                     # Show extracted medicine in corner
#                     if extracted is not None and extracted.size > 0:
#                         # Resize for preview
#                         preview_h = min(200, extracted.shape[0])
#                         preview_w = int(extracted.shape[1] * (preview_h / extracted.shape[0]))
                        
#                         if preview_w > 0 and preview_h > 0:
#                             preview = cv2.resize(extracted, (preview_w, preview_h))
                            
#                             # Place in top-right corner
#                             y_offset = 10
#                             x_offset = w - preview_w - 10
                            
#                             # Add background for preview
#                             cv2.rectangle(display_frame, 
#                                         (x_offset - 5, y_offset - 5),
#                                         (x_offset + preview_w + 5, y_offset + preview_h + 5),
#                                         (0, 0, 0), -1)
#                             cv2.rectangle(display_frame, 
#                                         (x_offset - 5, y_offset - 5),
#                                         (x_offset + preview_w + 5, y_offset + preview_h + 5),
#                                         (0, 255, 0), 2)
                            
#                             display_frame[y_offset:y_offset+preview_h, x_offset:x_offset+preview_w] = preview
                            
#                             # Label
#                             cv2.putText(display_frame, "Extracted Medicine", 
#                                       (x_offset, y_offset - 10),
#                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 except:
#                     pass  # If extraction fails, just continue
                
#                 # Update countdown
#                 current_time = time.time()
#                 if current_time - last_second >= 1.0:
#                     countdown -= 1
#                     last_second = current_time
                
#                 # Display instructions
#                 if countdown > 0:
#                     cv2.putText(display_frame, f"Capturing in {countdown}...", 
#                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#                     cv2.putText(display_frame, f"Medicine: {medicine_name}", 
#                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                     cv2.putText(display_frame, "Press SPACE to capture now", 
#                               (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#                     cv2.putText(display_frame, "Press ESC to cancel", 
#                               (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#                 else:
#                     # Auto capture
#                     captured_image = frame.copy()
#                     break
                
#                 cv2.imshow("Add Medicine Reference", display_frame)
                
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord(' '):  # Space to capture
#                     captured_image = frame.copy()
#                     break
#                 elif key == 27:  # ESC to cancel
#                     cap.release()
#                     cv2.destroyAllWindows()
#                     print("Capture cancelled")
#                     return False
            
#             cap.release()
            
#             if captured_image is None:
#                 cv2.destroyAllWindows()
#                 print("Error: Failed to capture image")
#                 return False
            
#             image = captured_image
            
#             # Show final captured image with extracted medicine
#             extracted, mask, bbox = self._extract_medicine_object(image)
            
#             # Create side-by-side comparison
#             h1, w1 = image.shape[:2]
#             h2, w2 = extracted.shape[:2] if extracted is not None else (h1, w1)
            
#             # Resize for display
#             display_h = 400
#             display_w1 = int(w1 * (display_h / h1))
#             display_w2 = int(w2 * (display_h / h2)) if h2 > 0 else display_w1
            
#             display_original = cv2.resize(image, (display_w1, display_h))
#             display_extracted = cv2.resize(extracted, (display_w2, display_h)) if extracted is not None else display_original
            
#             # Create comparison image
#             comparison = np.zeros((display_h, display_w1 + display_w2 + 20, 3), dtype=np.uint8)
#             comparison[:, :display_w1] = display_original
#             comparison[:, display_w1 + 20:] = display_extracted
            
#             # Add labels
#             cv2.putText(comparison, "Original", (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             cv2.putText(comparison, "Extracted Medicine", (display_w1 + 30, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(comparison, "Press 'Y' to save, 'N' to cancel", 
#                        (10, display_h - 10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
#             cv2.imshow("Confirm Medicine Reference", comparison)
#             print("Press 'y' to save this reference, 'n' to cancel")
            
#             while True:
#                 key = cv2.waitKey(0) & 0xFF
#                 if key == ord('y') or key == ord('Y'):
#                     cv2.destroyAllWindows()
#                     break
#                 elif key == ord('n') or key == ord('N') or key == 27:
#                     cv2.destroyAllWindows()
#                     print("Reference addition canceled")
#                     return False
        
#         # Ensure the reference folder exists
#         if not os.path.exists(self.reference_folder):
#             os.makedirs(self.reference_folder)
        
#         # Extract medicine from image
#         print("Processing and extracting medicine...")
#         extracted_medicine, mask, bbox = self._extract_medicine_object(image)
        
#         # Save BOTH original and extracted versions
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Save extracted medicine (primary)
#         filename_extracted = f"{safe_name}_{timestamp}.jpg"
#         file_path_extracted = os.path.join(self.reference_folder, filename_extracted)
#         cv2.imwrite(file_path_extracted, extracted_medicine)
        
#         # Save original as backup
#         filename_original = f"{safe_name}_{timestamp}_original.jpg"
#         file_path_original = os.path.join(self.reference_folder, filename_original)
#         cv2.imwrite(file_path_original, image)
        
#         # Extract and store features
#         features = self._extract_medicine_features(image)
        
#         if features['descriptors'] is not None:
#             # Add to database
#             if safe_name not in self.medicine_db:
#                 self.medicine_db[safe_name] = {
#                     'samples': [],
#                     'avg_color_hist': None
#                 }
            
#             self.medicine_db[safe_name]['samples'].append({
#                 'features': features,
#                 'image': extracted_medicine
#             })
            
#             self._update_color_histogram(safe_name, features['color_hist'])
            
#             sift_kp = len(features['sift_keypoints']) if features['sift_keypoints'] else 0
#             orb_kp = len(features['orb_keypoints']) if features['orb_keypoints'] else 0
#             print(f"✓ Added reference for: {medicine_name}")
#             print(f"  - SIFT keypoints: {sift_kp}")
#             print(f"  - ORB keypoints: {orb_kp}")
#             print(f"  - Extracted image saved: {filename_extracted}")
#             return True
#         else:
#             print(f"Warning: Not enough features detected in the image")
#             return False

#     def identify_medicine(self, image):
#         """
#         Identify medicine using fast ORB matching and color comparison.
        
#         Args:
#             image (numpy.ndarray): Image containing medicine to identify
        
#         Returns:
#             dict: Result containing medicine name, confidence, and match details
#         """
#         if not self.medicine_db:
#             return {'medicine': None, 'confidence': 0, 'message': "No medicine references loaded"}
        
#         # Extract features from query image
#         query_features = self._extract_medicine_features(image)
        
#         if query_features['descriptors'] is None:
#             return {'medicine': None, 'confidence': 0, 'message': "Not enough features detected"}
        
#         # Score each medicine (optimized)
#         best_name = None
#         best_score = 0
#         best_sample = None
#         all_scores = {}
        
#         for medicine_name, medicine_data in self.medicine_db.items():
#             max_score = 0
#             sample_match = None
            
#             # Compare against all samples of this medicine
#             for sample in medicine_data['samples']:
#                 score = self._fast_match_score(query_features, sample['features'], medicine_data)
                
#                 if score > max_score:
#                     max_score = score
#                     sample_match = sample
            
#             all_scores[medicine_name] = max_score
            
#             if max_score > best_score:
#                 best_score = max_score
#                 best_name = medicine_name
#                 best_sample = sample_match
        
#         # Check if confidence meets threshold
#         if best_score < self.confidence_threshold:
#             return {
#                 'medicine': None,
#                 'confidence': best_score,
#                 'message': f"Low confidence: {best_name} ({best_score:.0%})" if best_name else "No match",
#                 'top_match': best_name,
#                 'all_scores': all_scores
#             }
        
#         # Return result
#         return {
#             'medicine': best_name,
#             'confidence': best_score,
#             'match_details': {
#                 'reference_image': best_sample['image'] if best_sample else None,
#                 'query_extracted': query_features['extracted_image'],
#                 'bbox': query_features['bbox']
#             },
#             'message': f"Identified as {best_name}",
#             'all_scores': all_scores
#         }
    
#     def _fast_match_score(self, query_features, ref_features, medicine_data):
#         """
#         Fast matching using ORB features, color histogram and optional shape features.
#         Returns a normalized score in [0.0, 1.0].
#         """
#         scores = []
#         weights = []
        
#         # 1. ORB feature matching (primary - fast and accurate)
#         q_desc = query_features.get('descriptors')
#         r_desc = ref_features.get('descriptors')
#         if q_desc is not None and r_desc is not None and len(q_desc) >= 2 and len(r_desc) >= 2:
#             try:
#                 matches = self.matcher.knnMatch(q_desc, r_desc, k=2)
#                 good_matches = []
#                 for pair in matches:
#                     if len(pair) == 2:
#                         m, n = pair
#                         if m.distance < 0.75 * n.distance:
#                             good_matches.append(m)
#                 max_possible = min(len(q_desc), len(r_desc))
#                 orb_score = (len(good_matches) / max_possible) if max_possible > 0 else 0.0
#                 scores.append(orb_score)
#                 weights.append(0.7)
#             except Exception:
#                 pass
        
#         # 2. Color histogram (secondary)
#         q_hist = query_features.get('color_hist')
#         avg_hist = medicine_data.get('avg_color_hist')
#         if q_hist is not None and avg_hist is not None:
#             try:
#                 color_score = cv2.compareHist(q_hist, avg_hist, cv2.HISTCMP_CORREL)
#                 color_score = float(max(0.0, min(1.0, color_score)))  # clamp to [0,1]
#                 scores.append(color_score)
#                 weights.append(0.2)
#             except Exception:
#                 pass
        
#         # 3. Shape matching (optional, light weight)
#         q_shape = query_features.get('shape_features')
#         r_shape = ref_features.get('shape_features')
#         if q_shape is not None and r_shape is not None:
#             try:
#                 shape_score = self._match_shape_features(q_shape, r_shape)
#                 scores.append(shape_score)
#                 weights.append(0.1)
#             except Exception:
#                 pass
        
#         # If no scoring components, return 0
#         if not scores:
#             return 0.0
        
#         # Weighted average of scores
#         total_weight = sum(weights)
#         if total_weight <= 0:
#             return 0.0
#         normalized_weights = [w / total_weight for w in weights]
#         final_score = sum(s * w for s, w in zip(scores, normalized_weights))
        
#         return float(min(1.0, max(0.0, final_score)))
    
#     def _match_sift_features(self, query_desc, query_kp, ref_desc, ref_kp):
#         """Match SIFT features using FLANN matcher with ratio test."""
#         if len(query_desc) < 2 or len(ref_desc) < 2:
#             return 0.0
        
#         try:
#             matches = self.sift_matcher.knnMatch(query_desc, ref_desc, k=2)
#         except:
#             return 0.0
        
#         # Apply Lowe's ratio test
#         good_matches = []
#         for match_pair in matches:
#             if len(match_pair) == 2:
#                 m, n = match_pair
#                 if m.distance < 0.7 * n.distance:
#                     good_matches.append(m)
        
#         if len(good_matches) < 4:
#             return 0.0
        
#         # Try to find homography for geometric verification
#         src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#         dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
#         try:
#             H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#             inliers = np.sum(mask) if mask is not None else 0
#         except:
#             inliers = 0
        
#         # Score based on inliers ratio
#         if len(good_matches) > 0:
#             score = (inliers / len(good_matches)) * min(1.0, len(good_matches) / 50)
#         else:
#             score = 0.0
        
#         return score
    
#     def _match_orb_features(self, query_desc, ref_desc):
#         """Match ORB features using brute-force matcher."""
#         if len(query_desc) < 2 or len(ref_desc) < 2:
#             return 0.0
        
#         try:
#             matches = self.orb_matcher.knnMatch(query_desc, ref_desc, k=2)
#         except:
#             return 0.0
        
#         # Apply ratio test
#         good_matches = []
#         for match_pair in matches:
#             if len(match_pair) == 2:
#                 m, n = match_pair
#                 if m.distance < 0.75 * n.distance:
#                     good_matches.append(m)
        
#         # Score based on match ratio
#         max_possible = min(len(query_desc), len(ref_desc))
#         if max_possible > 0:
#             score = len(good_matches) / max_possible
#         else:
#             score = 0.0
        
#         return min(1.0, score * 2)  # Amplify a bit since ORB typically has fewer matches
    
#     def _match_shape_features(self, query_shape, ref_shape):
#         """Match shape features."""
#         # Compare aspect ratios
#         aspect_diff = abs(query_shape['aspect_ratio'] - ref_shape['aspect_ratio'])
#         aspect_score = max(0, 1 - aspect_diff)
        
#         # Compare extent
#         extent_diff = abs(query_shape['extent'] - ref_shape['extent'])
#         extent_score = max(0, 1 - extent_diff)
        
#         # Compare solidity
#         solidity_diff = abs(query_shape['solidity'] - ref_shape['solidity'])
#         solidity_score = max(0, 1 - solidity_diff)
        
#         # Hu moments comparison
#         try:
#             hu_diff = np.sum(np.abs(query_shape['hu_moments'] - ref_shape['hu_moments']))
#             hu_score = max(0, 1 - (hu_diff / 10))  # Normalize
#         except:
#             hu_score = 0
        
#         # Average all shape scores
#         shape_score = (aspect_score + extent_score + solidity_score + hu_score) / 4
        
#         return shape_score

#     def run_detection(self, camera_index=0, display=True):
#         """
#         Run continuous medicine detection from camera feed with enhanced visualization.
        
#         Args:
#             camera_index (int): Camera device index
#             display (bool): Whether to show visualization window
#         """
#         # Check if we have references
#         if not self.medicine_db:
#             print("No medicine references loaded. Add references first.")
#             return
            
#         cap = cv2.VideoCapture(camera_index)
#         if not cap.isOpened():
#             print("Error: Could not open camera")
#             return
        
#         # Set higher resolution
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
#         print("Medicine detection started. Press 'q' to quit.")
#         print("The extracted medicine will be shown in the top-right corner.")
        
#         # For FPS calculation
#         prev_frame_time = 0
#         new_frame_time = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Failed to capture image")
#                 break
                
#             # Calculate FPS
#             new_frame_time = time.time()
#             fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
#             prev_frame_time = new_frame_time
#             fps_text = f"FPS: {int(fps)}"
            
#             # Process frame (make a copy to avoid modifying the original)
#             display_frame = frame.copy()
#             h, w = display_frame.shape[:2]
            
#             # Detect medicine
#             result = self.identify_medicine(frame)
            
#             # Show extracted medicine in corner (always, to help with positioning)
#             try:
#                 extracted, mask, bbox = self._extract_medicine_object(frame)
                
#                 if extracted is not None and extracted.shape[0] > 0 and extracted.shape[1] > 0:
#                     # Resize for preview
#                     preview_h = min(200, h // 3)
#                     preview_w = int(extracted.shape[1] * (preview_h / extracted.shape[0]))
                    
#                     if preview_w > 0 and preview_h > 0 and preview_w < w:
#                         preview = cv2.resize(extracted, (preview_w, preview_h))
                        
#                         # Place in top-right corner
#                         y_offset = 10
#                         x_offset = w - preview_w - 10
                        
#                         # Add background
#                         cv2.rectangle(display_frame, 
#                                     (x_offset - 5, y_offset - 5),
#                                     (x_offset + preview_w + 5, y_offset + preview_h + 5),
#                                     (0, 0, 0), -1)
#                         cv2.rectangle(display_frame, 
#                                     (x_offset - 5, y_offset - 5),
#                                     (x_offset + preview_w + 5, y_offset + preview_h + 5),
#                                     (0, 255, 255), 2)
                        
#                         display_frame[y_offset:y_offset+preview_h, x_offset:x_offset+preview_w] = preview
                        
#                         # Label
#                         cv2.putText(display_frame, "Extracted", 
#                                   (x_offset, y_offset - 10),
#                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#             except Exception as e:
#                 pass  # If extraction fails, just continue
            
#             # Display results on frame
#             if result['medicine']:
#                 # Draw medicine info with better visibility
#                 # Background rectangle for text
#                 cv2.rectangle(display_frame, (5, 5), (450, 110), (0, 0, 0), -1)
#                 cv2.rectangle(display_frame, (5, 5), (450, 110), (0, 255, 0), 2)
                
#                 cv2.putText(display_frame, f"Medicine: {result['medicine']}", (10, 35), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                 cv2.putText(display_frame, f"Confidence: {result['confidence']:.1%}", (10, 70), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
#                 # Show top 3 matches if available
#                 if 'all_scores' in result and result['all_scores']:
#                     sorted_scores = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
#                     score_text = " | ".join([f"{name}: {score:.0%}" for name, score in sorted_scores])
#                     cv2.putText(display_frame, score_text, (10, 95), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
#             else:
#                 # No medicine detected or low confidence
#                 status_text = "No medicine detected"
#                 status_color = (0, 0, 255)
                
#                 if result.get('top_match'):
#                     status_text = f"Low confidence: {result['top_match']} ({result['confidence']:.0%})"
#                     status_color = (0, 165, 255)  # Orange
                
#                 cv2.rectangle(display_frame, (5, 5), (550, 50), (0, 0, 0), -1)
#                 cv2.rectangle(display_frame, (5, 5), (550, 50), status_color, 2)
#                 cv2.putText(display_frame, status_text, (10, 35), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
#             # Add FPS counter and instructions
#             cv2.putText(display_frame, fps_text, (10, h - 40), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             cv2.putText(display_frame, "Press 'Q' to quit", (10, h - 10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
#             # Show the frame
#             if display:
#                 cv2.imshow('Medicine Recognizer - Advanced', display_frame)
                
#             # Break loop on 'q' key
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#         # Clean up
#         cap.release()
#         if display:
#             cv2.destroyAllWindows()

#     def save_recognition(self, image, result, output_folder="recognized"):
#         """
#         Save the recognition result to file.
        
#         Args:
#             image: Image with the medicine
#             result: Recognition result from identify_medicine
#             output_folder: Folder to save results to
#         """
#         # Ensure output folder exists
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
            
#         # Generate filename with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         med_name = result['medicine'] if result['medicine'] else "unknown"
#         filename = f"{med_name}_{timestamp}.jpg"
#         file_path = os.path.join(output_folder, filename)
        
#         # Create an annotated image
#         annotated = image.copy()
#         cv2.putText(annotated, f"Medicine: {med_name}", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         cv2.putText(annotated, f"Confidence: {result['confidence']:.2f}", (10, 70), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         # Save the annotated image
#         cv2.imwrite(file_path, annotated)
#         print(f"Recognition saved to {file_path}")


# def main():
#     """Main function to demonstrate the MedicineRecognizer class."""
#     print("Medicine Recognizer")
#     print("==================")
    
#     # Create the recognizer
#     recognizer = MedicineRecognizer()
    
#     # Check if we have references
#     if not recognizer.medicine_db:
#         print("\nNo medicine references found. Let's add some:")
        
#         # Demo: Add a few references
#         add_more = True
#         while add_more:
#             medicine_name = input("\nEnter medicine name (or 'q' to quit adding): ")
#             if medicine_name.lower() == 'q':
#                 add_more = False
#                 continue
                
#             print(f"Adding reference for {medicine_name}...")
#             recognizer.add_reference(medicine_name)
            
#             choice = input("Add another medicine? (y/n): ")
#             if choice.lower() != 'y':
#                 add_more = False
    
#     # Run the detection loop
#     print("\nStarting medicine detection. Press 'q' to quit.")
#     recognizer.run_detection()


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Enhanced Medicine Recognizer

Uses YOLOv8 for object detection + EfficientNet for classification
Lightweight, fast, and highly accurate medicine recognition system.
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
from collections import defaultdict
import pickle

# Deep Learning imports (install: pip install ultralytics torch torchvision pillow)
try:
    from ultralytics import YOLO
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    import torch.nn as nn
    from PIL import Image
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    print("Warning: Deep learning libraries not installed. Using fallback mode.")
    print("Install with: pip install ultralytics torch torchvision pillow")
    DEEP_LEARNING_AVAILABLE = False
    YOLO = None
    torch = None
    transforms = None
    efficientnet_b0 = None
    EfficientNet_B0_Weights = None
    nn = None
    Image = None


class MedicineRecognizer:
    """
    High-performance medicine recognition using:
    1. YOLOv8-nano for fast object detection/segmentation
    2. EfficientNet-B0 for classification
    3. Smart caching and preprocessing
    """

    def __init__(self, reference_folder="medicine_references", 
                 model_folder="models",
                 confidence_threshold=0.70,
                 use_gpu=True):
        """
        Initialize the Enhanced Medicine Recognizer.
        
        Args:
            reference_folder (str): Folder containing reference medicine images
            model_folder (str): Folder to store trained models
            confidence_threshold (float): Threshold for detection confidence
            use_gpu (bool): Whether to use GPU acceleration
        """
        self.reference_folder = reference_folder
        self.model_folder = model_folder
        self.confidence_threshold = confidence_threshold
        self.medicine_db = {}
        
        # Create folders
        os.makedirs(reference_folder, exist_ok=True)
        os.makedirs(model_folder, exist_ok=True)

        # Default device and preprocess
        self.device = 'cpu'
        self.preprocess = None

        # Check device
        if DEEP_LEARNING_AVAILABLE and torch is not None:
            if use_gpu and torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("✓ Using CPU")
        else:
            print("✓ Using CPU (deep learning backend unavailable)")

        # Image preprocessing for classification
        if DEEP_LEARNING_AVAILABLE and transforms is not None:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

        # Initialize models (requires device)
        self._init_models()

        # Load references (uses preprocess if available)
        self._load_references()

        # Cache for speed
        self.frame_cache = {}
        self.cache_max_size = 50

    def _init_models(self):
        """Initialize detection and classification models."""
        if not DEEP_LEARNING_AVAILABLE:
            print("Deep learning not available. Using basic OpenCV detection.")
            self.detector = None
            self.classifier = None
            return
        
        print("Initializing models...")
        
        # 1. Object detector: YOLOv8-nano (very fast, ~3MB)
        try:
            # Try to load custom-trained model first
            custom_model_path = os.path.join(self.model_folder, "medicine_detector.pt")
            if os.path.exists(custom_model_path):
                self.detector = YOLO(custom_model_path)
                print("✓ Loaded custom medicine detector")
            else:
                # Use pretrained YOLOv8n for general object detection
                self.detector = YOLO('yolov8n.pt')
                print("✓ Loaded YOLOv8-nano detector")
        except Exception as e:
            print(f"Warning: Could not load YOLO: {e}")
            self.detector = None
        
        # 2. Classifier: EfficientNet-B0 (lightweight, ~20MB)
        try:
            classifier_path = os.path.join(self.model_folder, "medicine_classifier.pth")
            
            # Load pretrained EfficientNet
            self.classifier = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            
            # Modify final layer for our classes
            num_medicines = len(self.medicine_db) if self.medicine_db else 10
            self.classifier.classifier[1] = nn.Linear(
                self.classifier.classifier[1].in_features, 
                max(num_medicines, 2)  # At least 2 classes
            )
            
            # Load trained weights if available
            if os.path.exists(classifier_path):
                self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
                print("✓ Loaded trained classifier")
            else:
                print("✓ Loaded pretrained EfficientNet (needs training)")
            
            self.classifier.to(self.device)
            self.classifier.eval()
            
        except Exception as e:
            print(f"Warning: Could not load classifier: {e}")
            self.classifier = None

    def _load_references(self):
        """Load reference medicine images and compute embeddings."""
        print("Loading medicine references...")
        
        # Try to load cached database
        db_path = os.path.join(self.model_folder, "medicine_db.pkl")
        if os.path.exists(db_path):
            try:
                with open(db_path, 'rb') as f:
                    self.medicine_db = pickle.load(f)
                print(f"✓ Loaded {len(self.medicine_db)} medicines from cache")
                return
            except:
                print("Cache corrupted, rebuilding...")
        
        # Build database from images
        if not os.path.exists(self.reference_folder):
            return
        
        for filename in os.listdir(self.reference_folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if 'original' in filename.lower():
                continue
                
            # Extract medicine name
            medicine_name = "_".join(filename.split("_")[:-1]) if "_" in filename else os.path.splitext(filename)[0]
            image_path = os.path.join(self.reference_folder, filename)
            
            # Load and process
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            # Extract features
            features = self._extract_features(img)
            
            if medicine_name not in self.medicine_db:
                self.medicine_db[medicine_name] = {
                    'samples': [],
                    'embeddings': []
                }
            
            self.medicine_db[medicine_name]['samples'].append({
                'image_path': image_path,
                'features': features
            })
            
            print(f"✓ Loaded: {medicine_name}")
        
        # Save cache
        self._save_database()
        
        total = sum(len(v['samples']) for v in self.medicine_db.values())
        print(f"✓ Loaded {len(self.medicine_db)} medicines, {total} samples")

    def _save_database(self):
        """Save medicine database to disk."""
        db_path = os.path.join(self.model_folder, "medicine_db.pkl")
        try:
            with open(db_path, 'wb') as f:
                pickle.dump(self.medicine_db, f)
        except Exception as e:
            print(f"Warning: Could not save database: {e}")

    def _extract_features(self, image):
        """Extract features from medicine image using deep learning."""
        features = {}
        
        if (
            DEEP_LEARNING_AVAILABLE
            and self.classifier is not None
            and self.preprocess is not None
            and Image is not None
        ):
            try:
                # Convert to PIL
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                # Preprocess
                img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                
                # Extract embedding (features before final layer)
                with torch.no_grad():
                    # Get features from second-to-last layer
                    x = self.classifier.features(img_tensor)
                    x = self.classifier.avgpool(x)
                    embedding = torch.flatten(x, 1)
                    features['embedding'] = embedding.cpu().numpy()
                    
                    # Also get classification output
                    output = self.classifier(img_tensor)
                    features['output'] = output.cpu().numpy()
                    
            except Exception as e:
                print(f"Warning: Feature extraction failed: {e}")
        
        # Fallback: color histogram
        features['color_hist'] = self._compute_color_histogram(image)
        
        # Fallback: shape features
        features['shape'] = self._compute_shape_features(image)
        
        return features

    def _extract_medicine_object(self, image):
        """
        Extract medicine from image using YOLO or fallback method.
        Returns: (extracted_image, mask, bbox, confidence)
        """
        if DEEP_LEARNING_AVAILABLE and self.detector is not None:
            try:
                # Run YOLO detection
                results = self.detector(image, verbose=False)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Get the detection with highest confidence
                    boxes = results[0].boxes
                    confidences = boxes.conf.cpu().numpy()
                    best_idx = np.argmax(confidences)
                    
                    # Get bounding box
                    box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    confidence = confidences[best_idx]
                    
                    # Extract region
                    extracted = image[y1:y2, x1:x2].copy()
                    
                    # Create mask
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    mask[y1:y2, x1:x2] = 255
                    
                    # Try to get segmentation mask if available
                    if hasattr(results[0], 'masks') and results[0].masks is not None:
                        seg_mask = results[0].masks.data[best_idx].cpu().numpy()
                        mask = (seg_mask * 255).astype(np.uint8)
                    
                    return extracted, mask, (x1, y1, x2-x1, y2-y1), float(confidence)
                    
            except Exception as e:
                print(f"YOLO detection failed: {e}")
        
        # Fallback: Use advanced OpenCV segmentation
        return self._extract_medicine_opencv(image)

    def _extract_medicine_opencv(self, image):
        """
        Advanced OpenCV-based medicine extraction.
        Uses GrabCut algorithm for better segmentation.
        """
        h, w = image.shape[:2]
        
        # Create initial rectangle (assume medicine is in center 60% of image)
        margin = 0.2
        rect = (int(w*margin), int(h*margin), int(w*0.6), int(h*0.6))
        
        # Initialize mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # GrabCut algorithm models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Apply GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Find largest contour
            contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Create clean mask
                final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
                
                # Extract
                extracted = cv2.bitwise_and(image, image, mask=final_mask)
                extracted = extracted[y:y+h, x:x+w]
                final_mask = final_mask[y:y+h, x:x+w]
                
                return extracted, final_mask, (x, y, w, h), 0.8
                
        except Exception as e:
            print(f"GrabCut failed: {e}")
        
        # Simple fallback: center crop
        x1, y1 = int(w*0.2), int(h*0.2)
        x2, y2 = int(w*0.8), int(h*0.8)
        extracted = image[y1:y2, x1:x2].copy()
        mask = np.ones(extracted.shape[:2], dtype=np.uint8) * 255
        
        return extracted, mask, (x1, y1, x2-x1, y2-y1), 0.5

    def _compute_color_histogram(self, image):
        """Compute color histogram."""
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _compute_shape_features(self, image):
        """Compute shape features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        
        # Compute features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Hu moments
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        return {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'hu_moments': hu_moments
        }

    def add_reference(self, medicine_name, image=None, camera_index=0):
        """
        Add a new medicine reference with camera capture.
        
        Args:
            medicine_name (str): Name of the medicine
            image (numpy.ndarray, optional): Image or None to capture
            camera_index (int): Camera index
        
        Returns:
            bool: Success status
        """
        safe_name = "".join(c if c.isalnum() or c in "_ " else "_" for c in medicine_name)
        
        # Capture from camera if needed
        if image is None:
            image = self._capture_with_preview(medicine_name, camera_index)
            if image is None:
                return False
        
        # Extract medicine
        print("Processing medicine...")
        extracted, mask, bbox, conf = self._extract_medicine_object(image)
        
        # Show preview
        if not self._show_confirmation(image, extracted, medicine_name):
            return False
        
        # Save images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.jpg"
        filepath = os.path.join(self.reference_folder, filename)
        
        cv2.imwrite(filepath, extracted)
        
        # Extract features
        features = self._extract_features(extracted)
        
        # Add to database
        if safe_name not in self.medicine_db:
            self.medicine_db[safe_name] = {
                'samples': [],
                'embeddings': []
            }
        
        self.medicine_db[safe_name]['samples'].append({
            'image_path': filepath,
            'features': features
        })
        
        if 'embedding' in features:
            self.medicine_db[safe_name]['embeddings'].append(features['embedding'])
        
        # Save database
        self._save_database()
        
        print(f"✓ Added reference: {medicine_name}")
        print(f"  Total samples: {len(self.medicine_db[safe_name]['samples'])}")
        
        return True

    def _capture_with_preview(self, medicine_name, camera_index):
        """Capture image with live preview."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        countdown = 5
        last_time = time.time()
        
        print(f"Capturing {medicine_name}... Press SPACE to capture early")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            h, w = display.shape[:2]
            
            # Draw guide
            cv2.rectangle(display, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), (0, 255, 0), 2)
            
            # Show extracted medicine preview
            try:
                extracted, _, _, conf = self._extract_medicine_object(frame)
                if extracted.shape[0] > 0 and extracted.shape[1] > 0:
                    preview_h = 150
                    preview_w = int(extracted.shape[1] * (preview_h / extracted.shape[0]))
                    if preview_w > 0:
                        preview = cv2.resize(extracted, (preview_w, preview_h))
                        display[10:10+preview_h, w-preview_w-10:w-10] = preview
                        cv2.putText(display, f"Conf: {conf:.2f}", (w-preview_w-10, preview_h+30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except:
                pass
            
            # Countdown
            if time.time() - last_time >= 1:
                countdown -= 1
                last_time = time.time()
            
            if countdown > 0:
                cv2.putText(display, f"Capturing in {countdown}...", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                cap.release()
                cv2.destroyAllWindows()
                return frame
            
            cv2.putText(display, medicine_name, (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Capture Medicine", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                cap.release()
                cv2.destroyAllWindows()
                return frame
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        cv2.destroyAllWindows()
        return None

    def _show_confirmation(self, original, extracted, name):
        """Show confirmation dialog."""
        h, w = original.shape[:2]
        display_h = 400
        
        orig_w = int(w * (display_h / h))
        ext_w = int(extracted.shape[1] * (display_h / extracted.shape[0]))
        
        orig_resized = cv2.resize(original, (orig_w, display_h))
        ext_resized = cv2.resize(extracted, (ext_w, display_h))
        
        combined = np.zeros((display_h, orig_w + ext_w + 20, 3), dtype=np.uint8)
        combined[:, :orig_w] = orig_resized
        combined[:, orig_w+20:] = ext_resized
        
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Extracted", (orig_w+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Press 'Y' to save, 'N' to cancel", (10, display_h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Confirm Reference", combined)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y') or key == ord('Y'):
                cv2.destroyAllWindows()
                return True
            elif key == ord('n') or key == ord('N') or key == 27:
                cv2.destroyAllWindows()
                return False

    def identify_medicine(self, image):
        """
        Identify medicine using deep learning or fallback methods.
        
        Args:
            image: Input image
            
        Returns:
            dict: Recognition result
        """
        if not self.medicine_db:
            return {'medicine': None, 'confidence': 0, 'message': "No references loaded"}
        
        # Extract medicine
        extracted, mask, bbox, det_conf = self._extract_medicine_object(image)
        
        if extracted.shape[0] == 0 or extracted.shape[1] == 0:
            return {'medicine': None, 'confidence': 0, 'message': "Could not extract medicine"}
        
        # Extract features
        query_features = self._extract_features(extracted)
        
        # Match using embeddings if available
        if 'embedding' in query_features:
            result = self._match_by_embedding(query_features)
        else:
            result = self._match_by_features(query_features)
        
        result['extracted'] = extracted
        result['bbox'] = bbox
        result['detection_confidence'] = det_conf
        
        return result

    def _match_by_embedding(self, query_features):
        """Match using deep learning embeddings (cosine similarity)."""
        query_emb = query_features['embedding'].flatten()
        
        best_name = None
        best_score = 0
        all_scores = {}
        
        for medicine_name, data in self.medicine_db.items():
            scores = []
            
            for sample in data['samples']:
                if 'embedding' not in sample['features']:
                    continue
                    
                ref_emb = sample['features']['embedding'].flatten()
                
                # Cosine similarity
                similarity = np.dot(query_emb, ref_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(ref_emb))
                similarity = (similarity + 1) / 2  # Normalize to [0, 1]
                scores.append(similarity)
            
            if scores:
                avg_score = np.mean(scores)
                all_scores[medicine_name] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_name = medicine_name
        
        if best_score < self.confidence_threshold:
            return {
                'medicine': None,
                'confidence': best_score,
                'message': f"Low confidence: {best_name} ({best_score:.1%})" if best_name else "No match",
                'all_scores': all_scores
            }
        
        return {
            'medicine': best_name,
            'confidence': best_score,
            'message': f"Identified: {best_name}",
            'all_scores': all_scores
        }

    def _match_by_features(self, query_features):
        """Fallback matching using color and shape features."""
        best_name = None
        best_score = 0
        all_scores = {}
        
        query_hist = query_features.get('color_hist')
        
        for medicine_name, data in self.medicine_db.items():
            scores = []
            
            for sample in data['samples']:
                ref_hist = sample['features'].get('color_hist')
                
                if query_hist is not None and ref_hist is not None:
                    hist_score = cv2.compareHist(query_hist, ref_hist, cv2.HISTCMP_CORREL)
                    hist_score = max(0, hist_score)
                    scores.append(hist_score)
            
            if scores:
                avg_score = np.mean(scores)
                all_scores[medicine_name] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_name = medicine_name
        
        if best_score < self.confidence_threshold:
            return {
                'medicine': None,
                'confidence': best_score,
                'message': "No confident match",
                'all_scores': all_scores
            }
        
        return {
            'medicine': best_name,
            'confidence': best_score,
            'message': f"Identified: {best_name}",
            'all_scores': all_scores
        }

    def run_detection(self, camera_index=0):
        """Run real-time detection from camera."""
        if not self.medicine_db:
            print("No references loaded. Add medicines first.")
            return
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Detection started. Press 'Q' to quit.")
        
        fps_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - fps_time) if fps_time > 0 else 0
            fps_time = current_time
            
            # Detect
            result = self.identify_medicine(frame)
            
            # Display
            display = frame.copy()
            h, w = display.shape[:2]
            
            # Show extracted medicine
            if 'extracted' in result:
                try:
                    ext = result['extracted']
                    if ext.shape[0] > 0 and ext.shape[1] > 0:
                        preview_h = 180
                        preview_w = int(ext.shape[1] * (preview_h / ext.shape[0]))
                        if preview_w > 0 and preview_w < w:
                            preview = cv2.resize(ext, (preview_w, preview_h))
                            y_pos, x_pos = 10, w - preview_w - 10
                            
                            cv2.rectangle(display, (x_pos-5, y_pos-5),
                                        (x_pos+preview_w+5, y_pos+preview_h+5),
                                        (0, 255, 255), 2)
                            display[y_pos:y_pos+preview_h, x_pos:x_pos+preview_w] = preview
                except:
                    pass
            
            # Show results
            if result['medicine']:
                cv2.rectangle(display, (5, 5), (500, 120), (0, 0, 0), -1)
                cv2.rectangle(display, (5, 5), (500, 120), (0, 255, 0), 2)
                
                cv2.putText(display, f"Medicine: {result['medicine']}", (15, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display, f"Confidence: {result['confidence']:.1%}", (15, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Top matches
                if 'all_scores' in result:
                    top3 = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
                    match_text = " | ".join([f"{n}: {s:.0%}" for n, s in top3])
                    cv2.putText(display, match_text, (15, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.rectangle(display, (5, 5), (400, 60), (0, 0, 0), -1)
                cv2.rectangle(display, (5, 5), (400, 60), (0, 0, 255), 2)
                cv2.putText(display, "No medicine detected", (15, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # FPS
            cv2.putText(display, f"FPS: {int(fps)}", (10, h-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Press 'Q' to quit", (10, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Medicine Recognition", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def train_classifier(self, epochs=10, batch_size=8):
        """
        Train the classifier on collected medicine samples.
        Call this after adding multiple samples of each medicine.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        if not DEEP_LEARNING_AVAILABLE:
            print("Deep learning not available. Cannot train classifier.")
            return
        
        if len(self.medicine_db) < 2:
            print("Need at least 2 different medicines to train.")
            return
        
        print("\nPreparing training data...")
        
        # Prepare data
        X_train = []
        y_train = []
        class_names = list(self.medicine_db.keys())
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        for medicine_name, data in self.medicine_db.items():
            class_idx = class_to_idx[medicine_name]
            
            for sample in data['samples']:
                img_path = sample['image_path']
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert and preprocess
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                img_tensor = self.preprocess(pil_img)
                
                X_train.append(img_tensor)
                y_train.append(class_idx)
        
        if len(X_train) < 10:
            print("Not enough training samples. Need at least 10 total samples.")
            return
        
        X_train = torch.stack(X_train)
        y_train = torch.tensor(y_train, dtype=torch.long)
        
        print(f"Training on {len(X_train)} samples, {len(class_names)} classes")
        
        # Update classifier output layer
        num_classes = len(class_names)
        self.classifier.classifier[1] = nn.Linear(
            self.classifier.classifier[1].in_features,
            num_classes
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.classifier.train()
        
        print("\nTraining classifier...")
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.classifier(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(dataloader)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save model
        self.classifier.eval()
        model_path = os.path.join(self.model_folder, "medicine_classifier.pth")
        torch.save(self.classifier.state_dict(), model_path)
        
        # Save class mapping
        mapping_path = os.path.join(self.model_folder, "class_mapping.pkl")
        with open(mapping_path, 'wb') as f:
            pickle.dump({'class_names': class_names, 'class_to_idx': class_to_idx}, f)
        
        print(f"\n✓ Training complete! Model saved to {model_path}")
        print(f"Final accuracy: {accuracy:.2f}%")

    def get_statistics(self):
        """Get statistics about the medicine database."""
        stats = {
            'total_medicines': len(self.medicine_db),
            'total_samples': sum(len(d['samples']) for d in self.medicine_db.values()),
            'medicines': {}
        }
        
        for name, data in self.medicine_db.items():
            stats['medicines'][name] = {
                'samples': len(data['samples']),
                'has_embeddings': len(data.get('embeddings', [])) > 0
            }
        
        return stats

    def print_statistics(self):
        """Print database statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("MEDICINE DATABASE STATISTICS")
        print("="*50)
        print(f"Total Medicines: {stats['total_medicines']}")
        print(f"Total Samples: {stats['total_samples']}")
        print("\nMedicines:")
        print("-"*50)
        
        for name, info in stats['medicines'].items():
            emb_status = "✓" if info['has_embeddings'] else "✗"
            print(f"  {name}: {info['samples']} samples [Embeddings: {emb_status}]")
        
        print("="*50 + "\n")


def main():
    """Main function with interactive menu."""
    print("="*60)
    print("ENHANCED MEDICINE RECOGNIZER")
    print("Using Deep Learning for High Accuracy")
    print("="*60)
    
    # Initialize
    recognizer = MedicineRecognizer()
    
    # Check if deep learning is available
    if not DEEP_LEARNING_AVAILABLE:
        print("\n⚠ WARNING: Deep learning libraries not installed!")
        print("Install with: pip install ultralytics torch torchvision pillow")
        print("Running in fallback mode with limited accuracy.\n")
    
    while True:
        print("\n" + "-"*60)
        print("MENU:")
        print("-"*60)
        print("1. Add medicine reference")
        print("2. Run real-time detection")
        print("3. Train classifier (after adding samples)")
        print("4. View database statistics")
        print("5. Test on image file")
        print("6. Exit")
        print("-"*60)
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            medicine_name = input("\nEnter medicine name: ").strip()
            if medicine_name:
                print(f"\nAdding reference for: {medicine_name}")
                recognizer.add_reference(medicine_name)
            else:
                print("Invalid name!")
        
        elif choice == '2':
            if not recognizer.medicine_db:
                print("\n⚠ No medicines in database. Add references first!")
                continue
            
            print("\nStarting real-time detection...")
            print("Position medicine in front of camera")
            print("Press 'Q' to quit\n")
            recognizer.run_detection()
        
        elif choice == '3':
            if not DEEP_LEARNING_AVAILABLE:
                print("\n⚠ Deep learning not available. Cannot train.")
                continue
            
            if len(recognizer.medicine_db) < 2:
                print("\n⚠ Need at least 2 different medicines to train!")
                continue
            
            epochs = input("Enter number of epochs (default 10): ").strip()
            epochs = int(epochs) if epochs.isdigit() else 10
            
            recognizer.train_classifier(epochs=epochs)
        
        elif choice == '4':
            recognizer.print_statistics()
        
        elif choice == '5':
            filepath = input("\nEnter image file path: ").strip()
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    print("\nProcessing image...")
                    result = recognizer.identify_medicine(img)
                    
                    print(f"\nResult: {result['message']}")
                    if result['medicine']:
                        print(f"Medicine: {result['medicine']}")
                        print(f"Confidence: {result['confidence']:.1%}")
                    
                    if 'all_scores' in result:
                        print("\nAll scores:")
                        for name, score in sorted(result['all_scores'].items(), 
                                                 key=lambda x: x[1], reverse=True):
                            print(f"  {name}: {score:.1%}")
                    
                    # Show result
                    display = img.copy()
                    if result.get('extracted') is not None:
                        cv2.imshow("Query Image", img)
                        cv2.imshow("Extracted Medicine", result['extracted'])
                        print("\nPress any key to continue...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    print("Could not load image!")
            else:
                print("File not found!")
        
        elif choice == '6':
            print("\nExiting... Goodbye!")
            break
        
        else:
            print("\n⚠ Invalid option! Please select 1-6.")


if __name__ == "__main__":
    main()