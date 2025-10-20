#!/usr/bin/env python3
"""
Abnormal Behavior Recognizer

A class-based OpenCV implementation for detecting abnormal human behaviors
such as falls, heart attack symptoms, seizures, and other emergency situations
from camera input using pose estimation and movement analysis.
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
import math
from collections import deque


class AbnormalBehaviorDetector:
    """
    A class that uses OpenCV to detect abnormal human behaviors from camera input.
    This includes detecting falls, heart attack symptoms, seizures, and other emergency situations.
    """

    def __init__(self, history_size=60, alert_threshold=0.75, use_cuda=False):
        """
        Initialize the Abnormal Behavior Detector.
        
        Args:
            history_size (int): Number of frames to keep in history for analysis
            alert_threshold (float): Threshold for triggering alerts (0.0-1.0)
            use_cuda (bool): Whether to use CUDA for GPU acceleration
        """
        self.history_size = history_size
        self.alert_threshold = alert_threshold
        self.frame_history = deque(maxlen=history_size)
        self.motion_history = deque(maxlen=history_size)
        self.pose_history = deque(maxlen=history_size)
        self.alert_history = []
        self.current_status = "Normal"
        
        # Create output folder for saved alerts
        self.output_folder = "abnormal_behavior_alerts"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Setup pose estimation model
        print("Loading pose estimation model...")
        self._setup_pose_model(use_cuda)
        
        # Setup motion detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        
        print("Abnormal Behavior Detector initialized")
        
    def _setup_pose_model(self, use_cuda=False):
        """Set up the human pose estimation model."""
        # Try to load OpenPose if available (more accurate but requires additional setup)
        try:
            # Check for OpenPose installation
            from openpose import pyopenpose as op
            
            # Configuration for OpenPose
            params = dict()
            params["model_folder"] = "models/"  # Path to OpenPose models
            params["net_resolution"] = "256x256"
            params["number_people_max"] = 1  # Focus on one person for better performance
            
            # Enable GPU if requested and available
            if use_cuda:
                params["gpu"] = 0
            else:
                params["gpu"] = -1  # Use CPU
                
            # Initialize OpenPose
            self.pose_model = op.WrapperPython()
            self.pose_model.configure(params)
            self.pose_model.start()
            self.pose_type = "openpose"
            print("Using OpenPose for pose estimation")
            
        except (ImportError, ModuleNotFoundError):
            # Fall back to MediaPipe (easier to set up but less accurate for distance)
            try:
                import mediapipe as mp
                
                self.mp = mp
                self.mp_pose = mp.solutions.pose
                self.pose_model = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=2,  # 0, 1, or 2. Higher means more accurate but slower
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.pose_type = "mediapipe"
                print("Using MediaPipe for pose estimation")
                
            except (ImportError, ModuleNotFoundError):
                # Fall back to simple contour detection as last resort
                self.pose_model = None
                self.pose_type = "contour"
                print("No pose estimation library available, falling back to contour detection")
    
    def _detect_pose(self, frame):
        """
        Detect human pose in the given frame.
        
        Args:
            frame: Input frame from camera
        
        Returns:
            dict: Pose keypoints and metadata
        """
        result = {
            'keypoints': {},
            'pose_detected': False,
            'posture': 'unknown',
            'height': 0,
            'width': 0,
            'center': (0, 0)
        }
        
        # Skip small or invalid frames
        if frame is None or frame.size == 0:
            return result
            
        h, w = frame.shape[:2]
        result['height'] = h
        result['width'] = w
        
        # Different pose detection based on available library
        if self.pose_type == "openpose":
            # OpenPose detection
            datum = op.Datum()
            datum.cvInputData = frame
            self.pose_model.emplaceAndPop(op.VectorDatum([datum]))
            
            if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
                # Extract keypoints from first person
                kpts = datum.poseKeypoints[0]
                if kpts.shape[0] >= 25:  # OpenPose has 25 keypoints
                    result['pose_detected'] = True
                    
                    # Map keypoints (OpenPose format)
                    keypoint_map = {
                        'nose': kpts[0][:2] if kpts[0][2] > 0.1 else None,
                        'neck': kpts[1][:2] if kpts[1][2] > 0.1 else None,
                        'rshoulder': kpts[2][:2] if kpts[2][2] > 0.1 else None,
                        'relbow': kpts[3][:2] if kpts[3][2] > 0.1 else None,
                        'rwrist': kpts[4][:2] if kpts[4][2] > 0.1 else None,
                        'lshoulder': kpts[5][:2] if kpts[5][2] > 0.1 else None,
                        'lelbow': kpts[6][:2] if kpts[6][2] > 0.1 else None,
                        'lwrist': kpts[7][:2] if kpts[7][2] > 0.1 else None,
                        'rhip': kpts[8][:2] if kpts[8][2] > 0.1 else None,
                        'rknee': kpts[9][:2] if kpts[9][2] > 0.1 else None,
                        'rankle': kpts[10][:2] if kpts[10][2] > 0.1 else None,
                        'lhip': kpts[11][:2] if kpts[11][2] > 0.1 else None,
                        'lknee': kpts[12][:2] if kpts[12][2] > 0.1 else None,
                        'lankle': kpts[13][:2] if kpts[13][2] > 0.1 else None,
                        'reye': kpts[14][:2] if kpts[14][2] > 0.1 else None,
                        'leye': kpts[15][:2] if kpts[15][2] > 0.1 else None,
                        'rear': kpts[16][:2] if kpts[16][2] > 0.1 else None,
                        'lear': kpts[17][:2] if kpts[17][2] > 0.1 else None
                    }
                    result['keypoints'] = keypoint_map
                    
                    # Calculate center of body
                    if keypoint_map['neck'] is not None and keypoint_map['rhip'] is not None and keypoint_map['lhip'] is not None:
                        neck = np.array(keypoint_map['neck'])
                        mid_hip = (np.array(keypoint_map['rhip']) + np.array(keypoint_map['lhip'])) / 2
                        center = (neck + mid_hip) / 2
                        result['center'] = (int(center[0]), int(center[1]))
        
        elif self.pose_type == "mediapipe":
            # MediaPipe detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_model.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                result['pose_detected'] = True
                landmarks = pose_results.pose_landmarks.landmark
                mp_pose = self.mp_pose
                
                # Create mapping for easier access
                keypoint_map = {
                    'nose': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.NOSE.value], w, h),
                    'neck': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.NECK.value], w, h) if hasattr(mp_pose.PoseLandmark, 'NECK') else 
                            self._estimate_neck(landmarks, mp_pose, w, h),
                    'rshoulder': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h),
                    'relbow': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], w, h),
                    'rwrist': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], w, h),
                    'lshoulder': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h),
                    'lelbow': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], w, h),
                    'lwrist': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], w, h),
                    'rhip': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], w, h),
                    'rknee': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], w, h),
                    'rankle': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value], w, h),
                    'lhip': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], w, h),
                    'lknee': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], w, h),
                    'lankle': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value], w, h),
                    'reye': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value], w, h),
                    'leye': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_EYE.value], w, h),
                    'rear': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value], w, h),
                    'lear': self._get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value], w, h)
                }
                result['keypoints'] = keypoint_map
                
                # Calculate center of body
                if all(k in keypoint_map for k in ['lshoulder', 'rshoulder', 'lhip', 'rhip']):
                    mid_shoulder = np.mean([keypoint_map['lshoulder'], keypoint_map['rshoulder']], axis=0)
                    mid_hip = np.mean([keypoint_map['lhip'], keypoint_map['rhip']], axis=0)
                    center = np.mean([mid_shoulder, mid_hip], axis=0)
                    result['center'] = (int(center[0]), int(center[1]))
        
        else:
            # Basic contour detection as fallback
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Apply background subtraction
            fg_mask = self.fgbg.apply(gray)
            
            # Threshold and find contours
            thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (presumably the person)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Only proceed if the contour is large enough
                if cv2.contourArea(largest_contour) > 1000:
                    result['pose_detected'] = True
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Simplistic keypoint estimation
                    head_y = y + h // 8
                    shoulder_y = y + h // 4
                    hip_y = y + h // 2
                    knee_y = y + 3 * h // 4
                    foot_y = y + h
                    
                    center_x = x + w // 2
                    left_x = x + w // 4
                    right_x = x + 3 * w // 4
                    
                    # Create a basic skeleton
                    result['keypoints'] = {
                        'nose': (center_x, head_y),
                        'neck': (center_x, shoulder_y),
                        'rshoulder': (right_x, shoulder_y),
                        'relbow': (right_x, (shoulder_y + hip_y) // 2),
                        'rwrist': (right_x, hip_y),
                        'lshoulder': (left_x, shoulder_y),
                        'lelbow': (left_x, (shoulder_y + hip_y) // 2),
                        'lwrist': (left_x, hip_y),
                        'rhip': (right_x, hip_y),
                        'rknee': (right_x, knee_y),
                        'rankle': (right_x, foot_y),
                        'lhip': (left_x, hip_y),
                        'lknee': (left_x, knee_y),
                        'lankle': (left_x, foot_y)
                    }
                    
                    # Set center
                    result['center'] = (center_x, hip_y)
        
        # Determine posture if pose was detected
        if result['pose_detected']:
            result['posture'] = self._analyze_posture(result)
            
        return result
    
    def _get_landmark_point(self, landmark, frame_width, frame_height):
        """Convert MediaPipe landmark to pixel coordinates."""
        return (int(landmark.x * frame_width), int(landmark.y * frame_height))
    
    def _estimate_neck(self, landmarks, mp_pose, w, h):
        """Estimate neck position from shoulders and nose for MediaPipe."""
        # Calculate the neck position as the midpoint between shoulders, slightly higher
        right_shoulder = self._get_landmark_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
        left_shoulder = self._get_landmark_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
        
        # Find midpoint between shoulders
        mid_shoulder_x = (right_shoulder[0] + left_shoulder[0]) // 2
        mid_shoulder_y = (right_shoulder[1] + left_shoulder[1]) // 2
        
        # Move slightly up from the mid-shoulder
        neck_y = mid_shoulder_y - int((w + h) / 100)  # Adjust this value as needed
        
        return (mid_shoulder_x, neck_y)
    
    def _analyze_posture(self, pose_data):
        """
        Analyze pose to determine current posture (standing, sitting, lying, etc.)
        
        Args:
            pose_data: Dictionary containing keypoint data
        
        Returns:
            str: Detected posture
        """
        keypoints = pose_data['keypoints']
        
        # Check for minimum required keypoints
        required_points = ['neck', 'rhip', 'lhip', 'rknee', 'lknee']
        if not all(point in keypoints and keypoints[point] is not None for point in required_points):
            return "unknown"
        
        # Calculate the average height (vertical position) of hips
        hip_y = (keypoints['rhip'][1] + keypoints['lhip'][1]) / 2
        
        # Calculate the average height of ankles (if available)
        ankle_y = None
        if 'rankle' in keypoints and 'lankle' in keypoints and keypoints['rankle'] and keypoints['lankle']:
            ankle_y = (keypoints['rankle'][1] + keypoints['lankle'][1]) / 2
        
        # Calculate body orientation angle
        # Angle between vertical line and the line from mid-hip to neck
        mid_hip = ((keypoints['rhip'][0] + keypoints['lhip'][0]) / 2, 
                  (keypoints['rhip'][1] + keypoints['lhip'][1]) / 2)
        
        neck = keypoints['neck']
        dx = mid_hip[0] - neck[0]
        dy = mid_hip[1] - neck[1]
        angle = abs(math.degrees(math.atan2(dx, dy)))
        
        # Determine posture based on angle and positions
        if angle > 45:  # Body is more horizontal than vertical
            return "lying"
        elif ankle_y is not None:
            # Calculate the ratio of hip-to-ankle distance vs image height
            hip_ankle_ratio = (ankle_y - hip_y) / pose_data['height']
            
            if hip_ankle_ratio < 0.2:  # Legs are bent/tucked, likely sitting
                return "sitting"
            else:
                return "standing"
        else:
            # Fallback when ankles aren't detected
            # Use knee position relative to hip
            knee_y = (keypoints['rknee'][1] + keypoints['lknee'][1]) / 2
            hip_knee_ratio = (knee_y - hip_y) / pose_data['height']
            
            if hip_knee_ratio < 0.15:  # Small distance from hip to knee
                return "sitting"
            else:
                return "standing"
    
    def _detect_motion(self, frame, prev_frame):
        """
        Detect and quantify motion between frames.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame
        
        Returns:
            dict: Motion metrics
        """
        if prev_frame is None or frame.shape != prev_frame.shape:
            return {'motion_score': 0, 'motion_areas': [], 'dominant_direction': None}
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Get mean magnitude as overall motion score
        motion_score = np.mean(magnitude)
        
        # Find areas with significant motion
        motion_threshold = np.mean(magnitude) + np.std(magnitude)
        motion_mask = magnitude > motion_threshold
        
        # Create contours from motion mask
        motion_areas = []
        if np.any(motion_mask):
            motion_uint8 = motion_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(motion_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Adjust threshold as needed
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_areas.append({
                        'x': x, 'y': y, 'width': w, 'height': h, 
                        'area': area, 'center': (x + w//2, y + h//2)
                    })
        
        # Calculate dominant motion direction
        dominant_direction = None
        if np.any(motion_mask):
            # Calculate histogram of angles in degrees (0-360)
            angle_deg = np.degrees(angle)
            hist, _ = np.histogram(angle_deg[motion_mask], bins=8, range=(0, 360))
            
            # Get dominant direction
            max_idx = np.argmax(hist)
            bin_size = 360 / 8
            dominant_angle = max_idx * bin_size + bin_size / 2
            
            # Convert angle to cardinal direction
            directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
            direction_idx = int(((dominant_angle + 22.5) % 360) / 45)
            dominant_direction = directions[direction_idx]
        
        return {
            'motion_score': float(motion_score),
            'motion_areas': motion_areas,
            'dominant_direction': dominant_direction
        }
    
    def _analyze_behavior(self, frame_data):
        """
        Analyze the frame data to detect abnormal behaviors.
        
        Args:
            frame_data: Dictionary containing pose and motion data
        
        Returns:
            dict: Analysis results with detected behaviors and confidence
        """
        results = {
            'behaviors': [],
            'confidence': 0.0,
            'status': 'Normal',
            'details': {}
        }
        
        # If no pose detected, can't analyze further
        if not frame_data['pose']['pose_detected']:
            return results
        
        # Get current pose and motion data
        pose = frame_data['pose']
        motion = frame_data['motion']
        posture = pose['posture']
        
        # Get history for analysis if available
        pose_history = list(self.pose_history) if self.pose_history else []
        motion_history = list(self.motion_history) if self.motion_history else []
        
        # 1. Fall detection
        fall_detected, fall_confidence = self._detect_fall(frame_data, pose_history, motion_history)
        if fall_detected:
            results['behaviors'].append('fall')
            results['details']['fall'] = fall_confidence
            results['confidence'] = max(results['confidence'], fall_confidence)
        
        # 2. Heart attack / chest pain symptoms
        heart_attack_detected, heart_attack_confidence = self._detect_heart_attack(frame_data, pose_history, motion_history)
        if heart_attack_detected:
            results['behaviors'].append('heart_attack')
            results['details']['heart_attack'] = heart_attack_confidence
            results['confidence'] = max(results['confidence'], heart_attack_confidence)
        
        # 3. Seizure detection
        seizure_detected, seizure_confidence = self._detect_seizure(frame_data, pose_history, motion_history)
        if seizure_detected:
            results['behaviors'].append('seizure')
            results['details']['seizure'] = seizure_confidence
            results['confidence'] = max(results['confidence'], seizure_confidence)
        
        # 4. Unusual motion detection (erratic/unstable movement)
        unusual_motion, unusual_confidence = self._detect_unusual_motion(frame_data, pose_history, motion_history)
        if unusual_motion:
            results['behaviors'].append('unusual_motion')
            results['details']['unusual_motion'] = unusual_confidence
            results['confidence'] = max(results['confidence'], unusual_confidence)
        
        # 5. Unconsciousness/immobility detection
        immobile_detected, immobile_confidence = self._detect_immobility(frame_data, pose_history, motion_history)
        if immobile_detected:
            results['behaviors'].append('immobile')
            results['details']['immobile'] = immobile_confidence
            results['confidence'] = max(results['confidence'], immobile_confidence)
        
        # Set overall status based on behaviors detected
        if results['behaviors']:
            results['status'] = 'Abnormal'
            # Sort behaviors by confidence
            results['behaviors'] = sorted(results['behaviors'], 
                                         key=lambda b: results['details'].get(b, 0), 
                                         reverse=True)
        else:
            results['status'] = 'Normal'
        
        return results
    
    def _detect_fall(self, frame_data, pose_history, motion_history):
        """
        Detect if a person has fallen.
        
        Args:
            frame_data: Current frame analysis
            pose_history: Historical pose data
            motion_history: Historical motion data
            
        Returns:
            tuple: (detected, confidence)
        """
        # If no sufficient history, can't detect fall
        if len(pose_history) < 5:
            return False, 0.0
        
        pose = frame_data['pose']
        motion = frame_data['motion']
        
        # Fall indicators
        indicators = []
        
        # 1. Check for sudden posture change to lying position
        recent_postures = [p['posture'] for p in pose_history[-10:] if 'posture' in p]
        if pose['posture'] == 'lying' and 'standing' in recent_postures:
            posture_indicator = 0.8
            indicators.append(posture_indicator)
        
        # 2. Check for sudden downward motion
        if motion_history and motion['motion_score'] > 0.5:
            recent_directions = [m['dominant_direction'] for m in motion_history[-5:] 
                              if 'dominant_direction' in m and m['dominant_direction']]
            
            if motion['dominant_direction'] in ['S', 'SE', 'SW'] and 'S' in recent_directions:
                direction_indicator = 0.7
                indicators.append(direction_indicator)
        
        # 3. Check for high motion score followed by low motion (impact then stillness)
        if len(motion_history) > 5:
            recent_motion = [m['motion_score'] for m in motion_history[-5:]]
            max_recent_motion = max(recent_motion) if recent_motion else 0
            current_motion = motion['motion_score']
            
            if max_recent_motion > 1.0 and current_motion < 0.2:
                motion_pattern_indicator = 0.6
                indicators.append(motion_pattern_indicator)
        
        # 4. Check body orientation change
        if len(pose_history) > 3 and 'keypoints' in pose and pose['keypoints']:
            # Calculate the vertical orientation change
            if all(k in pose['keypoints'] for k in ['neck', 'lhip', 'rhip']):
                current_mid_hip = ((pose['keypoints']['lhip'][1] + pose['keypoints']['rhip'][1]) / 2)
                current_neck = pose['keypoints']['neck'][1]
                current_vertical_dist = abs(current_mid_hip - current_neck)
                
                # Get previous measurements
                prev_measurements = []
                for p in pose_history[-5:-1]:
                    if 'keypoints' in p and p['keypoints']:
                        if all(k in p['keypoints'] for k in ['neck', 'lhip', 'rhip']):
                            prev_mid_hip = ((p['keypoints']['lhip'][1] + p['keypoints']['rhip'][1]) / 2)
                            prev_neck = p['keypoints']['neck'][1]
                            prev_measurements.append(abs(prev_mid_hip - prev_neck))
                
                if prev_measurements:
                    avg_prev = sum(prev_measurements) / len(prev_measurements)
                    if avg_prev > 0 and current_vertical_dist / avg_prev < 0.5:
                        orientation_change = 0.9  # Significant vertical compression - strong fall indicator
                        indicators.append(orientation_change)
        
        # Calculate overall confidence
        if indicators:
            confidence = min(1.0, sum(indicators) / len(indicators) * 1.2)  # Scale up slightly but cap at 1.0
            return confidence > self.alert_threshold, confidence
        
        return False, 0.0
    
    def _detect_heart_attack(self, frame_data, pose_history, motion_history):
        """
        Detect symptoms that might indicate a heart attack.
        
        Args:
            frame_data: Current frame analysis
            pose_history: Historical pose data
            motion_history: Historical motion data
            
        Returns:
            tuple: (detected, confidence)
        """
        # If no sufficient history, can't detect patterns
        if len(pose_history) < 10:
            return False, 0.0
        
        pose = frame_data['pose']
        motion = frame_data['motion']
        
        # Heart attack indicators
        indicators = []
        
        # 1. Check for hand to chest movement
        if 'keypoints' in pose and pose['keypoints']:
            keypoints = pose['keypoints']
            if all(k in keypoints for k in ['rwrist', 'lwrist', 'neck', 'rshoulder', 'lshoulder']):
                # Calculate chest area boundaries
                chest_top = keypoints['neck'][1]
                chest_bottom = (keypoints['rshoulder'][1] + keypoints['lshoulder'][1]) / 2 + 30  # Estimate
                chest_left = keypoints['rshoulder'][0] + (keypoints['lshoulder'][0] - keypoints['rshoulder'][0]) * 0.3
                chest_right = keypoints['rshoulder'][0] + (keypoints['lshoulder'][0] - keypoints['rshoulder'][0]) * 0.7
                
                # Check if either hand is near chest
                r_wrist = keypoints['rwrist']
                l_wrist = keypoints['lwrist']
                
                r_hand_on_chest = (chest_left <= r_wrist[0] <= chest_right and 
                                  chest_top <= r_wrist[1] <= chest_bottom)
                l_hand_on_chest = (chest_left <= l_wrist[0] <= chest_right and 
                                  chest_top <= l_wrist[1] <= chest_bottom)
                
                if r_hand_on_chest or l_hand_on_chest:
                    # Check if this is persistent
                    persistent_count = 0
                    for p in pose_history[-10:]:
                        if 'keypoints' in p and p['keypoints']:
                            if all(k in p['keypoints'] for k in ['rwrist', 'lwrist', 'neck', 'rshoulder', 'lshoulder']):
                                p_chest_top = p['keypoints']['neck'][1]
                                p_chest_bottom = (p['keypoints']['rshoulder'][1] + p['keypoints']['lshoulder'][1]) / 2 + 30
                                p_chest_left = p['keypoints']['rshoulder'][0] + (p['keypoints']['lshoulder'][0] - p['keypoints']['rshoulder'][0]) * 0.3
                                p_chest_right = p['keypoints']['rshoulder'][0] + (p['keypoints']['lshoulder'][0] - p['keypoints']['rshoulder'][0]) * 0.7
                                
                                p_r_wrist = p['keypoints']['rwrist']
                                p_l_wrist = p['keypoints']['lwrist']
                                
                                p_r_hand_on_chest = (p_chest_left <= p_r_wrist[0] <= p_chest_right and 
                                                   p_chest_top <= p_r_wrist[1] <= p_chest_bottom)
                                p_l_hand_on_chest = (p_chest_left <= p_l_wrist[0] <= p_chest_right and 
                                                   p_chest_top <= p_l_wrist[1] <= p_chest_bottom)
                                
                                if p_r_hand_on_chest or p_l_hand_on_chest:
                                    persistent_count += 1
                    
                    if persistent_count >= 6:  # Hand on chest for 60% of recent frames
                        chest_grabbing = 0.7
                        indicators.append(chest_grabbing)
        
        # 2. Check for hunched posture
        if 'keypoints' in pose and pose['keypoints']:
            keypoints = pose['keypoints']
            if all(k in keypoints for k in ['neck', 'lshoulder', 'rshoulder', 'lhip', 'rhip']):
                # Calculate angle of upper body (neck to mid-hip)
                mid_shoulder = ((keypoints['lshoulder'][0] + keypoints['rshoulder'][0]) / 2,
                              (keypoints['lshoulder'][1] + keypoints['rshoulder'][1]) / 2)
                mid_hip = ((keypoints['lhip'][0] + keypoints['rhip'][0]) / 2,
                          (keypoints['lhip'][1] + keypoints['rhip'][1]) / 2)
                neck = keypoints['neck']
                
                # Angle between vertical and neck-to-mid-shoulder line
                dx1 = mid_shoulder[0] - neck[0]
                dy1 = mid_shoulder[1] - neck[1]
                upper_angle = abs(math.degrees(math.atan2(dx1, dy1)))
                
                # Angle between vertical and mid-shoulder-to-mid-hip line
                dx2 = mid_hip[0] - mid_shoulder[0]
                dy2 = mid_hip[1] - mid_shoulder[1]
                lower_angle = abs(math.degrees(math.atan2(dx2, dy2)))
                
                # A hunched posture has the upper body leaning forward
                if upper_angle > 20 and lower_angle < 10:
                    hunched_posture = 0.6
                    indicators.append(hunched_posture)
        
        # 3. Detect frequent position changes (restlessness)
        if len(motion_history) > 15:
            motion_scores = [m['motion_score'] for m in motion_history[-15:]]
            motion_variations = np.std(motion_scores)
            if motion_variations > 0.3 and np.mean(motion_scores) > 0.2:
                restlessness = 0.5
                indicators.append(restlessness)
        
        # 4. Detect face touching (might indicate distress)
        if 'keypoints' in pose and pose['keypoints']:
            keypoints = pose['keypoints']
            if all(k in keypoints for k in ['rwrist', 'lwrist', 'nose', 'reye', 'leye']):
                # Define face area
                if keypoints['reye'] and keypoints['leye']:
                    face_left = min(keypoints['reye'][0], keypoints['leye'][0]) - 30
                    face_right = max(keypoints['reye'][0], keypoints['leye'][0]) + 30
                    face_top = min(keypoints['reye'][1], keypoints['leye'][1]) - 30
                    face_bottom = keypoints['nose'][1] + 40
                    
                    # Check if hands near face
                    r_hand_on_face = (face_left <= keypoints['rwrist'][0] <= face_right and 
                                    face_top <= keypoints['rwrist'][1] <= face_bottom)
                    l_hand_on_face = (face_left <= keypoints['lwrist'][0] <= face_right and 
                                    face_top <= keypoints['lwrist'][1] <= face_bottom)
                    
                    if r_hand_on_face or l_hand_on_face:
                        face_touching = 0.4
                        indicators.append(face_touching)
        
        # Calculate overall confidence
        if indicators:
            confidence = min(1.0, sum(indicators) / 2.2)  # Adjusted scale, max would be ~1.0
            return confidence > self.alert_threshold, confidence
        
        return False, 0.0
    
    def _detect_seizure(self, frame_data, pose_history, motion_history):
        """
        Detect seizure-like movements.
        
        Args:
            frame_data: Current frame analysis
            pose_history: Historical pose data
            motion_history: Historical motion data
            
        Returns:
            tuple: (detected, confidence)
        """
        # Need sufficient history for reliable detection
        if len(motion_history) < 15:
            return False, 0.0
            
        indicators = []
        
        # 1. Rapid, rhythmic motion
        recent_motion = [m['motion_score'] for m in motion_history[-15:]]
        if len(recent_motion) >= 15:
            # Check for high variance in motion
            motion_std = np.std(recent_motion)
            motion_mean = np.mean(recent_motion)
            
            if motion_std > 0.4 and motion_mean > 0.5:
                # Check for rhythmic patterns using autocorrelation
                autocorr = np.correlate(recent_motion, recent_motion, mode='full')
                autocorr = autocorr[len(autocorr)//2:] # Take only the positive lags
                
                # Look for peaks in autocorrelation (indicates repeating pattern)
                if len(autocorr) > 5:
                    peaks = []
                    for i in range(2, len(autocorr)-2):
                        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i-2] and \
                           autocorr[i] > autocorr[i+1] and autocorr[i] > autocorr[i+2]:
                            peaks.append(i)
                    
                    # If we find regularly spaced peaks, it suggests rhythmic motion
                    if len(peaks) >= 2:
                        rhythmic_indicator = 0.8
                        indicators.append(rhythmic_indicator)
        
        # 2. Erratic limb movements
        if len(pose_history) >= 10:
            # Track wrist positions over time
            r_wrist_positions = []
            l_wrist_positions = []
            
            for p in pose_history[-10:]:
                if 'keypoints' in p and p['keypoints']:
                    if 'rwrist' in p['keypoints'] and p['keypoints']['rwrist']:
                        r_wrist_positions.append(p['keypoints']['rwrist'])
                    if 'lwrist' in p['keypoints'] and p['keypoints']['lwrist']:
                        l_wrist_positions.append(p['keypoints']['lwrist'])
            
            # Calculate displacement variance
            r_wrist_var = 0
            l_wrist_var = 0
            
            if len(r_wrist_positions) >= 5:
                r_wrist_x = [pos[0] for pos in r_wrist_positions]
                r_wrist_y = [pos[1] for pos in r_wrist_positions]
                r_wrist_var = np.std(r_wrist_x) + np.std(r_wrist_y)
            
            if len(l_wrist_positions) >= 5:
                l_wrist_x = [pos[0] for pos in l_wrist_positions]
                l_wrist_y = [pos[1] for pos in l_wrist_positions]
                l_wrist_var = np.std(l_wrist_x) + np.std(l_wrist_y)
            
            # High variance in wrist position indicates jerky movements
            max_wrist_var = max(r_wrist_var, l_wrist_var)
            if max_wrist_var > 50:  # Threshold for erratic movement
                erratic_indicator = min(1.0, max_wrist_var / 100)
                indicators.append(erratic_indicator)
        
        # 3. Detect body shaking
        if len(pose_history) >= 10 and 'keypoints' in frame_data['pose'] and frame_data['pose']['keypoints']:
            # Track neck position (or other stable body part)
            neck_positions = []
            
            for p in pose_history[-10:]:
                if 'keypoints' in p and p['keypoints'] and 'neck' in p['keypoints'] and p['keypoints']['neck']:
                    neck_positions.append(p['keypoints']['neck'])
            
            if len(neck_positions) >= 5:
                # Calculate variance in neck position
                neck_x = [pos[0] for pos in neck_positions]
                neck_y = [pos[1] for pos in neck_positions]
                neck_var = np.std(neck_x) + np.std(neck_y)
                
                # High variance without large displacement indicates shaking
                if neck_var > 15:
                    # Calculate max displacement
                    max_disp_x = max(neck_x) - min(neck_x)
                    max_disp_y = max(neck_y) - min(neck_y)
                    max_disp = max(max_disp_x, max_disp_y)
                    
                    # High variance with small displacement is characteristic of shaking
                    if max_disp < 100 and neck_var > 20:
                        shaking_indicator = min(1.0, neck_var / 40)
                        indicators.append(shaking_indicator)
        
        # Calculate overall confidence
        if indicators:
            confidence = min(1.0, sum(indicators) / 2.0)  # Adjusted scale
            return confidence > self.alert_threshold, confidence
        
        return False, 0.0
    
    def _detect_unusual_motion(self, frame_data, pose_history, motion_history):
        """
        Detect unusual/abnormal motion patterns.
        
        Args:
            frame_data: Current frame analysis
            pose_history: Historical pose data
            motion_history: Historical motion data
            
        Returns:
            tuple: (detected, confidence)
        """
        # Need sufficient history
        if len(pose_history) < 10 or len(motion_history) < 10:
            return False, 0.0
            
        indicators = []
        
        # 1. Detect staggering motion (side-to-side swaying while standing)
        if frame_data['pose']['posture'] == 'standing':
            # Track center position over time
            center_positions = []
            
            for p in pose_history[-10:]:
                if 'center' in p:
                    center_positions.append(p['center'])
            
            if len(center_positions) >= 8:
                # Calculate horizontal variance vs vertical variance
                center_x = [pos[0] for pos in center_positions]
                center_y = [pos[1] for pos in center_positions]
                x_var = np.var(center_x)
                y_var = np.var(center_y)
                
                # Staggering has high horizontal variance compared to vertical
                if x_var > 2 * y_var and x_var > 100:
                    stagger_indicator = min(1.0, x_var / 300)
                    indicators.append(stagger_indicator)
        
        # 2. Detect spinning/turning motion
        if len(motion_history) >= 8:
            # Count direction changes
            directions = [m['dominant_direction'] for m in motion_history[-8:] 
                       if 'dominant_direction' in m and m['dominant_direction']]
            
            if len(directions) >= 6:
                # Count how many times direction changes by more than 90 degrees
                direction_changes = 0
                dir_to_angle = {'N': 90, 'NE': 45, 'E': 0, 'SE': 315, 'S': 270, 'SW': 225, 'W': 180, 'NW': 135}
                
                for i in range(1, len(directions)):
                    if directions[i-1] in dir_to_angle and directions[i] in dir_to_angle:
                        angle1 = dir_to_angle[directions[i-1]]
                        angle2 = dir_to_angle[directions[i]]
                        diff = min((angle1 - angle2) % 360, (angle2 - angle1) % 360)
                        if diff >= 90:
                            direction_changes += 1
                
                if direction_changes >= 3:  # Multiple direction changes
                    spinning_indicator = min(1.0, direction_changes / 4)
                    indicators.append(spinning_indicator)
        
        # 3. Detect unusually fast motion
        if len(motion_history) >= 5:
            recent_motion = [m['motion_score'] for m in motion_history[-5:]]
            avg_motion = np.mean(recent_motion)
            
            if avg_motion > 1.5:  # Threshold for unusually fast motion
                fast_motion_indicator = min(1.0, avg_motion / 2)
                indicators.append(fast_motion_indicator)
        
        # 4. Detect uncoordinated limb movements
        if len(pose_history) >= 5:
            # Check correlation between different limb movements
            r_arm_positions = []
            l_arm_positions = []
            
            for p in pose_history[-5:]:
                if 'keypoints' in p and p['keypoints']:
                    if all(k in p['keypoints'] for k in ['relbow', 'rwrist']):
                        r_arm_pos = (p['keypoints']['relbow'][0] - p['keypoints']['rwrist'][0],
                                    p['keypoints']['relbow'][1] - p['keypoints']['rwrist'][1])
                        r_arm_positions.append(r_arm_pos)
                    
                    if all(k in p['keypoints'] for k in ['lelbow', 'lwrist']):
                        l_arm_pos = (p['keypoints']['lelbow'][0] - p['keypoints']['lwrist'][0],
                                    p['keypoints']['lelbow'][1] - p['keypoints']['lwrist'][1])
                        l_arm_positions.append(l_arm_pos)
            
            if len(r_arm_positions) >= 5 and len(l_arm_positions) >= 5:
                # Calculate movement correlation
                r_arm_x = [pos[0] for pos in r_arm_positions]
                r_arm_y = [pos[1] for pos in r_arm_positions]
                l_arm_x = [pos[0] for pos in l_arm_positions]
                l_arm_y = [pos[1] for pos in l_arm_positions]
                
                # Correlation coefficient between right and left arm movements
                if np.std(r_arm_x) > 0 and np.std(l_arm_x) > 0:
                    corr_x = np.corrcoef(r_arm_x, l_arm_x)[0, 1]
                    corr_y = np.corrcoef(r_arm_y, l_arm_y)[0, 1]
                    
                    # Low or negative correlation indicates uncoordinated movement
                    avg_corr = (corr_x + corr_y) / 2
                    if avg_corr < 0.2 and np.std(r_arm_x) > 10 and np.std(l_arm_x) > 10:
                        uncoord_indicator = min(1.0, (0.2 - avg_corr) * 2)
                        indicators.append(uncoord_indicator)
        
        # Calculate overall confidence
        if indicators:
            confidence = min(1.0, sum(indicators) / 2.0)  # Adjusted scale
            return confidence > self.alert_threshold, confidence
        
        return False, 0.0
    
    def _detect_immobility(self, frame_data, pose_history, motion_history):
        """
        Detect prolonged immobility or unconsciousness.
        
        Args:
            frame_data: Current frame analysis
            pose_history: Historical pose data
            motion_history: Historical motion data
            
        Returns:
            tuple: (detected, confidence)
        """
        # Need sufficient history
        if len(motion_history) < 20:
            return False, 0.0
            
        indicators = []
        
        # 1. Check for prolonged low motion
        recent_motion = [m['motion_score'] for m in motion_history[-20:]]
        avg_motion = np.mean(recent_motion)
        max_motion = max(recent_motion)
        
        if avg_motion < 0.1 and max_motion < 0.2:
            # Person has been very still for a while
            immobile_indicator = 0.6
            indicators.append(immobile_indicator)
            
            # Higher confidence if lying down and immobile
            if frame_data['pose']['posture'] == 'lying':
                lying_immobile = 0.3  # Additional indicator
                indicators.append(lying_immobile)
        
        # 2. Check for lack of response to motion around them
        # This would require scene understanding/multiple person detection
        # Simplified version: If there is scene motion but person is static
        if 'motion_areas' in frame_data['motion'] and frame_data['motion']['motion_areas']:
            scene_motion = len(frame_data['motion']['motion_areas'])
            if scene_motion > 2 and avg_motion < 0.05:
                unresponsive_indicator = 0.4
                indicators.append(unresponsive_indicator)
        
        # Calculate overall confidence
        if indicators:
            confidence = min(1.0, sum(indicators) / 1.3)  # Adjusted scale
            return confidence > self.alert_threshold, confidence
        
        return False, 0.0
    
    def run_detection(self, camera_index=0, display=True, alert_callback=None):
        """
        Run continuous behavior detection from camera feed.
        
        Args:
            camera_index (int): Camera device index
            display (bool): Whether to show visualization window
            alert_callback (callable): Function to call when abnormal behavior is detected,
                                     will be passed the analysis results
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Abnormal behavior detection started. Press 'q' to quit.")
        
        # For FPS calculation
        prev_frame_time = 0
        new_frame_time = 0
        
        # Initialize previous frame
        ret, prev_frame = cap.read()
        if not ret:
            print("Error: Failed to capture initial frame")
            cap.release()
            return
        
        # Alert state management
        current_alert_id = None
        alert_frames = []
        alert_duration = 0
        alert_cooldown = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {int(fps)}"
            
            # Process frame (make a copy to avoid modifying the original)
            display_frame = frame.copy()
            
            # Detect pose
            pose_data = self._detect_pose(frame)
            
            # Detect motion
            motion_data = self._detect_motion(frame, prev_frame)
            prev_frame = frame.copy()
            
            # Update history
            self.frame_history.append(frame.copy())
            self.pose_history.append(pose_data)
            self.motion_history.append(motion_data)
            
            # Analyze behavior
            frame_data = {
                'pose': pose_data,
                'motion': motion_data,
                'timestamp': new_frame_time
            }
            
            analysis = self._analyze_behavior(frame_data)
            
            # Update current status
            self.current_status = analysis['status']
            
            # Handle alert state
            if analysis['status'] == 'Abnormal' and analysis['confidence'] > self.alert_threshold:
                # We have a detected abnormal behavior
                alert_cooldown = 0  # Reset cooldown
                
                if current_alert_id is None:
                    # New alert
                    current_alert_id = int(time.time())
                    alert_frames = []
                    alert_duration = 0
                    print(f"⚠️ ALERT: Detected {', '.join(analysis['behaviors'])} - Confidence: {analysis['confidence']:.2f}")
                    
                    # Call callback if provided
                    if alert_callback:
                        alert_callback(analysis)
                
                # Add frame to alert sequence
                alert_frames.append(frame.copy())
                alert_duration += 1
                
                # Save alert after accumulating enough frames or if very high confidence
                if alert_duration >= 15 or analysis['confidence'] > 0.9:
                    self._save_alert(current_alert_id, analysis, alert_frames)
                    alert_frames = []  # Clear frames after saving
            
            else:
                # No alert or alert resolved
                if current_alert_id is not None:
                    alert_cooldown += 1
                    
                    # End alert after cooldown
                    if alert_cooldown > 10:  # Wait for 10 frames before ending alert
                        if alert_frames:  # Save any remaining frames
                            self._save_alert(current_alert_id, analysis, alert_frames)
                        
                        print(f"Alert ended: ID {current_alert_id}")
                        current_alert_id = None
                        alert_frames = []
                        alert_duration = 0
            
            # Display results on frame
            if display:
                # Draw pose keypoints and skeleton
                self._draw_pose(display_frame, pose_data)
                
                # Draw motion vectors
                self._draw_motion(display_frame, motion_data)
                
                # Draw status info
                cv2.putText(display_frame, f"Status: {self.current_status}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            (0, 255, 0) if self.current_status == 'Normal' else (0, 0, 255), 2)
                
                if analysis['behaviors']:
                    behavior_text = f"Detected: {', '.join(analysis['behaviors'])}"
                    cv2.putText(display_frame, behavior_text, (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Confidence: {analysis['confidence']:.2f}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(display_frame, f"Posture: {pose_data['posture']}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Add FPS counter
                cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('Abnormal Behavior Detector', display_frame)
                
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Clean up
        cap.release()
        if display:
            cv2.destroyAllWindows()
    
    def _draw_pose(self, frame, pose_data):
        """Draw pose skeleton on frame."""
        if not pose_data['pose_detected']:
            return
            
        keypoints = pose_data['keypoints']
        
        # Define connections for skeleton
        connections = [
            ('nose', 'neck'),
            ('neck', 'rshoulder'),
            ('neck', 'lshoulder'),
            ('rshoulder', 'relbow'),
            ('relbow', 'rwrist'),
            ('lshoulder', 'lelbow'),
            ('lelbow', 'lwrist'),
            ('neck', 'rhip'),
            ('neck', 'lhip'),
            ('rhip', 'rknee'),
            ('rknee', 'rankle'),
            ('lhip', 'lknee'),
            ('lknee', 'lankle'),
            ('reye', 'leye')
        ]
        
        # Draw keypoints
        for key, point in keypoints.items():
            if point is not None:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1)
        
        # Draw skeleton
        for connection in connections:
            start, end = connection
            if start in keypoints and end in keypoints and keypoints[start] and keypoints[end]:
                start_point = (int(keypoints[start][0]), int(keypoints[start][1]))
                end_point = (int(keypoints[end][0]), int(keypoints[end][1]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    def _draw_motion(self, frame, motion_data):
        """Draw motion information on frame."""
        # Draw motion areas
        for area in motion_data.get('motion_areas', []):
            cv2.rectangle(frame, (area['x'], area['y']), 
                         (area['x'] + area['width'], area['y'] + area['height']), 
                         (255, 0, 0), 2)
        
        # Draw motion direction if available
        direction = motion_data.get('dominant_direction')
        if direction:
            h, w = frame.shape[:2]
            center = (w//2, h//2)
            direction_map = {
                'N': (0, -1), 'NE': (1, -1), 'E': (1, 0), 'SE': (1, 1),
                'S': (0, 1), 'SW': (-1, 1), 'W': (-1, 0), 'NW': (-1, -1)
            }
            
            if direction in direction_map:
                dir_x, dir_y = direction_map[direction]
                arrow_len = min(w, h) // 8
                end_x = center[0] + dir_x * arrow_len
                end_y = center[1] + dir_y * arrow_len
                
                # Draw only if motion score is significant
                if motion_data['motion_score'] > 0.2:
                    cv2.arrowedLine(frame, center, (end_x, end_y), (255, 0, 0), 2)
    
    def _save_alert(self, alert_id, analysis, frames):
        """Save alert information and frames."""
        # Create directory for this alert
        alert_dir = os.path.join(self.output_folder, f"alert_{alert_id}")
        if not os.path.exists(alert_dir):
            os.makedirs(alert_dir)
        
        # Save alert metadata
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        meta_file = os.path.join(alert_dir, "metadata.txt")
        
        with open(meta_file, 'w') as f:
            f.write(f"Alert ID: {alert_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Behaviors: {', '.join(analysis['behaviors'])}\n")
            f.write(f"Confidence: {analysis['confidence']:.2f}\n")
            f.write(f"Details: {analysis['details']}\n")
        
        # Save frames
        for i, frame in enumerate(frames):
            frame_file = os.path.join(alert_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(frame_file, frame)
        
        print(f"Alert saved: {alert_dir}")
        
        # Add to alert history
        self.alert_history.append({
            'id': alert_id,
            'timestamp': timestamp,
            'behaviors': analysis['behaviors'],
            'confidence': analysis['confidence'],
            'frame_count': len(frames)
        })
    
    def process_frame(self, frame):
        """
        Process a single frame and return behavior analysis results.
        
        Args:
            frame: Input frame from camera
        
        Returns:
            dict: Analysis results with behavior type, confidence, and alert status
        """
        # Initialize previous frame if needed
        if not hasattr(self, 'prev_frame') or self.prev_frame is None:
            self.prev_frame = frame.copy()
            return {
                'behavior_type': 'Normal',
                'confidence': 0,
                'alert': False,
                'details': 'Initializing...'
            }
        
        # Detect pose
        pose_data = self._detect_pose(frame)
        
        # Detect motion
        motion_data = self._detect_motion(frame, self.prev_frame)
        self.prev_frame = frame.copy()
        
        # Update history
        self.frame_history.append(frame.copy())
        self.pose_history.append(pose_data)
        self.motion_history.append(motion_data)
        
        # Analyze behavior
        frame_data = {
            'pose': pose_data,
            'motion': motion_data,
            'timestamp': time.time()
        }
        
        analysis = self._analyze_behavior(frame_data)
        
        # Update current status
        self.current_status = analysis['status']
        
        # Prepare result
        result = {
            'behavior_type': ', '.join(analysis['behaviors']) if analysis['behaviors'] else 'Normal',
            'confidence': analysis['confidence'],
            'alert': analysis['status'] == 'Abnormal' and analysis['confidence'] > self.alert_threshold,
            'details': analysis['details'],
            'pose_data': pose_data,
            'motion_data': motion_data
        }
        
        return result
    
    def get_annotated_frame(self, frame, result):
        """
        Get an annotated version of the frame with detection results.
        
        Args:
            frame: Original frame
            result: Result from process_frame()
        
        Returns:
            numpy.ndarray: Annotated frame
        """
        annotated = frame.copy()
        
        # Draw pose if available
        if 'pose_data' in result:
            self._draw_pose(annotated, result['pose_data'])
        
        # Draw motion if available
        if 'motion_data' in result:
            self._draw_motion(annotated, result['motion_data'])
        
        # Display status
        status_color = (0, 0, 255) if result['alert'] else (0, 255, 0)
        status_text = f"Status: {result['behavior_type']}"
        
        cv2.putText(
            annotated,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )
        
        if result['confidence'] > 0:
            cv2.putText(
                annotated,
                f"Confidence: {result['confidence']:.2%}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )
        
        # Add alert banner if needed
        if result['alert']:
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w, 100), (0, 0, 255), -1)
            cv2.putText(
                annotated,
                "⚠️ ABNORMAL BEHAVIOR DETECTED",
                (w//2 - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                annotated,
                result['behavior_type'],
                (w//2 - 150, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return annotated


def main():
    """Main function to demonstrate the AbnormalBehaviorDetector class."""
    print("Abnormal Behavior Detector")
    print("=========================")
    
    # Create the detector
    detector = AbnormalBehaviorDetector()
    
    # Define a simple alert callback function
    def alert_callback(analysis):
        behaviors = ", ".join(analysis['behaviors'])
        print(f"⚠️ ALERT CALLBACK: {behaviors} (Confidence: {analysis['confidence']:.2f})")
    
    # Run the detection loop
    print("\nStarting abnormal behavior detection. Press 'q' to quit.")
    detector.run_detection(alert_callback=alert_callback)


if __name__ == "__main__":
    main()