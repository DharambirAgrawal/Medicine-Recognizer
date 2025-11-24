#!/usr/bin/env python3
"""
AAMA (AI-Assisted Medical Assistant) GUI Application

A comprehensive GUI for medicine recognition and abnormal behavior detection.
Uses tkinter for the interface with integrated camera controls.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import queue
from datetime import datetime
import os
import sys

# Import our detection modules
try:
    from medicine_recognizer import MedicineRecognizer
    from abnormal_behavior_detector import AbnormalBehaviorDetector
    from medicine_counter import MedicineCounter
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure medicine_recognizer.py, abnormal_behavior_detector.py, and medicine_counter.py are in the same directory")
    sys.exit(1)


class AAMAApp:
    """Main application class for AAMA GUI."""
    
    def __init__(self, root):
        """Initialize the AAMA application."""
        self.root = root
        self.root.title("AAMA - AI-Assisted Medical Assistant")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # State variables
        self.camera_active = False
        self.current_mode = None  # 'medicine' or 'behavior'
        self.camera_index = 0
        self.video_capture = None
        self.stop_thread = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_preview_job = None
        self.capture_last_frame = None
        self.capture_medicine_name = None
        self.capture_countdown = 0
        
        # Initialize detectors
        self.medicine_recognizer = None
        self.behavior_detector = None
        self.medicine_counter = None
        
        # Setup GUI
        self._setup_gui()
        
        # Initialize detectors in background
        # self.root.after(100, self._initialize_detectors)
        self.root.after(100, lambda: threading.Thread(target=self._initialize_detectors, daemon=True).start())

        
    def _setup_gui(self):
        """Setup the GUI layout."""
        # Title bar
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X, side=tk.TOP)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🏥 AAMA - AI-Assisted Medical Assistant",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main content area
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Video feed
        left_panel = tk.Frame(main_frame, bg="white", relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video label
        video_label_frame = tk.Frame(left_panel, bg="white")
        video_label_frame.pack(pady=10)
        
        tk.Label(
            video_label_frame,
            text="📹 Camera Feed",
            font=("Arial", 14, "bold"),
            bg="white"
        ).pack()
        
        # Video display
        self.video_label = tk.Label(left_panel, bg="black")
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_frame = tk.Frame(left_panel, bg="#ecf0f1", height=40)
        self.status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="🔴 Camera: Inactive",
            font=("Arial", 10),
            bg="#ecf0f1",
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Right panel - Controls and info
        right_panel = tk.Frame(main_frame, bg="white", width=350, relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Control buttons section
        control_frame = tk.LabelFrame(
            right_panel,
            text="⚙️ Controls",
            font=("Arial", 12, "bold"),
            bg="white",
            fg="#2c3e50"
        )
        control_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Medicine Recognition Section
        medicine_section = tk.LabelFrame(
            control_frame,
            text="💊 Medicine Recognition",
            font=("Arial", 10, "bold"),
            bg="white",
            fg="#27ae60"
        )
        medicine_section.pack(fill=tk.X, padx=10, pady=10)
        
        # Add Medicine button
        self.add_medicine_btn = tk.Button(
            medicine_section,
            text="➕ Add Medicine Reference",
            font=("Arial", 10, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            cursor="hand2",
            command=self.add_medicine_reference,
            height=2
        )
        self.add_medicine_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Recognize Medicine button
        self.recognize_medicine_btn = tk.Button(
            medicine_section,
            text="🔍 Recognize Medicine",
            font=("Arial", 10, "bold"),
            bg="#27ae60",
            fg="white",
            activebackground="#229954",
            activeforeground="white",
            cursor="hand2",
            command=self.start_medicine_recognition,
            height=2
        )
        self.recognize_medicine_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Count Medicines button
        self.count_medicine_btn = tk.Button(
            medicine_section,
            text="🔢 Count Medicines",
            font=("Arial", 10, "bold"),
            bg="#16a085",
            fg="white",
            activebackground="#138d75",
            activeforeground="white",
            cursor="hand2",
            command=self.start_medicine_counting,
            height=2
        )
        self.count_medicine_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Behavior Detection Section
        behavior_section = tk.LabelFrame(
            control_frame,
            text="👤 Behavior Detection",
            font=("Arial", 10, "bold"),
            bg="white",
            fg="#e74c3c"
        )
        behavior_section.pack(fill=tk.X, padx=10, pady=10)
        
        # Detect Abnormal Behavior button
        self.detect_behavior_btn = tk.Button(
            behavior_section,
            text="🚨 Detect Abnormal Behavior",
            font=("Arial", 10, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            cursor="hand2",
            command=self.start_behavior_detection,
            height=2
        )
        self.detect_behavior_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Stop button
        self.stop_btn = tk.Button(
            control_frame,
            text="⏹️ Stop Detection",
            font=("Arial", 10, "bold"),
            bg="#95a5a6",
            fg="white",
            activebackground="#7f8c8d",
            activeforeground="white",
            cursor="hand2",
            command=self.stop_detection,
            height=2,
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, padx=10, pady=10)
        
        # Info/Log section
        info_frame = tk.LabelFrame(
            right_panel,
            text="📋 Information & Logs",
            font=("Arial", 12, "bold"),
            bg="white",
            fg="#2c3e50"
        )
        info_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            info_frame,
            font=("Courier", 9),
            bg="#f9f9f9",
            fg="#2c3e50",
            wrap=tk.WORD,
            height=15
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Clear log button
        clear_log_btn = tk.Button(
            info_frame,
            text="🗑️ Clear Logs",
            font=("Arial", 9),
            bg="#95a5a6",
            fg="white",
            cursor="hand2",
            command=self.clear_logs
        )
        clear_log_btn.pack(padx=10, pady=(0, 10))
        
        # Initial log message
        self.log("Welcome to AAMA - AI-Assisted Medical Assistant")
        self.log("Initializing detectors...")
        
    def _initialize_detectors(self):
        """Initialize the medicine recognizer and behavior detector."""
        try:
            self.log("Loading Medicine Recognizer...")
            self.medicine_recognizer = MedicineRecognizer()
            self.log(f"✓ Medicine Recognizer loaded with {len(self.medicine_recognizer.medicine_db)} references")
            
            self.log("Loading Behavior Detector...")
            self.behavior_detector = AbnormalBehaviorDetector()
            self.log("✓ Behavior Detector loaded")
            
            self.log("Loading Medicine Counter...")
            self.medicine_counter = MedicineCounter()
            self.log("✓ Medicine Counter loaded")
            
            self.log("System ready! Please select an operation.")
        except Exception as e:
            self.log(f"✗ Error initializing detectors: {str(e)}", error=True)
            messagebox.showerror("Initialization Error", f"Failed to initialize detectors:\n{str(e)}")
    
    def log(self, message, error=False):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "❌" if error else "•"
        log_message = f"[{timestamp}] {prefix} {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        if error:
            # Color error messages in red
            self.log_text.tag_add("error", "end-2l", "end-1l")
            self.log_text.tag_config("error", foreground="red")
        
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_logs(self):
        """Clear the log text area."""
        self.log_text.delete(1.0, tk.END)
        self.log("Logs cleared")
    
    def add_medicine_reference(self):
        """Add a new medicine reference."""
        if self.medicine_recognizer is None:
            messagebox.showerror("Error", "Medicine Recognizer not initialized")
            return
        
        # Create dialog for medicine name
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Medicine Reference")
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        dialog.configure(bg="white")
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(
            dialog,
            text="Enter Medicine Name:",
            font=("Arial", 12),
            bg="white"
        ).pack(pady=20)
        
        name_entry = tk.Entry(dialog, font=("Arial", 11), width=30)
        name_entry.pack(pady=10)
        name_entry.focus()
        
        def submit():
            medicine_name = name_entry.get().strip()
            if not medicine_name:
                messagebox.showwarning("Invalid Input", "Please enter a medicine name")
                return
            
            dialog.destroy()
            self._capture_medicine_reference(medicine_name)
        
        tk.Button(
            dialog,
            text="Capture from Camera",
            font=("Arial", 10, "bold"),
            bg="#3498db",
            fg="white",
            cursor="hand2",
            command=submit
        ).pack(pady=10)
        
        name_entry.bind('<Return>', lambda e: submit())
    
    def _capture_medicine_reference(self, medicine_name):
        """Capture medicine reference from camera."""
        self.log(f"Preparing to capture reference for: {medicine_name}")
        
        # Start camera if not active
        if not self.camera_active:
            if not self._start_camera():
                return
        
        # Show countdown and capture
        self.current_mode = 'capture_reference'
        self.capture_medicine_name = medicine_name
        self.capture_countdown = 5

        # Start live preview while waiting to capture
        self._start_capture_preview()
        
        def countdown():
            if self.capture_countdown > 0:
                self.log(f"Capturing in {self.capture_countdown}...")
                self.capture_countdown -= 1
                self.root.after(1000, countdown)
            else:
                self._perform_capture()
        
        countdown()

    def _start_capture_preview(self):
        """Begin updating the live preview during reference capture."""
        if self.capture_preview_job is not None:
            return
        self.capture_preview_job = self.root.after(10, self._capture_preview_loop)

    def _stop_capture_preview(self, clear_last=True):
        """Stop the live preview loop."""
        if self.capture_preview_job is not None:
            self.root.after_cancel(self.capture_preview_job)
            self.capture_preview_job = None
        if clear_last:
            self.capture_last_frame = None

    def _capture_preview_loop(self):
        """Continuously update the GUI with live camera frames during capture."""
        if self.current_mode != 'capture_reference' or not self.camera_active:
            self.capture_preview_job = None
            return

        if self.video_capture is None or not self.video_capture.isOpened():
            self.capture_preview_job = self.root.after(100, self._capture_preview_loop)
            return

        ret, frame = self.video_capture.read()
        if not ret or frame is None:
            self.capture_preview_job = self.root.after(50, self._capture_preview_loop)
            return

        self.capture_last_frame = frame.copy()

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Draw guidance rectangle
        margin = 0.18
        x1, y1 = int(w * margin), int(h * margin)
        x2, y2 = int(w * (1 - margin)), int(h * (1 - margin))
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Overlay countdown and instructions
        if self.capture_countdown > 0:
            countdown_text = f"Capturing in {self.capture_countdown}s"
        else:
            countdown_text = "Capturing..."
        cv2.putText(
            display_frame,
            countdown_text,
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3
        )
        if self.capture_medicine_name:
            cv2.putText(
                display_frame,
                f"Medicine: {self.capture_medicine_name}",
                (40, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        cv2.putText(
            display_frame,
            "Keep the medicine inside the box",
            (40, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            display_frame,
            "Hold steady for a clear capture",
            (40, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Show extracted preview in the corner if available
        try:
            if self.medicine_recognizer is not None:
                extracted, _, _, conf = self.medicine_recognizer._extract_medicine_object(frame)
                if extracted is not None and extracted.size > 0:
                    preview_h = min(180, h // 3)
                    preview_w = int(extracted.shape[1] * (preview_h / max(extracted.shape[0], 1)))
                    if preview_w > 0 and preview_w < w:
                        preview = cv2.resize(extracted, (preview_w, preview_h))
                        x_offset = w - preview_w - 15
                        y_offset = 15
                        cv2.rectangle(
                            display_frame,
                            (x_offset - 5, y_offset - 5),
                            (x_offset + preview_w + 5, y_offset + preview_h + 5),
                            (0, 0, 0),
                            -1
                        )
                        cv2.rectangle(
                            display_frame,
                            (x_offset - 5, y_offset - 5),
                            (x_offset + preview_w + 5, y_offset + preview_h + 5),
                            (0, 255, 255),
                            2
                        )
                        display_frame[y_offset:y_offset + preview_h, x_offset:x_offset + preview_w] = preview
                        cv2.putText(
                            display_frame,
                            f"Extracted ({conf:.0%})" if conf is not None else "Extracted",
                            (x_offset, y_offset + preview_h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )
        except Exception:
            pass

        self._update_video_display(display_frame)

        # Schedule next frame
        self.capture_preview_job = self.root.after(30, self._capture_preview_loop)
    
    def _perform_capture(self):
        """Perform the actual capture of medicine reference."""
        if self.video_capture is None or not self.video_capture.isOpened():
            self.log("Camera not available", error=True)
            return
        
        frame = self.capture_last_frame.copy() if self.capture_last_frame is not None else None
        self._stop_capture_preview(clear_last=False)

        if frame is None:
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                self.log("Failed to capture frame", error=True)
                return
        
        # Add the reference
        self.log(f"Capturing reference for {self.capture_medicine_name}...")
        success = self.medicine_recognizer.add_reference(self.capture_medicine_name, image=frame)
        
        if success:
            self.log(f"✓ Successfully added reference for {self.capture_medicine_name}")
            messagebox.showinfo("Success", f"Medicine reference '{self.capture_medicine_name}' added successfully!")
        else:
            self.log(f"✗ Failed to add reference for {self.capture_medicine_name}", error=True)
            messagebox.showerror("Error", f"Failed to add medicine reference. Please ensure the medicine is clearly visible.")
        
        self.capture_last_frame = None

        # Reset mode
        self.current_mode = None
        self.stop_detection()
    
    def start_medicine_recognition(self):
        """Start medicine recognition mode."""
        if self.medicine_recognizer is None:
            messagebox.showerror("Error", "Medicine Recognizer not initialized")
            return
        
        if not self.medicine_recognizer.medicine_db:
            messagebox.showwarning(
                "No References",
                "No medicine references found.\nPlease add medicine references first."
            )
            return
        
        self.log("Starting medicine recognition...")
        self.current_mode = 'medicine'
        
        if not self._start_camera():
            return
        
        # Update button states
        self._update_button_states(active=True)
        
        # Start processing thread
        self.stop_thread = False
        thread = threading.Thread(target=self._medicine_recognition_loop, daemon=True)
        thread.start()
    
    def start_medicine_counting(self):
        """Start medicine counting mode."""
        if self.medicine_counter is None:
            messagebox.showerror("Error", "Medicine Counter not initialized")
            return
        
        self.log("Starting medicine counting...")
        self.current_mode = 'counting'
        
        if not self._start_camera():
            return
        
        # Reset counter
        self.medicine_counter.reset_count()
        self.medicine_counter.start_counting()
        
        # Update button states
        self._update_button_states(active=True)
        
        # Start processing thread
        self.stop_thread = False
        thread = threading.Thread(target=self._medicine_counting_loop, daemon=True)
        thread.start()
    
    def start_behavior_detection(self):
        """Start abnormal behavior detection mode."""
        if self.behavior_detector is None:
            messagebox.showerror("Error", "Behavior Detector not initialized")
            return
        
        self.log("Starting abnormal behavior detection...")
        self.current_mode = 'behavior'
        
        if not self._start_camera():
            return
        
        # Update button states
        self._update_button_states(active=True)
        
        # Start processing thread
        self.stop_thread = False
        thread = threading.Thread(target=self._behavior_detection_loop, daemon=True)
        thread.start()
    
    def stop_detection(self):
        """Stop the current detection mode."""
        self.log("Stopping detection...")
        self.stop_thread = True
        self.current_mode = None
        self._stop_capture_preview()
        
        # Stop counter if active
        if self.medicine_counter is not None:
            self.medicine_counter.stop_counting()
        
        # Stop camera
        self._stop_camera()
        
        # Update button states
        self._update_button_states(active=False)
        
        self.log("Detection stopped")
    
    def _start_camera(self):
        """Start the camera."""
        if self.camera_active:
            return True
        
        try:
            self.video_capture = cv2.VideoCapture(self.camera_index)
            if not self.video_capture.isOpened():
                raise Exception("Could not open camera")
            
            # Set camera properties for better quality
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.camera_active = True
            self.status_label.config(text="🟢 Camera: Active")
            self.log("Camera started")
            return True
            
        except Exception as e:
            self.log(f"Failed to start camera: {str(e)}", error=True)
            messagebox.showerror("Camera Error", f"Could not start camera:\n{str(e)}")
            return False
    
    def _stop_camera(self):
        """Stop the camera."""
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
        self.camera_active = False
        self.status_label.config(text="🔴 Camera: Inactive")
        
        # Clear video display
        self.video_label.config(image='')
    
    def _medicine_recognition_loop(self):
        """Main loop for medicine recognition with enhanced visualization."""
        while not self.stop_thread and self.camera_active:
            if self.video_capture is None or not self.video_capture.isOpened():
                break
            
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                continue
            
            # Perform medicine recognition
            result = self.medicine_recognizer.identify_medicine(frame)
            
            # Annotate frame
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Show extracted medicine in corner
            try:
                extracted, _, _, det_conf = self.medicine_recognizer._extract_medicine_object(frame)
                
                if extracted is not None and extracted.shape[0] > 0 and extracted.shape[1] > 0:
                    # Resize for preview
                    preview_h = min(200, h // 3)
                    preview_w = int(extracted.shape[1] * (preview_h / extracted.shape[0]))
                    
                    if preview_w > 0 and preview_h > 0 and preview_w < w:
                        preview = cv2.resize(extracted, (preview_w, preview_h))
                        
                        # Place in top-right corner
                        y_offset = 10
                        x_offset = w - preview_w - 10
                        
                        # Add background
                        cv2.rectangle(display_frame, 
                                    (x_offset - 5, y_offset - 5),
                                    (x_offset + preview_w + 5, y_offset + preview_h + 5),
                                    (0, 0, 0), -1)
                        cv2.rectangle(display_frame, 
                                    (x_offset - 5, y_offset - 5),
                                    (x_offset + preview_w + 5, y_offset + preview_h + 5),
                                    (0, 255, 255), 2)
                        
                        display_frame[y_offset:y_offset+preview_h, x_offset:x_offset+preview_w] = preview
                        
                        # Label
                        label_text = "Extracted"
                        if det_conf is not None:
                            label_text = f"Extracted ({det_conf:.0%})"
                        cv2.putText(
                            display_frame,
                            label_text,
                            (x_offset, y_offset - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2
                        )
            except Exception as e:
                pass  # If extraction fails, just continue
            
            # Display results on frame
            if result['medicine']:
                # Background rectangle for text
                cv2.rectangle(display_frame, (5, 5), (450, 120), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (5, 5), (450, 120), (0, 255, 0), 2)
                
                # Draw medicine info
                cv2.putText(
                    display_frame,
                    f"Medicine: {result['medicine']}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    display_frame,
                    f"Confidence: {result['confidence']:.1%}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                
                # Show top matches if available
                if 'all_scores' in result and result['all_scores']:
                    sorted_scores = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
                    score_text = " | ".join([f"{name}: {score:.0%}" for name, score in sorted_scores])
                    cv2.putText(display_frame, score_text, (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
            else:
                # No medicine detected or low confidence
                status_text = "No medicine detected"
                status_color = (0, 0, 255)
                
                if result.get('top_match'):
                    status_text = f"Low confidence: {result['top_match']} ({result['confidence']:.0%})"
                    status_color = (0, 165, 255)  # Orange
                
                cv2.rectangle(display_frame, (5, 5), (550, 50), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (5, 5), (550, 50), status_color, 2)
                cv2.putText(
                    display_frame,
                    status_text,
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2
                )
            
            # Update display
            self._update_video_display(display_frame)
    
    def _behavior_detection_loop(self):
        """Main loop for behavior detection."""
        while not self.stop_thread and self.camera_active:
            if self.video_capture is None or not self.video_capture.isOpened():
                break
            
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                continue
            
            # Process frame for behavior detection
            result = self.behavior_detector.process_frame(frame)
            
            # Get annotated frame
            display_frame = self.behavior_detector.get_annotated_frame(frame, result)
            
            # Log alerts
            if result.get('alert') and result.get('behavior_type') != 'Normal':
                self.log(f"⚠️ ALERT: {result['behavior_type']} detected!", error=True)
            
            # Update display
            self._update_video_display(display_frame)
    
    def _medicine_counting_loop(self):
        """Main loop for medicine counting."""
        while not self.stop_thread and self.camera_active:
            if self.video_capture is None or not self.video_capture.isOpened():
                break
            
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                continue
            
            # Count medicines in frame
            result = self.medicine_counter.count_medicines(frame)
            
            # Log count changes
            if result['count'] > 0:
                current_count = result['count']
                if not hasattr(self, '_last_logged_count') or self._last_logged_count != current_count:
                    self.log(f"📊 Medicines counted: {current_count}")
                    self._last_logged_count = current_count
            
            # Display annotated frame
            if result['annotated_frame'] is not None:
                self._update_video_display(result['annotated_frame'])
            else:
                self._update_video_display(frame)
    
    def _update_video_display(self, frame):
        """Update the video display with a new frame."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit display area
            display_height = 500
            aspect_ratio = frame.shape[1] / frame.shape[0]
            display_width = int(display_height * aspect_ratio)
            
            rgb_frame = cv2.resize(rgb_frame, (display_width, display_height))
            
            # Convert to PhotoImage
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        except Exception as e:
            pass  # Silently ignore display errors
    
    def _update_button_states(self, active):
        """Update button states based on detection activity."""
        if active:
            self.add_medicine_btn.config(state=tk.DISABLED)
            self.recognize_medicine_btn.config(state=tk.DISABLED)
            self.count_medicine_btn.config(state=tk.DISABLED)
            self.detect_behavior_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        else:
            self.add_medicine_btn.config(state=tk.NORMAL)
            self.recognize_medicine_btn.config(state=tk.NORMAL)
            self.count_medicine_btn.config(state=tk.NORMAL)
            self.detect_behavior_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit AAMA?"):
            self.stop_thread = True
            self._stop_camera()
            self.root.destroy()


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = AAMAApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

