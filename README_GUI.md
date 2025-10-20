# AAMA - AI-Assisted Medical Assistant

A comprehensive medical assistance application that uses computer vision and AI to:

1. **Recognize medicines** from camera input
2. **Detect abnormal human behaviors** such as falls, seizures, heart attack symptoms, etc.

## Features

### 💊 Medicine Recognition

- Add medicine references by capturing images from camera
- Real-time medicine recognition with confidence scores
- Visual bounding box highlighting detected medicines
- Persistent medicine database

### 👤 Abnormal Behavior Detection

- Real-time pose estimation using MediaPipe
- Detects:
  - Falls
  - Heart attack symptoms
  - Seizures
  - Unusual motion patterns
  - Immobility/unconsciousness
- Alert system with confidence scores
- Automatic alert saving with timestamps

## Installation

### 1. Activate Virtual Environment (if using one)

```powershell
# Navigate to the aama folder
cd "C:\Users\Dharambir Agrawal\Desktop\Testing\practice\aama"

# Activate virtual environment
.\env\Scripts\Activate.ps1
```

### 2. Install Required Packages

```powershell
pip install opencv-contrib-python numpy mediapipe pillow
```

## Usage

### Starting the Application

Run the GUI application:

```powershell
python gui_app.py
```

### Using the Interface

#### Adding Medicine References:

1. Click **"➕ Add Medicine Reference"**
2. Enter the medicine name in the dialog
3. Click **"Capture from Camera"**
4. Position the medicine clearly in front of the camera
5. Wait for the 5-second countdown
6. The reference will be captured and saved

#### Recognizing Medicines:

1. Click **"🔍 Recognize Medicine"**
2. Show the medicine to the camera
3. The system will identify it in real-time
4. Green text and bounding box will appear if recognized
5. Click **"⏹️ Stop Detection"** when done

#### Detecting Abnormal Behavior:

1. Click **"🚨 Detect Abnormal Behavior"**
2. Ensure the person is visible in the camera frame
3. The system will monitor for abnormal behaviors
4. Alerts will appear in red if detected
5. Click **"⏹️ Stop Detection"** when done

## File Structure

```
aama/
├── gui_app.py                      # Main GUI application
├── medicine_recognizer.py          # Medicine recognition module
├── abnormal_behavior_detector.py   # Behavior detection module
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── medicine_references/            # Stored medicine images
└── abnormal_behavior_alerts/       # Saved behavior alerts
    └── alert_XXXXXXXX/
        ├── metadata.txt            # Alert information
        └── frame_XXX.jpg           # Alert video frames
```

## Troubleshooting

### Camera Issues

- **Error: Could not open camera**
  - Make sure no other application is using the camera
  - Try changing the camera index (default is 0)
  - Check camera permissions in Windows Settings

### Import Errors

- **Import "cv2" could not be resolved**

  - Make sure you've activated the virtual environment
  - Install opencv-contrib-python: `pip install opencv-contrib-python`

- **Import "numpy" could not be resolved**

  - Install numpy: `pip install numpy`

- **Import "mediapipe" could not be resolved**
  - Install mediapipe: `pip install mediapipe`

### Medicine Recognition Not Working

- **No medicine detected**

  - Ensure good lighting conditions
  - Keep the medicine stable in frame
  - Make sure you've added references first
  - Try adding multiple reference images of the same medicine from different angles

- **Low confidence scores**
  - Improve lighting
  - Ensure medicine label/packaging is clearly visible
  - Add more reference images
  - Hold medicine at similar distance as reference

### Behavior Detection Issues

- **No pose detected**

  - Ensure person is fully visible in frame
  - Improve lighting conditions
  - Make sure MediaPipe is properly installed
  - Try standing further from camera to fit full body in frame

- **False alerts**
  - Adjust alert_threshold in the code (default: 0.75)
  - Allow system to calibrate (takes ~5 seconds)

## Configuration

You can adjust detection parameters by modifying the initialization in `gui_app.py`:

```python
# Medicine Recognizer
self.medicine_recognizer = MedicineRecognizer(
    confidence_threshold=0.7,  # Lower = more detections, but less accurate
    use_cuda=False             # Set True if you have CUDA GPU
)

# Behavior Detector
self.behavior_detector = AbnormalBehaviorDetector(
    history_size=60,           # Frames to keep in history
    alert_threshold=0.75,      # Confidence needed for alerts
    use_cuda=False             # Set True if you have CUDA GPU
)
```

## Tips for Best Results

### Medicine Recognition:

- Use clear, well-lit images
- Capture medicine from the same angle you'll scan it
- Include distinctive features (logo, text, unique patterns)
- Add multiple references for the same medicine from different angles

### Behavior Detection:

- Ensure full body is visible in frame
- Use consistent, good lighting
- Position camera at chest/head height
- Avoid cluttered backgrounds
- Allow 5-10 seconds for system calibration

## Known Limitations

1. **Medicine Recognition**:

   - Requires pre-added references
   - Performance depends on image quality and lighting
   - Similar-looking medicines may be confused

2. **Behavior Detection**:
   - Best with single person in frame
   - Requires full body visibility
   - May have false positives with unusual movements
   - Lighting conditions affect accuracy

## Future Improvements

- [ ] Support for multiple medicine databases
- [ ] Cloud-based medicine recognition
- [ ] Multi-person behavior tracking
- [ ] Mobile app integration
- [ ] Alert notifications (email, SMS)
- [ ] Historical data analytics
- [ ] Export reports in PDF format

## License

This project is for educational and research purposes.

## Support

For issues or questions, check the logs in the application or review the console output.
