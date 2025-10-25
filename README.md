# AAMA - AI-Assisted Medical Assistant

## Overview

AAMA (AI-Assisted Medical Assistant) is a Python application that combines real-time medicine recognition with abnormal behavior monitoring inside a single, user-friendly desktop experience. The system detects medicines shown to a camera, recognises abnormal human behaviors such as falls or seizures, and helps surface alerts so caregivers can connect patients with medical professionals. Planned extensions also cover prescription reading, sentiment-aware alerts, and contextual recommendations for patients.

## Core Capabilities

- **Medicine recognition**: capture reference images, auto-extract the medicine from the background, and recognise items with confidence scoring and bounding boxes.
- **Medicine counting**: real-time continuous counting of medicine objects in the camera view using advanced object detection and tracking algorithms.
- **Abnormal behavior detection**: analyse human pose streams to flag falls, seizures, heart attack indicators, unusual motion patterns, and sustained immobility.
- **Guided capture workflow**: live preview, countdown timers, capture guides, and automatic saving to a structured reference library.
- **Alerting and logging**: timestamped alert folders, on-screen status banners, and persistent logs inside the GUI.

## Modules and Architecture

- `gui_app.py`: tkinter-based interface coordinating camera threads, logs, capture controls, and visual overlays at 640x480 resolution.
- `medicine_recognizer.py`: computer-vision engine handling reference capture, feature extraction, matching, and scoring with ORB features, color histograms, and shape descriptors.
- `medicine_counter.py`: real-time medicine counting using contour detection, color segmentation, blob detection, and non-maximum suppression for accurate object tracking.
- `abnormal_behavior_detector.py`: MediaPipe-driven pose tracker exposing `process_frame` and `get_annotated_frame` for single-frame analysis within the GUI loop.
- `launch_aama.bat` and `launch_aama.ps1`: Windows launchers that activate environments, verify dependencies, and start the app.
- `requirements.txt`: dependency pinning, including Pillow for tkinter image bridges.

```
User Actions → tkinter GUI (gui_app.py)
  → Medicine Recognizer (medicine_recognizer.py)
  → Medicine Counter (medicine_counter.py)
  → Abnormal Behavior Detector (abnormal_behavior_detector.py)
    ↘ Storage: medicine_references/, abnormal_behavior_alerts/
```

## Project Structure

```
aama/
  gui_app.py                 # Main GUI application
  medicine_recognizer.py     # Recognition engine
  medicine_counter.py        # Real-time medicine counting engine
  abnormal_behavior_detector.py
  requirements.txt
  launch_aama.bat
  launch_aama.ps1
  medicine_references/       # Stored medicine samples
  abnormal_behavior_alerts/  # Alert archives with metadata and frames
  models/                    # Pre-trained model weights
  temp/                      # Working files generated during capture
```

## Why Tkinter

- Bundled with Python, so end users avoid an additional install step on Windows deployments.
- Lightweight widget toolkit suited for forms, status panes, and button-driven workflows common in healthcare dashboards.
- Smooth bridge to OpenCV through Pillow, enabling live camera feeds and annotated frames without custom event loops.
- Cross-platform compatibility, keeping the same code running on Windows, macOS, or Linux where Python is available.
- Supports the logging panel, status bar, and modal prompts that guide non-technical users through capture and review tasks.

## Recent Enhancements

- **Advanced capture pipeline**: automatic medicine extraction blends GrabCut, HSV color segmentation, and edge-based contour isolation so references store only the pill or package, not the environment.
- **Multi-feature scoring**: configurable weighting across SIFT, ORB, color histograms, and Hu moment shape descriptors supports more robust matching when multiple samples per medicine are present.
- **Recognition feedback**: live preview windows, guide rectangles, five-second countdowns, manual capture overrides, and side-by-side before/after comparisons streamline reference onboarding.
- **Optimised performance mode**: a fast ORB-only pipeline with adaptive thresholding, contour filtering, and simplified matching boosts throughput to roughly 20-30 FPS while retaining color similarity checks.
- **Visual telemetry**: extracted sample thumbnails, color-coded status blocks, and top-match score tables clarify recognition outcomes inside the GUI.
- **Thread-safe processing**: background camera threads, queues, and stop flags keep the UI responsive during long-running capture or detection sessions.

## Requirements

- Python 3.6 or newer.
- OpenCV (opencv-contrib-python for ORB and GUI previews).
- NumPy, MediaPipe, and Pillow.
- Optional CUDA support when configuring GPU acceleration.

## Installation and Environment Setup

- Clone the repository and move into the `aama` directory.
- (Optional) create and activate a virtual environment with `python -m venv env` followed by `env\Scripts\Activate.ps1`.
- Install dependencies: `pip install opencv-contrib-python numpy mediapipe pillow`.
- Pillow is required for tkinter's PhotoImage workflow; other packages cover vision and pose tracking.

## Quick Start

- Preferred: run `launch_aama.bat` or execute `.\launch_aama.ps1` from PowerShell to activate the environment and start the GUI.
- Manual: from the `aama` folder run `python gui_app.py`.
- Minimal setup only needs Pillow if the rest of the stack is already present.

## First-Time Reference Capture

- Choose **Add Medicine Reference** in the GUI, supply the medicine name, and use the guided countdown or press the space bar for manual capture.
- Position the medicine within the on-screen guide rectangle, confirm the extracted preview, and save when the quality looks correct.
- Multiple images per medicine are encouraged; filenames are timestamped inside `medicine_references/` with both extracted and original frames retained when enabled.

## Day-to-Day Usage

- **Recognise medicine**: select **Recognize Medicine**, hold the package steady, watch the confidence display, and stop detection with the provided control once identification is complete.
- **Count medicines**: choose **Count Medicines** to start real-time counting of medicine objects in view. The system continuously tracks and counts pills, tablets, or medicine packages. The count is displayed with visual annotations showing each detected object.
- **Monitor behavior**: choose **Detect Abnormal Behavior**, ensure the subject's full body is visible, and respond to alert banners that include confidence values; stop monitoring when finished.
- **Keyboard shortcuts**: `Q` stops the active camera window, `Y` confirms save prompts during reference capture.

## Programmatic API Example

```python
from medicine_recognizer import MedicineRecognizer

recognizer = MedicineRecognizer(confidence_threshold=0.65)
recognizer.add_reference("Aspirin")  # guided camera capture

# Real-time detection loop
recognizer.run_detection()

# One-off identification from an image
image = cv2.imread("path/to/image.jpg")
result = recognizer.identify_medicine(image)
print(f"Detected {result['medicine']} with confidence {result['confidence']:.2f}")
```

## Configuration and Tuning

- In `gui_app.py` you can set `confidence_threshold`, `use_cuda`, `history_size`, and `alert_threshold` when initialising recogniser and detector instances.
- Modify `medicine_recognizer.py` to adjust ORB feature counts, histogram bin sizes, weighting factors, or fallback thresholds for low-confidence matches.
- Example snippet:

```python
self.medicine_recognizer = MedicineRecognizer(
  confidence_threshold=0.7,
  use_cuda=False
)

self.behavior_detector = AbnormalBehaviorDetector(
  history_size=60,
  alert_threshold=0.75,
  use_cuda=False
)
```

## Medicine Recognition Pipelines

- **Advanced extraction flow**: resize input, apply GrabCut, HSV segmentation, and edge-based contour analysis; combine masks; run SIFT, ORB, histogram, and shape feature extraction; compute weighted scores (40 percent SIFT, 30 percent ORB, 20 percent color, 10 percent shape) with homography validation.
- **Optimised real-time flow**: bilateral filtering, adaptive thresholding, morphological cleanup, contour ranking by area and centre distance, ORB detection with 500 features, Hamming-distance matching, Lowe ratio filtering, and blended scoring of ORB similarity (70 percent) plus color histogram correlation (30 percent).
- Both flows support multiple samples per medicine, maintain average color histograms, and cache extraction results to avoid redundant processing.

## Medicine Counting System

The Medicine Counter provides real-time, continuous counting of medicine objects using multiple detection methods:

- **Multi-method detection**: combines contour-based detection, color segmentation, and blob detection for robust object identification across different lighting conditions and medicine types.
- **Intelligent filtering**: uses Non-Maximum Suppression (NMS) to eliminate duplicate detections and ensure accurate counts.
- **Adaptive tracking**: maintains a rolling history of counts across frames to smooth out jitter and provide stable count readings.
- **Visual feedback**: annotates each detected medicine with bounding boxes, unique IDs, confidence scores, and detection methods.
- **Real-time performance**: processes frames at high speed with configurable area thresholds and detection confidence levels.

### Counting Detection Methods

1. **Contour-based detection**: uses adaptive thresholding and morphological operations to identify medicine shapes. Filters by area, aspect ratio, and circularity to distinguish pills from background noise.

2. **Color segmentation**: leverages HSV color space to detect common medicine colors (white, orange, blue pills). Effective for colored medications and provides high confidence matches.

3. **Blob detection**: employs SimpleBlobDetector with circularity, convexity, and inertia filters to identify pill-shaped objects. Particularly effective for round tablets and capsules.

### Programmatic Counting API

```python
from medicine_counter import MedicineCounter

# Initialize counter
counter = MedicineCounter(
    min_area=500,         # Minimum object area in pixels
    max_area=50000,       # Maximum object area in pixels
    detection_threshold=0.6  # Confidence threshold
)

# Start counting
counter.start_counting()

# Count medicines in a frame
result = counter.count_medicines(frame)
print(f"Counted {result['count']} medicines")
print(f"Average confidence: {result['confidence']:.2%}")

# Get statistics
stats = counter.get_statistics()
print(f"Current count: {stats['current_count']}")

# Reset counter
counter.reset_count()

# Stop counting
counter.stop_counting()
```

## Performance Notes

- Advanced pipeline accuracy usually improves to roughly 85-95 percent thanks to background removal, multi-feature scoring, and shape analysis, though it consumes more compute time.
- The optimised pipeline reduces per-frame processing to about 50 ms, enabling 20-30 FPS camera feeds with far lower CPU usage by dropping GrabCut, reducing histogram bins, and focusing on ORB features.
- Benchmarks highlight a move from 3-7 FPS under the old approach to smooth real-time performance after optimisation.

## Performance Tuning

- Raise ORB `nfeatures` toward 1000 for tougher scenarios when accuracy matters more than speed, or lower to 300 when you need the lightest pipeline.
- Drop camera resolution to 640x480 (or every second frame) on constrained hardware to maintain responsiveness.
- Trim histogram bins to `[4, 4, 4]` if bandwidth is tight; increase to `[8, 8, 8]` for finer colour discrimination when CPU headroom exists.
- Adjust `confidence_threshold` upward for stricter matches, and downward for more permissive behaviour during testing.
- For behaviour detection, reduce `history_size` or MediaPipe model complexity when latency matters, and revert the values when accuracy is the priority.

## Best Practices

- **Medicine capture**: centre the medicine, use plain high-contrast backgrounds, fill around 70 percent of the frame, avoid glare, and record two or three angles per item.
- **Medicine counting**: spread medicines evenly on a plain, contrasting surface; avoid overlapping objects; ensure good lighting without shadows; keep the camera steady; place medicines within the camera's field of view; use white or dark backgrounds for best contrast.
- **Lighting**: favour bright, even light; natural light is ideal; minimise shadows and reflections; clean the camera lens before capture.
- **Behavior monitoring**: keep the subject's full body inside the frame, position the camera at chest height, allow a brief calibration period, and maintain uncluttered backgrounds.

## Troubleshooting

- Camera access errors usually mean another application owns the device or permissions are missing; close other apps or change the camera index.
- Missing imports can be resolved by installing `opencv-contrib-python`, `numpy`, `mediapipe`, or `pillow` after activating the correct environment.
- Low recognition confidence improves with better lighting, more reference images, or a lower `confidence_threshold`; also ensure reference images are not stale or poorly cropped.
- False behavior alerts may require raising `alert_threshold`, verifying full-body visibility, or giving MediaPipe a few seconds to stabilise.
- If extraction previews cover too much or too little, switch to a plain background, re-centre the object, and ensure it occupies most of the guide rectangle.
- When the log reports "not enough features detected," move closer, improve lighting, or add references with richer textures so ORB can find keypoints.

## Data Output and Storage

- `medicine_references/`: extracted samples named `MedicineName_YYYYMMDD_HHMMSS.jpg`, with optional `_original` backups for raw captures.
- `abnormal_behavior_alerts/alert_XXXXXXXX/`: alert folders with `metadata.txt` and `frame_###.jpg` captures for auditing.
- Logs appear within the GUI and persist for the active session.

## Reliability and Threading

- The GUI keeps UI work on the main thread while camera operations run in background threads guarded by queues, stop flags, and clean shutdown routines.
- Error handling surfaces friendly message boxes, updates the status bar, and prevents duplicate camera sessions from starting simultaneously.

## Known Limitations

- Accuracy depends on distinctive packaging, good lighting, and camera quality; lookalike medicines or reflective surfaces can still confuse the matcher.
- Behavior detection expects a single person centred in the frame and may raise false alarms with cluttered scenes or partial visibility.
- Both recognition and detection assume a stable camera feed; sudden occlusions or lens glare reduce reliability.

## Security Considerations

- Camera access requires explicit operating-system permission; review Windows privacy settings before deployment.
- Reference images and alert frames are stored unencrypted on disk, so restrict physical access or add storage encryption for sensitive deployments.
- No data is transmitted off-device by default, but adding cloud sync or alerts should include encryption, access control, and audit logging.
- Operators should periodically purge obsolete alert folders and medicine references to minimise exposure of personal data.

## Future Enhancements

- Expand to multiple medicine databases, cloud-synchronised references, and shared alert dashboards.
- Integrate barcode or QR scanning, OCR for label text, and deep-learning classifiers for tougher differentiation.
- Add mobile or web clients, push notifications, report exports, and advanced analytics across historical detections.
- Support multi-person behavior tracking, richer alert routing (email or SMS), and integration with health data platforms.

## License and Support

- The project is intended for educational and research purposes.
- For questions, review the console output, check GUI logs, or open an issue in the repository.
