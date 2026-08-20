# AAMA: AI-Assisted Medical Assistant

![Python](https://img.shields.io/badge/python-3.6%2B-blue)

A desktop computer-vision application that recognizes medicines from a camera feed, counts pills in view, and monitors for abnormal human behavior such as falls or seizures.

## Overview

AAMA is a Tkinter desktop app built around three independent computer-vision engines: a medicine recognizer that identifies pills or packages against a reference library, a medicine counter that tallies objects in real time, and a pose-based behavior monitor that watches for falls, seizures, and other emergency indicators. All three run against a live camera feed and share a single GUI for capture, detection, logging, and alert review.

The repository also includes `proj2`, a standalone React web app with a multilingual disease encyclopedia and a browser-based speech translator, unrelated to the Python computer-vision core.

## Features

- **Medicine recognition**: capture reference images of a medicine, then identify it later from a live camera feed with a confidence score and bounding box.
- **Medicine counting**: real-time counting of pills, tablets, or packages in the camera view, with per-object bounding boxes, IDs, and confidence scores.
- **Abnormal behavior detection**: pose-based analysis that flags falls, seizure-like motion, possible heart-attack indicators, unusual movement, and prolonged immobility, saving annotated frames and metadata for each alert.
- **Guided reference capture**: live preview, countdown timer, on-screen capture guide, and automatic background removal when saving new reference images.
- **In-app logging**: timestamped status log and alert history inside the GUI.

## Tech Stack

- **GUI**: Tkinter, Pillow (camera frame ↔ Tkinter image bridge)
- **Computer vision**: OpenCV (`opencv-contrib-python`) for capture, segmentation (GrabCut), and image preprocessing
- **Object detection**: Ultralytics YOLOv8-nano, with an OpenCV/GrabCut fallback when the deep-learning stack isn't available
- **Classification / embeddings**: PyTorch + TorchVision, EfficientNet-B0 for feature embeddings and cosine-similarity matching, with a color-histogram fallback
- **Pose estimation**: MediaPipe Pose (OpenPose supported if installed separately; contour-based fallback otherwise)
- **Packaging**: PyInstaller (`gui_app.spec`)

## How It Works

### Medicine recognition (`medicine_recognizer.py`)

1. **Detection/extraction**: each frame is passed to YOLOv8-nano to locate the medicine's bounding box (and segmentation mask, when available). If the deep-learning stack can't be loaded, the app falls back to an OpenCV pipeline that combines GrabCut, HSV color segmentation, and contour analysis to isolate the object from the background.
2. **Feature extraction**: the cropped region is run through EfficientNet-B0 to produce an embedding vector from the layer before the classification head. Color histograms and shape (Hu moment) descriptors are computed as well, for the fallback path.
3. **Matching**: a query embedding is compared against every stored reference sample per medicine using cosine similarity, and the best-scoring medicine is returned if it clears `confidence_threshold`. Without embeddings, matching falls back to color-histogram correlation (`cv2.compareHist`).
4. **Reference storage**: captured samples are saved to `medicine_references/` as timestamped JPEGs, and computed features are cached to `models/medicine_db.pkl` to avoid recomputation on startup.

### Medicine counting (`medicine_counter.py`)

Combines three independent detectors (adaptive-threshold contour detection, HSV color segmentation for common pill colors, and OpenCV `SimpleBlobDetector` for round shapes), merges their results with non-maximum suppression to remove duplicates, and smooths the count over a rolling window of frames for a stable readout.

### Abnormal behavior detection (`abnormal_behavior_detector.py`)

Tracks human pose landmarks per frame (MediaPipe Pose by default) alongside a background-subtraction motion signal, then runs the recent pose/motion history through a set of heuristics to flag falls, seizure-like oscillation, possible heart-attack indicators (e.g., chest-clutching posture with reduced motion), unusual motion patterns, and sustained immobility. Triggered alerts save annotated frames and a metadata file under `abnormal_behavior_alerts/`.

## Getting Started

### Requirements

- Python 3.6+
- A webcam
- Windows, macOS, or Linux (Tkinter ships with standard Python installs)

### Installation

```bash
git clone https://github.com/DharambirAgrawal/Medicine-Recognizer.git
cd Medicine-Recognizer
python -m venv env
source env/bin/activate      # Windows: env\Scripts\Activate.ps1
pip install -r requirements.txt
```

On first run, `medicine_recognizer.py` downloads `yolov8n.pt` and pretrained EfficientNet-B0 weights automatically via Ultralytics and TorchVision.

### Running the app

```bash
python gui_app.py
```

On Windows, `launch_aama.bat` / `launch_aama.ps1` activate the virtual environment and start the GUI in one step.

From the GUI you can:

- **Add Medicine Reference**: capture one or more labeled reference images for a medicine.
- **Recognize Medicine**: identify a medicine held up to the camera.
- **Count Medicines**: start a live count of pills/packages in view.
- **Detect Abnormal Behavior**: monitor a person for falls, seizures, or other emergency indicators.

### Testing the counter standalone

```bash
python medicine_counter.py
```

Runs a standalone OpenCV window (`Q` to quit, `R` to reset, `S` to print statistics) for tuning without the full GUI.

## Web Companion (`proj2/`)

A separate Vite + React app, unrelated to the Python CV pipeline:

- **Disease encyclopedia**: a searchable, filterable static reference of conditions.
- **Speech translator**: captures speech via the browser's Web Speech API and translates it using the MyMemory translation API, across several languages including English, Spanish, Hindi, and Nepali.

```bash
cd proj2
npm install
npm run dev
```

## Project Structure

```
gui_app.py                    # Tkinter GUI tying the three engines together
medicine_recognizer.py        # YOLOv8 + EfficientNet-B0 recognition engine
medicine_counter.py           # Contour/color/blob-based real-time counter
abnormal_behavior_detector.py # MediaPipe pose-based behavior monitor
medicine_references/          # Saved reference images
requirements.txt
launch_aama.bat / .ps1        # Windows launch scripts
gui_app.spec                  # PyInstaller build spec
proj2/                        # Standalone React disease encyclopedia + translator
```

## Known Limitations

- Recognition accuracy depends on distinctive packaging, lighting, and camera quality; lookalike medicines can still be confused.
- Behavior detection assumes a single, mostly unobstructed person in frame and can raise false positives in cluttered scenes.
- Reference images and alert frames are stored unencrypted on disk.
- This is a prototype built for educational/research purposes, not a certified medical device.
