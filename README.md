# Medicine Recognizer

A Python application that uses OpenCV to detect and recognize medicines shown in front of a camera.

## Features

- Accurate medicine recognition using feature matching (SIFT algorithm)
- Class-based design for easy integration into other projects
- Reference database management for storing known medicines
- Real-time detection with confidence scoring
- Visualization of detection results

## Requirements

- Python 3.6+
- OpenCV 4.2+
- NumPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/DharambirAgrawal/Medicine-Recognizer.git
cd Medicine-Recognizer
```

2. Install dependencies:

```bash
pip install opencv-python numpy
```

## Usage

### Running the Application

Run the main script to start medicine detection:

```bash
python medicine_recognizer.py
```

When first run, you'll be prompted to add reference medicines to the database. Follow the on-screen instructions to add medicine references.

### Adding Medicine References

You can add medicine references by:

1. Running the application and following the prompts
2. Place reference images in the `medicine_references` folder (filename format: `medicine_name.jpg`)

### Using the MedicineRecognizer Class in Your Code

```python
from medicine_recognizer import MedicineRecognizer

# Create a recognizer instance
recognizer = MedicineRecognizer()

# Add a reference (from camera)
recognizer.add_reference("Aspirin")

# Detect medicines from camera
recognizer.run_detection()

# Or identify medicine in a specific image
image = cv2.imread("path/to/image.jpg")
result = recognizer.identify_medicine(image)
print(f"Detected: {result['medicine']} with confidence {result['confidence']}")
```

## How It Works

The application uses the SIFT (Scale-Invariant Feature Transform) algorithm to detect keypoints and descriptors in images. These features are rotation, scale, and illumination invariant, making them ideal for object recognition.

The recognition process involves:

1. Extracting features from the input image
2. Matching these features with the reference database
3. Computing a homography (perspective transformation) to verify spatial consistency
4. Determining the best match based on inlier ratio and confidence threshold

## Limitations

- Works best with medicines that have distinctive packaging or shapes
- May struggle with reflective surfaces or very similar-looking medicines
- Performance depends on lighting conditions and camera quality
  It detects the images of medicines and the customer sentements to alarm alerts and alo connect doctor with the patients. It reads the prescription and gives the user required recommendations.
