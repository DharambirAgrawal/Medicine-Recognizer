# 🚀 Quick Start Guide - AAMA Application

## Step 1: Install Dependencies

Open PowerShell in the `aama` folder and run:

```powershell
pip install pillow
```

All other dependencies should already be installed based on your existing packages.

## Step 2: Run the Application

### Option A: Using the Launcher (Recommended)

Double-click on `launch_aama.bat` or run in PowerShell:

```powershell
.\launch_aama.ps1
```

### Option B: Manual Start

```powershell
python gui_app.py
```

## Step 3: Using the Application

### 📸 First Time Setup - Add Medicine References

1. **Click "➕ Add Medicine Reference"**
2. Enter medicine name (e.g., "Aspirin", "Vitamin C")
3. Click "Capture from Camera"
4. Hold medicine clearly in front of camera
5. Wait for 5-second countdown
6. Reference saved! ✅

**Tip**: Add 2-3 references from different angles for better recognition.

### 💊 Recognize Medicines

1. **Click "🔍 Recognize Medicine"**
2. Hold medicine in front of camera
3. Green box appears when recognized
4. View medicine name and confidence score
5. Click "⏹️ Stop Detection" when done

### 🚨 Detect Abnormal Behavior

1. **Click "🚨 Detect Abnormal Behavior"**
2. Stand in front of camera (full body visible)
3. System monitors for:
   - Falls
   - Seizures
   - Heart attack symptoms
   - Unusual movements
4. Alerts appear in RED if detected
5. Click "⏹️ Stop Detection" when done

## Keyboard Shortcuts

- **Q**: Quit/Stop current detection (when camera window has focus)
- **Y**: Confirm action (when adding reference)

## Troubleshooting

### "Could not open camera"

- Close other apps using the camera (Zoom, Teams, etc.)
- Restart the application
- Try a different camera if you have multiple

### "No medicine detected"

- Improve lighting
- Hold medicine steadier
- Ensure you've added references first
- Move medicine closer/further from camera

### "Module not found" errors

Make sure all packages are installed:

```powershell
pip install opencv-contrib-python numpy mediapipe pillow
```

## Tips for Best Results

### Medicine Recognition 💊

- ✅ Good lighting (natural light or bright room)
- ✅ Steady camera/hand
- ✅ Clear view of label/packaging
- ✅ Add multiple reference angles
- ❌ Avoid blurry images
- ❌ Don't cover important features

### Behavior Detection 👤

- ✅ Full body visible in frame
- ✅ Position camera at chest height
- ✅ Clear background
- ✅ Good lighting
- ✅ Wait 5 seconds for calibration
- ❌ Avoid partial body shots
- ❌ Don't wear similar colors to background

## Output Files

### Medicine References

Saved in: `medicine_references/`

- Format: `medicine_name_timestamp.jpg`

### Behavior Alerts

Saved in: `abnormal_behavior_alerts/alert_XXXXXXXX/`

- `metadata.txt` - Alert details
- `frame_XXX.jpg` - Video frames during alert

## Need Help?

Check the logs in the application window (bottom right panel) for detailed information about what's happening.

---

**Enjoy using AAMA! 🏥**
