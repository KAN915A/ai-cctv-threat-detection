# ⚡ Quick Start Guide - Weapon Detection System

**Get started in 5 minutes!**

---

## 1️⃣ Installation (5 min)

```bash
# Install Python packages
pip install -r requirements_weapon_detector.txt
```

Done! The model downloads automatically on first run (~100MB).

---

## 2️⃣ Run Detection

### Windows
```bash
# Double-click: start_detector.bat
# OR run in command prompt:
python weapon_detector.py --mode webcam
```

### macOS/Linux
```bash
# Make script executable (first time only)
chmod +x start_detector.sh

# Run it
./start_detector.sh
# OR
python3 weapon_detector.py --mode webcam
```

---

## 3️⃣ What Happens

```
1. Webcam opens
2. System scans for weapons
3. Red boxes appear around detected weapons
4. Alert sound plays when weapon found
5. Press 'q' to quit
```

---

## 🎯 Common Commands

| Task | Command |
|------|---------|
| **Start webcam** | `python weapon_detector.py --mode webcam` |
| **Detect in image** | `python weapon_detector.py --mode image --image photo.jpg` |
| **View today's logs** | `python weapon_detector.py --mode logs` |
| **Faster detection** | `python weapon_detector.py --model nano` |
| **More accurate** | `python weapon_detector.py --model large` |
| **Save video** | `python weapon_detector.py --save-video` |
| **IP Camera** | `python weapon_detector.py --camera "rtsp://192.168.1.100/stream"` |

---

## 🆘 Not Working?

### Webcam Error
```bash
# Make sure no other app is using your camera
# Close: Zoom, Teams, Skype, OBS, etc.
```

### Dependencies Missing
```bash
pip install --upgrade -r requirements_weapon_detector.txt
```

### Very Slow (FPS < 5)
```bash
# Use smaller model
python weapon_detector.py --model nano
```

### False Positives (Too many alerts)
```bash
# Increase confidence threshold
python weapon_detector.py --confidence 0.7
```

---

## 📊 Understanding Output

```
Frame: 523 | Weapons: 2 | FPS: 25.3
↑           ↑            ↑
Frame #     Detections   Speed
```

- **Red boxes** = Detected weapons
- **Numbers** = Confidence (0-1, higher = more certain)
- **FPS** = Frames per second (30+ is good)

---

## 💾 Logs Location

All detections saved here:
```
detection_logs/detections_2026-07-10.json
```

View with:
```bash
python weapon_detector.py --mode logs
```

---

## ⌨️ While Running

| Key | Action |
|-----|--------|
| **q** | Quit |
| **s** | Save frame |

---

## 📱 IP Camera Setup

Find your camera's RTSP URL:

**Common URLs:**
```
rtsp://192.168.1.100:554/stream          # Generic
rtsp://admin:password@192.168.1.100      # Hikvision
rtsp://user:pass@192.168.1.100:554/live  # Dahua
```

Then run:
```bash
python weapon_detector.py --camera "rtsp://YOUR_CAMERA_URL"
```

---

## 🚀 Performance

| Model | Speed | Accuracy |
|-------|-------|----------|
| **nano** | 🟢 Fast (25+ FPS) | 75% |
| **small** | 🟡 Medium (15 FPS) | 82% |
| **medium** | 🔴 Slow (5 FPS) | 88% |

**Start with `small` - good balance**

---

## 📝 Example Session

```bash
$ python weapon_detector.py

Loading YOLOv8 model...
✓ Model loaded
Using device: cpu

🎥 Starting webcam detection
Press 'q' to quit

[Webcam opens...]

🚨 GUN DETECTED (0.92) 🚨
[Red box appears around gun]

✓ Frame saved: detection_20260710_142345.jpg

[Press 'q']

✓ Detection finished
  Detections: 5
  Logs saved to: detection_logs/
```

---

## 🔗 Links

- **YOLOv8 Docs**: https://docs.ultralytics.com
- **OpenCV Docs**: https://docs.opencv.org
- **PyTorch**: https://pytorch.org

---

## 🎓 Next Steps

1. ✅ Test with webcam
2. ✅ Try different models
3. ✅ Connect IP camera
4. ✅ Review detection logs
5. ✅ Adjust confidence threshold

---

**Questions? Check `WEAPON_DETECTOR_SETUP.md` for detailed guide**

**Ready?** 🎯
```bash
python weapon_detector.py --mode webcam
```
