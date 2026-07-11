#!/usr/bin/env python3
"""
Browser-based Weapon Detection
Runs YOLOv8 inference on the server; the browser captures webcam frames
(or accepts image uploads) and draws detection boxes live on a canvas.

Usage:
    python web_detector.py [--model yolov8n.pt] [--confidence 0.45] [--port 5000]
"""

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from ultralytics import YOLO

app = Flask(__name__)

model = None
confidence_threshold = 0.45

DEFAULT_MODEL = str(Path(__file__).parent
                    / 'Weapons-and-Knives-Detector-with-YOLOv8-main'
                    / 'runs' / 'detect' / 'Normal' / 'weights' / 'best.pt')

WEAPON_KEYWORDS = ('gun', 'pistol', 'rifle', 'firearm', 'knife', 'knive', 'weapon')
weapon_class_names = set()


def resolve_weapon_classes():
    """Mark which of the loaded model's classes count as weapons."""
    global weapon_class_names
    weapon_class_names = {
        name for name in model.names.values()
        if any(kw in name.lower() for kw in WEAPON_KEYWORDS)
    }

LOG_DIR = Path('detection_logs')
LOG_DIR.mkdir(exist_ok=True)


def log_detections(detections):
    timestamp = datetime.now()
    log_file = LOG_DIR / f"detections_{timestamp.strftime('%Y-%m-%d')}.json"
    with open(log_file, 'a') as f:
        for det in detections:
            if det['is_weapon']:
                json.dump({'timestamp': timestamp.isoformat(), **det}, f)
                f.write('\n')


def run_inference(frame):
    """Run YOLO on a frame; return (detections, elapsed_ms)."""
    start = time.time()
    results = model(frame, conf=confidence_threshold, verbose=False)
    elapsed_ms = (time.time() - start) * 1000

    detections = []
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                'class': class_name,
                'confidence': round(float(box.conf[0]), 3),
                'box': [x1, y1, x2, y2],
                'is_weapon': class_name in weapon_class_names,
            })
    return detections, elapsed_ms


def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['box']
        weapon = det['is_weapon']
        color = (0, 0, 255) if weapon else (150, 150, 150)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if weapon else 2)
        label = f"{det['class']} {det['confidence']:.0%}"
        cv2.rectangle(frame, (x1, max(0, y1 - 26)), (x1 + 11 * len(label), y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, max(16, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('frame')
    if file is None:
        return jsonify({'error': 'no frame provided'}), 400

    data = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'could not decode image'}), 400

    detections, elapsed_ms = run_inference(frame)
    log_detections(detections)
    return jsonify({
        'detections': detections,
        'inference_ms': round(elapsed_ms, 1),
        'image_size': [frame.shape[1], frame.shape[0]],
    })


# --- Server-side camera (no browser permission needed) -----------------------

camera_state = {
    'running': False,
    'thread': None,
    'jpeg': None,          # latest annotated frame, JPEG bytes
    'detections': [],
    'inference_ms': 0.0,
    'error': None,
}
camera_lock = threading.Lock()


def camera_loop(source):
    cap = None
    try:
        # CAP_DSHOW opens much faster than the default MSMF backend on Windows
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            with camera_lock:
                camera_state['error'] = f'could not open camera {source}'
                camera_state['running'] = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while camera_state['running']:
            ret, frame = cap.read()
            if not ret:
                with camera_lock:
                    camera_state['error'] = 'camera stream ended'
                    camera_state['running'] = False
                break

            detections, elapsed_ms = run_inference(frame)
            log_detections(detections)
            draw_boxes(frame, detections)
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with camera_lock:
                    camera_state['jpeg'] = buf.tobytes()
                    camera_state['detections'] = detections
                    camera_state['inference_ms'] = round(elapsed_ms, 1)
    except Exception as e:
        with camera_lock:
            camera_state['error'] = f'camera thread failed: {e}'
            camera_state['running'] = False
    finally:
        if cap is not None:
            cap.release()


@app.route('/camera/start', methods=['POST'])
def camera_start():
    with camera_lock:
        if camera_state['running']:
            return jsonify({'running': True})
    # Let a previous camera thread fully release the device before reopening —
    # tearing down and reopening DirectShow concurrently can crash the process.
    prev = camera_state['thread']
    if prev is not None and prev.is_alive():
        prev.join(timeout=5)
    with camera_lock:
        camera_state['running'] = True
        camera_state['error'] = None
        camera_state['jpeg'] = None
    source = request.json.get('source', 0) if request.is_json else 0
    t = threading.Thread(target=camera_loop, args=(source,), daemon=True)
    camera_state['thread'] = t
    t.start()
    return jsonify({'running': True})


@app.route('/camera/stop', methods=['POST'])
def camera_stop():
    camera_state['running'] = False
    return jsonify({'running': False})


@app.route('/camera/status')
def camera_status():
    with camera_lock:
        return jsonify({
            'running': camera_state['running'],
            'error': camera_state['error'],
            'has_frame': camera_state['jpeg'] is not None,
            'detections': camera_state['detections'],
            'inference_ms': camera_state['inference_ms'],
        })


@app.route('/video_feed')
def video_feed():
    def generate():
        while camera_state['running'] or camera_state['jpeg'] is None:
            with camera_lock:
                jpeg = camera_state['jpeg']
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.03)
            if not camera_state['running']:
                break
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    classes = ', '.join(sorted(model.names.values()))
    weapons = ', '.join(sorted(weapon_class_names)) or 'none'
    info = (f"Model: <b>{Path(model.ckpt_path or 'model').name}</b> · "
            f"classes: <b>{classes}</b> · treated as weapons: <b>{weapons}</b>")
    return Response(PAGE.replace('{{MODEL_INFO}}', info), mimetype='text/html')


PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weapon Detection — Live</title>
<style>
  :root {
    --bg: #0f1115; --panel: #1a1d24; --text: #e8eaed; --muted: #9aa0a6;
    --accent: #4c8bf5; --danger: #e5484d; --ok: #46a758; --border: #2a2e37;
  }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, sans-serif; padding: 24px; }
  .wrap { max-width: 920px; margin: 0 auto; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .toolbar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 16px; }
  button {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 9px 16px; font-size: 14px; cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .45; cursor: default; }
  label.upload {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 9px 16px; font-size: 14px; cursor: pointer;
  }
  label.upload:hover { border-color: var(--accent); }
  input[type=file] { display: none; }
  .stage {
    position: relative; background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden; min-height: 300px;
    display: flex; align-items: center; justify-content: center;
  }
  canvas { display: block; max-width: 100%; height: auto; }
  .placeholder { color: var(--muted); padding: 60px 20px; text-align: center; }
  .alert {
    display: none; margin: 14px 0; padding: 12px 16px; border-radius: 10px;
    background: rgba(229,72,77,.12); border: 1px solid var(--danger);
    color: #ff8f92; font-weight: 600;
  }
  .alert.show { display: block; }
  .stats { display: flex; gap: 18px; margin-top: 12px; color: var(--muted); font-size: 13px; flex-wrap: wrap; }
  .stats b { color: var(--text); font-weight: 600; }
  .note {
    margin-top: 18px; padding: 12px 16px; border-radius: 10px; font-size: 13px;
    background: rgba(76,139,245,.08); border: 1px solid rgba(76,139,245,.35); color: var(--muted);
  }
  video { display: none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🔍 Weapon Detection — Live Browser Demo</h1>
  <div class="sub">YOLOv8 running on the local Python server · frames captured in your browser</div>

  <div class="toolbar">
    <button id="btnSrvCam" class="primary">📷 Start camera (server)</button>
    <button id="btnCam">▶ Browser webcam</button>
    <button id="btnStop" disabled>■ Stop</button>
    <label class="upload">🖼 Detect in image<input type="file" id="fileInput" accept="image/*"></label>
    <label style="color:var(--muted);font-size:13px;display:flex;align-items:center;gap:6px">
      <input type="checkbox" id="chkSound" checked> alert sound
    </label>
  </div>

  <div id="alertBar" class="alert">🚨 WEAPON DETECTED</div>

  <div class="stage">
    <div class="placeholder" id="placeholder">Start the camera or upload an image to begin.</div>
    <canvas id="canvas" hidden></canvas>
    <img id="mjpeg" hidden style="display:block;max-width:100%">
  </div>

  <div class="stats">
    <span>Inference: <b id="stInfer">–</b></span>
    <span>Round-trip FPS: <b id="stFps">–</b></span>
    <span>Objects: <b id="stObjects">–</b></span>
    <span>Weapons: <b id="stWeapons">–</b></span>
  </div>

  <div class="note">{{MODEL_INFO}}</div>

  <video id="video" playsinline muted></video>
</div>

<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const placeholder = document.getElementById('placeholder');
const alertBar = document.getElementById('alertBar');
const btnCam = document.getElementById('btnCam');
const btnStop = document.getElementById('btnStop');

let stream = null;
let running = false;
let audioCtx = null;
let lastBeep = 0;

function beep() {
  if (!document.getElementById('chkSound').checked) return;
  const now = Date.now();
  if (now - lastBeep < 1500) return;
  lastBeep = now;
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.frequency.value = 1000;
  gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.5);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + 0.5);
}

function drawDetections(dets) {
  let weapons = 0;
  for (const d of dets) {
    const [x1, y1, x2, y2] = d.box;
    const weapon = d.is_weapon;
    if (weapon) weapons++;
    ctx.lineWidth = weapon ? 4 : 2;
    ctx.strokeStyle = weapon ? '#e5484d' : 'rgba(154,160,166,.8)';
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${d.class} ${(d.confidence * 100).toFixed(0)}%`;
    ctx.font = '600 15px system-ui';
    const w = ctx.measureText(label).width + 12;
    ctx.fillStyle = weapon ? '#e5484d' : 'rgba(60,64,72,.9)';
    ctx.fillRect(x1, Math.max(0, y1 - 24), w, 24);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 6, Math.max(17, y1 - 6));
  }
  return weapons;
}

function updateStats(data, fps, weapons) {
  document.getElementById('stInfer').textContent = data.inference_ms + ' ms';
  document.getElementById('stFps').textContent = fps ? fps.toFixed(1) : '–';
  document.getElementById('stObjects').textContent = data.detections.length;
  document.getElementById('stWeapons').textContent = weapons;
  alertBar.classList.toggle('show', weapons > 0);
  if (weapons > 0) beep();
}

async function sendFrame(blob) {
  const form = new FormData();
  form.append('frame', blob, 'frame.jpg');
  const res = await fetch('/detect', { method: 'POST', body: form });
  if (!res.ok) throw new Error('detect failed: ' + res.status);
  return res.json();
}

async function loop() {
  if (!running) return;
  const t0 = performance.now();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  try {
    const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.7));
    const data = await sendFrame(blob);
    if (!running) return;
    ctx.drawImage(video, 0, 0);           // redraw freshest frame
    const weapons = drawDetections(data.detections);
    updateStats(data, 1000 / (performance.now() - t0), weapons);
  } catch (e) {
    console.error(e);
  }
  requestAnimationFrame(loop);
}

btnCam.onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  } catch (e) {
    placeholder.textContent = 'Camera access denied or unavailable: ' + e.message;
    return;
  }
  video.srcObject = stream;
  await video.play();
  placeholder.hidden = true;
  mjpeg.hidden = true;
  canvas.hidden = false;
  running = true;
  btnCam.disabled = true;
  btnStop.disabled = false;
  loop();
};

// --- Server-side camera (MJPEG stream, no browser permission needed) ---
const mjpeg = document.getElementById('mjpeg');
const btnSrvCam = document.getElementById('btnSrvCam');
let statusTimer = null;

btnSrvCam.onclick = async () => {
  const res = await fetch('/camera/start', { method: 'POST' });
  if (!res.ok) { placeholder.textContent = 'Failed to start server camera'; return; }
  placeholder.textContent = 'Opening camera…';
  canvas.hidden = true;
  mjpeg.hidden = false;
  mjpeg.src = '/video_feed?' + Date.now();
  mjpeg.onload = () => { placeholder.hidden = true; };
  btnSrvCam.disabled = true;
  btnStop.disabled = false;
  statusTimer = setInterval(async () => {
    try {
      const s = await (await fetch('/camera/status')).json();
      if (s.error) {
        placeholder.hidden = false;
        placeholder.textContent = 'Camera error: ' + s.error;
        stopServerCam();
        return;
      }
      const weapons = s.detections.filter(d => d.is_weapon).length;
      updateStats({ inference_ms: s.inference_ms, detections: s.detections }, 1000 / (s.inference_ms + 30), weapons);
    } catch (e) { /* server restarting */ }
  }, 500);
};

function stopServerCam() {
  if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
  fetch('/camera/stop', { method: 'POST' });
  mjpeg.src = '';
  mjpeg.hidden = true;
  btnSrvCam.disabled = false;
}

btnStop.onclick = () => {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stopServerCam();
  btnCam.disabled = false;
  btnStop.disabled = true;
};

document.getElementById('fileInput').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  btnStop.onclick();
  const img = new Image();
  img.onload = async () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    placeholder.hidden = true;
    canvas.hidden = false;
    const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.9));
    const data = await sendFrame(blob);
    const weapons = drawDetections(data.detections);
    updateStats(data, 0, weapons);
  };
  img.src = URL.createObjectURL(file);
};
</script>
</body>
</html>
"""


def main():
    global model, confidence_threshold

    parser = argparse.ArgumentParser(description='Browser-based weapon detection')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='YOLO model file (default: custom weapons model, '
                             'Normal variant, mAP50 0.87)')
    parser.add_argument('--confidence', type=float, default=0.45)
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    confidence_threshold = args.confidence
    print(f"Loading model {args.model}...")
    model = YOLO(args.model)
    resolve_weapon_classes()
    print(f"Classes: {model.names} | weapons: {weapon_class_names}")
    print(f"Model loaded. Open http://localhost:{args.port} in your browser.")

    app.run(host='127.0.0.1', port=args.port, threaded=True)


if __name__ == '__main__':
    main()
