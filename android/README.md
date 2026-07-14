# AI CCTV — Android App

Native Android client for the AI CCTV threat-detection server in this repo.
Ring-style experience: live camera view with AI detection overlays, threat
alerts with snapshots, and push-style notifications on HIGH / CRITICAL threats.

## How it works

```
IP camera (RTSP)  ──►  Detection server (FastAPI + YOLO, this repo)  ──►  Android app
                        uvicorn app.main:app --port 8000                  live view, alerts,
                                                                          notifications
```

The app does not talk to cameras directly — it talks to your detection
server, which pulls the camera stream, runs the AI ensemble, draws the
detection boxes, and stores alert events. This means the phone gets the
**annotated** video and the full alert history.

> **Note on Ring cameras:** Ring does not offer a public live-stream API, so
> no third-party app can view Ring cameras directly. Use any camera that
> supports RTSP (TP-Link Tapo, Hikvision, Reolink, Amcrest, Wyze with RTSP
> firmware, or an old phone running an "IP Webcam" app).

## Features

- **Live tab** — real-time MJPEG feed with detection boxes, color-coded
  threat banner (green → yellow → orange → red), fps/inference stats,
  start/stop any camera source (RTSP URL or webcam index).
- **Alerts tab** — event history with level, message, time, and snapshot
  thumbnails; pull to refresh.
- **Settings tab** — server URL, default camera, notification toggle, and
  optional background monitoring (foreground service that keeps polling and
  notifies on HIGH/CRITICAL even when the app is closed).

## Setup

1. Start the detection server on your PC:
   `uvicorn app.main:app --host 0.0.0.0 --port 8000`
   (`--host 0.0.0.0` is required so the phone can reach it over Wi-Fi).
2. Find your PC's LAN IP (`ipconfig` → IPv4 Address, e.g. `192.168.1.100`).
3. In the app's **Settings**, set server URL to `http://<pc-ip>:8000`, save.
4. In **Live**, enter your camera's RTSP URL, e.g.
   `rtsp://user:pass@192.168.1.50:554/stream1`, and tap **Start camera**.

Phone and PC must be on the same network (or the server must be exposed via
VPN/tunnel — do not port-forward it raw to the internet, it has no auth).

## Building

Requirements: Android Studio's bundled JDK (or any JDK 17+) and the Android
SDK. From the `android/` folder:

```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio1\jbr"
# This machine: default TMP breaks JDK AF_UNIX sockets ("Unable to establish
# loopback connection"); this makes every JVM (Gradle/Kotlin daemons) use C:\Temp
$env:JDK_JAVA_OPTIONS = "-Djdk.net.unixdomain.tmpdir=C:\Temp"
.\gradlew.bat assembleDebug      # debug APK  -> app/build/outputs/apk/debug/
.\gradlew.bat bundleRelease      # signed AAB -> app/build/outputs/bundle/release/
```

Release signing reads `keystore.properties` (gitignored) next to this file:

```properties
storeFile=release.keystore
storePassword=...
keyAlias=aicctv
keyPassword=...
```

**Keep `release.keystore` safe and back it up** — Play Store updates must be
signed with the same key forever (or enroll in Play App Signing, recommended).

## Publishing to the Play Store

1. Create a [Google Play Console](https://play.google.com/console) developer
   account ($25 one-time fee).
2. Create an app, fill in the store listing (title, description, screenshots,
   feature graphic, privacy policy URL — required).
3. Upload `app-release.aab` from `app/build/outputs/bundle/release/` to an
   internal-testing track first; opt in to Play App Signing.
4. Complete the content-rating questionnaire and data-safety form
   (the app sends no data to third parties; video stays on your own server).
5. Because the app uses a `dataSync` foreground service, the console will ask
   for a foreground-service declaration — describe it as "polls the user's own
   self-hosted camera server to deliver security alerts."
6. Promote to production once testing passes review.

## Sideloading (no Play Store needed)

Copy `app/build/outputs/apk/debug/app-debug.apk` to the phone and open it
(enable "install unknown apps" when prompted). Fastest way to try it today.
