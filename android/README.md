# EZ2DIGITIZE Capture (Android): skeleton

The capture companion planned in docs/ROADMAP.md (Phase 7), as a first
working skeleton. It speaks the contract in docs/COMPANION.md: it records
the phone's motion while you take photos and sends both to the desktop's
upload page.

What it does now:

- **Connect:** scans the QR code the desktop shows (From phone… → Over
  Wi-Fi, or `ez2d upload P`) from the camera image, or takes the address
  typed in.
- **Shoot:** saves the current camera image as a JPEG and notes its time.
  While ARCore tracks, every camera frame's pose is recorded; so are the
  gyroscope, accelerometer and gravity sensors, converted into the camera
  image's axes, if the camera and the sensors share a clock (checked).
- **Other side:** marks the capture as the object turned over.
- **Send:** writes `capture.motion.json`, checks the desktop's API
  version, describes the capture (device, app, side) and sends the photos
  and the log in resumable chunks. Then press Import on the desktop.

Still to do, roughly in order:

1. **Check it on a real phone** (none was used to write it): that a photo
   taken upright in portrait gets gravity close to `[1, 0, 0]` on the
   desktop, that ARCore's poses agree with COLMAP's camera placement, and
   that the scale from tracking is near the marker sheet's.
2. **Full-resolution photos.** The photo is ARCore's CPU camera image (the
   largest the phone offers ARCore, often 1920 x 1080). Full resolution
   needs ARCore's shared camera: Camera2 still captures on the same camera
   while ARCore tracks, matched to frames by sensor timestamp.
3. **Locked focus, exposure and white balance** for the whole capture, and
   one lens only (Camera2 settings, with the shared camera).
4. **Guidance:** a ring of the angles already shot (from the poses), a
   blur check after each photo (the gyroscope's turning rate), an
   automatic shutter once the phone has moved enough.
5. Video recording (with the log named after the video), an icon, and
   release signing.

## Building

Needs JDK 17 or newer and the Android SDK (platform 35); Android Studio
has both.

```sh
cd android
./gradlew testDebugUnitTest   # unit tests: axes, the log format
./gradlew assembleDebug       # app/build/outputs/apk/debug/app-debug.apk
adb install app/build/outputs/apk/debug/app-debug.apk
```

The phone needs ARCore (Google Play Services for AR; the app asks to
install it) and must be on the same network as the computer. The Android
app workflow builds the APK in CI.

The log format is checked from both sides: `MotionLogTest` writes a log
for fixed readings and compares it with
`app/src/test/resources/sample.motion.json`, which the desktop's
`tests/test_android_app.py` reads with its own parser.
