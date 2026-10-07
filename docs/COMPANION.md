# Android capture companion: what the desktop expects

The companion app is planned (docs/ROADMAP.md, Phase 7). This page is the
contract between it and the desktop app, so the app can be written
against something that already works: everything here is implemented and
tested on the desktop side, and `tools/companion/send_capture.py` is a
reference client that sends a capture exactly as the app should.

Before building the app, run `ez2d photos PROJECT --exposure` on real
phone captures: it shows whether the phone's automatic exposure actually
costs placed photos. That, more than anything here, decides whether the
app is worth building (see the roadmap).

## What the app is for

What a browser upload can't do:

- lock focus, exposure and white balance for the whole capture, and use
  one lens only (phones otherwise switch lenses and readjust between
  shots);
- guide the capture live: the ring of angles already shot, a blur check
  after each photo, optionally a shutter that fires once the phone has
  moved enough;
- record the phone's motion with the capture, on the camera frames'
  clock (below).

## Sending a capture

The desktop's **From phone… → Over Wi-Fi** dialog (or `ez2d upload P`)
shows a QR code holding a URL like `http://192.168.1.20:41234/<token>/`.
The app scans it and talks to that URL; the token is the only credential
and is valid while the dialog is open. Requests from outside the local
network are refused.

1. `GET <url>api` returns
   `{"api": 1, "app": "EZ2DIGITIZE", "version": "...", "chunk_limit": N,
   "file_limit": N, "accepts": {"image": [...], "video": [...], "motion":
   [".motion.json"]}}`. Refuse to continue if `api` isn't the version the
   app speaks; a 404 means an address that isn't an upload page, or a
   desktop version from before capture apps.
2. `POST <url>capture` with a JSON body, each field optional:
   `{"source": "android", "device": {"make": "Google", "model": "Pixel 8"},
   "app": "EZ2DIGITIZE Capture 0.1", "flipped": false}`. `flipped` marks
   the object turned over (the second side of a two-sided scan). 400 with
   `{"error": ...}` if invalid.
3. Each file: `PUT <url>files/<id>?offset=N&name=<file name>&size=<bytes>`
   with up to `chunk_limit` bytes (the page sends 4 MB). `<id>` is the
   app's own (letters, digits, `-`, `_`, up to 80), the same for every
   chunk of a file. A chunk must start where the stored part ends: the
   answer is `{"received": N, "complete": bool}`, with 409 when the offset
   was wrong (continue from `received`). After a dropped connection,
   `GET <url>files/<id>` says how much arrived. Files are stored bit for
   bit; send the originals.
4. `POST <url>done` when everything is sent. The user then presses Import
   on the desktop.

What the desktop makes of it: the photos (with their motion log) become
one capture, recorded with the source, device and app sent; each video
(with the log named after it) is imported as frames, like Import video.

## The motion log

One JSON file per photo set, `<anything>.motion.json`; for a video,
`<video stem>.motion.json` (sent alongside it). The format is defined in
`src/ez2digitize/motion_log.py`; in short:

```json
{
  "format": "ez2digitize-motion", "version": 1,
  "device": {"make": "Google", "model": "Pixel 8"},
  "gyroscope":     [[t, x, y, z], ...],
  "accelerometer": [[t, x, y, z], ...],
  "gravity":       [[t, x, y, z], ...],
  "poses": [[t, r00, r01, r02, r10, r11, r12, r20, r21, r22, x, y, z], ...],
  "metric": true,
  "frames": [{"file": "IMG_0001.jpg", "t": 12.345}, ...]
}
```

- **Clock.** `t` in seconds, one clock for everything: the one camera
  frames are stamped with. On Android, `SensorEvent.timestamp` and the
  camera's `SENSOR_TIMESTAMP` share it on most phones
  (`SENSOR_INFO_TIMESTAMP_SOURCE` is `REALTIME`); check that, and convert
  nanoseconds to seconds. A photo's `t` is its frame's timestamp; a
  video's single entry is the timestamp of its first frame.
- **Axes.** Vectors in the camera's axes as the image is stored, before
  any EXIF or display rotation: x right, y down, z forward (the way the
  camera looks). Android reports sensors in the device's axes (x right, y
  up, z out of the screen, phone in its natural portrait). For the back
  camera, with the usual `SENSOR_ORIENTATION` of 90°, that is
  `(x, y, z)_image = (-y, -x, -z)_device`. Check it: a photo taken with
  the phone upright in portrait must get gravity's direction (`down` in
  its capture.json entry) close to `[1, 0, 0]`.
- **Readings.** Gyroscope in rad/s; accelerometer and gravity in m/s² as
  Android reports them (at rest they point up, away from the ground). The
  desktop takes gravity from `gravity` (`TYPE_GRAVITY`), else from the
  accelerometer averaged over a second.
- **Poses** (ARCore), camera to world: the rotation by rows, then the
  camera centre. The world has y pointing down along gravity, the camera
  OpenCV's axes (x right, y down, z forward), in the stored image's axes
  as above. ARCore's `Camera.getPose()` has y up in both, and the camera
  looking along -z; with `F = diag(1, -1, -1)`, the pose to write is
  `R = F · R_arcore · F` and `c = F · c_arcore`. Set `"metric": true`:
  ARCore's unit is the metre, which the desktop then uses for a rough
  real-world scale. (The axes are to check on a real phone: the camera
  placement and the tracked poses should agree once the scale is set.)
- **Frames.** Every photo's file name as sent, with its time.

Everything but `frames` is optional, but the raw sensors cost nothing and
always help: gravity stands the model upright; the gyroscope picks video
frames and drops blurred ones; poses guide matching for long captures,
give plugins a starting placement and set a default scale.

## Testing without the app

```sh
uv run ez2d upload P      # prints the QR code and the URL
python tools/companion/send_capture.py URL photos/*.jpg photos/shoot.motion.json \
    --model "Pixel 8" --app "test"
```
