// SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
// SPDX-License-Identifier: GPL-3.0-or-later
package org.ez2digitize.capture

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.TextView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.UnavailableException
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * The capture screen: camera preview, Connect (scan the desktop's QR code),
 * Shoot, Send. While ARCore tracks, every camera frame's pose goes into the
 * motion log, with the gyroscope, accelerometer and gravity sensors.
 *
 * A skeleton: see README.md for what is still to do (full-resolution photos,
 * locked exposure, guidance).
 */
class MainActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var surface: GLSurfaceView
    private lateinit var status: TextView
    private lateinit var url: EditText
    private lateinit var shoot: Button
    private lateinit var send: Button
    private lateinit var flipped: CheckBox

    private var session: Session? = null
    private var installRequested = false
    private val background = BackgroundRenderer()
    private val work = Executors.newSingleThreadExecutor()

    private var capture = newCapture()
    private var sensors: SensorRecorder? = null
    private var sensorOrientation = 90
    // Whether sensor readings and camera frames share one clock: the camera
    // says so, and the first frame's time agrees (ARCore leaves its time base
    // undefined). Poses and photos come from frames, so they always agree.
    private var sensorsOnFrameClock = true
    private var clockChecked = false

    private val scanning = AtomicBoolean(false)
    private val shootRequested = AtomicBoolean(false)
    private var frameNumber = 0L

    /** One capture: its folder of photos and its motion log. */
    private class Capture(val folder: File, val log: MotionLog) {
        val photos = ArrayList<File>()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        surface = findViewById(R.id.surface)
        status = findViewById(R.id.status)
        url = findViewById(R.id.url)
        shoot = findViewById(R.id.shoot)
        send = findViewById(R.id.send)
        flipped = findViewById(R.id.flipped)
        surface.preserveEGLContextOnPause = true
        surface.setEGLContextClientVersion(2)
        surface.setRenderer(this)
        surface.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        findViewById<Button>(R.id.connect).setOnClickListener {
            scanning.set(true)
            say("Point the camera at the QR code on the computer")
        }
        shoot.setOnClickListener { shootRequested.set(true) }
        send.setOnClickListener { sendCapture() }
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST)
            return
        }
        if (session == null && !startSession()) return
        try {
            session?.resume()
        } catch (e: Exception) {
            say("The camera can't be opened: ${e.message}")
            return
        }
        surface.onResume()
        startSensors()
    }

    override fun onPause() {
        super.onPause()
        sensors?.stop()
        surface.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        session?.close()
        work.shutdown()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(code: Int, permissions: Array<String>, results: IntArray) {
        super.onRequestPermissionsResult(code, permissions, results)
        if (results.firstOrNull() != PackageManager.PERMISSION_GRANTED) {
            say("The camera permission is needed to take photos")
        }
    }

    /** ARCore installed and a session made; false while ARCore is being installed. */
    private fun startSession(): Boolean {
        try {
            val install = ArCoreApk.getInstance().requestInstall(this, !installRequested)
            if (install == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                installRequested = true
                return false
            }
            val created = Session(this)
            // The largest CPU image ARCore offers: it is the photo, for now.
            val configs = created.getSupportedCameraConfigs(
                CameraConfigFilter(created).setFacingDirection(com.google.ar.core.CameraConfig.FacingDirection.BACK),
            )
            configs.maxByOrNull { it.imageSize.width * it.imageSize.height }?.let { created.cameraConfig = it }
            val config = Config(created)
            config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            config.focusMode = Config.FocusMode.AUTO // TODO: lock focus once set (README.md)
            created.configure(config)
            readCamera(created.cameraConfig.cameraId)
            session = created
            return true
        } catch (e: UnavailableException) {
            say("ARCore isn't available on this phone: ${e.message}")
        } catch (e: Exception) {
            say("ARCore couldn't start: ${e.message}")
        }
        return false
    }

    /** The camera's sensor orientation, and whether it stamps frames on the sensors' clock. */
    private fun readCamera(cameraId: String) {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val characteristics = manager.getCameraCharacteristics(cameraId)
        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        val source = characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
        if (source != CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) noSensors()
    }

    /** Sensor events are stamped with elapsedRealtimeNanos; frames must be too. */
    private fun checkClock(frameTimestamp: Long) {
        clockChecked = true
        val gap = Math.abs(SystemClock.elapsedRealtimeNanos() - frameTimestamp)
        if (gap > CLOCK_TOLERANCE_NS) noSensors()
    }

    private fun noSensors() {
        sensorsOnFrameClock = false
        sensors?.stop()
        capture.log.clearSensors()
        say("This phone's camera and sensors keep different clocks: only ARCore's poses are recorded")
    }

    private fun startSensors() {
        sensors?.stop()
        if (!sensorsOnFrameClock) return
        val manager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sensors = SensorRecorder(manager, capture.log, sensorOrientation).also { it.start() }
    }

    // --- rendering (GL thread) ---------------------------------------------------------

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        background.create()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        @Suppress("DEPRECATION")
        session?.setDisplayGeometry(windowManager.defaultDisplay.rotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val session = session ?: return
        session.setCameraTextureName(background.textureId)
        val frame = try {
            session.update()
        } catch (e: Exception) {
            return
        }
        background.draw(frame)
        frameNumber++
        if (!clockChecked && frame.timestamp != 0L) checkClock(frame.timestamp)
        val camera = frame.camera
        val t = MotionLog.seconds(frame.timestamp)
        if (camera.trackingState == TrackingState.TRACKING) {
            val matrix = FloatArray(16)
            camera.pose.toMatrix(matrix, 0) // the physical camera, in the stored image's axes
            capture.log.addPose(t, Axes.arcorePoseToLog(matrix))
        }
        if (scanning.get() && frameNumber % QR_EVERY_FRAMES == 0L) scanQrCode(frame)
        if (shootRequested.getAndSet(false)) takePhoto(frame, camera.trackingState)
    }

    private fun scanQrCode(frame: com.google.ar.core.Frame) {
        val image = try {
            frame.acquireCameraImage()
        } catch (e: NotYetAvailableException) {
            return
        }
        val text = image.use { CameraImages.readQrCode(it) } ?: return
        if (text.startsWith("http://") || text.startsWith("https://")) {
            scanning.set(false)
            runOnUiThread {
                url.setText(text)
                say("Connected to the computer. Shoot all around the object, then Send.")
            }
        }
    }

    private fun takePhoto(frame: com.google.ar.core.Frame, tracking: TrackingState) {
        val image = try {
            frame.acquireCameraImage()
        } catch (e: NotYetAvailableException) {
            say("No camera image yet, try again")
            return
        }
        val pixels = image.use { CameraImages.toNv21(it) }
        val current = capture
        val name = "IMG_%04d.jpg".format(current.photos.size + 1)
        val file = File(current.folder, name)
        current.photos.add(file)
        current.log.addFrame(name, MotionLog.seconds(frame.timestamp))
        work.execute { CameraImages.saveJpeg(pixels, file) }
        val note = if (tracking == TrackingState.TRACKING) "" else " (not tracking: move slowly)"
        say("${current.photos.size} photos$note")
    }

    // --- sending ---------------------------------------------------------------------

    private fun sendCapture() {
        val target = url.text.toString().trim()
        val sent = capture
        if (target.isEmpty()) return say("Connect first: scan the QR code on the computer")
        if (sent.photos.isEmpty()) return say("Take some photos first")
        shoot.isEnabled = false
        send.isEnabled = false
        val otherSide = flipped.isChecked
        work.execute {
            try {
                val log = File(sent.folder, "capture.motion.json")
                log.writer().use { sent.log.write(it) }
                val uploader = Uploader(target) { message -> say(message) }
                val api = uploader.checkApi()
                uploader.describeCapture(Build.MANUFACTURER, Build.MODEL, APP_NAME, otherSide)
                uploader.sendAll(sent.photos + log, api.optInt("chunk_limit", Uploader.CHUNK))
                runOnUiThread {
                    capture = newCapture()
                    startSensors()
                    say("Sent ${sent.photos.size} photos. Press Import on the computer.")
                }
            } catch (e: Exception) {
                say("Sending failed: ${e.message}. Press Send to try again.")
            } finally {
                runOnUiThread {
                    shoot.isEnabled = true
                    send.isEnabled = true
                }
            }
        }
    }

    private fun newCapture(): Capture {
        val folder = File(cacheDir, "capture-${System.currentTimeMillis()}").apply { mkdirs() }
        return Capture(folder, MotionLog(Build.MANUFACTURER, Build.MODEL))
    }

    private fun say(text: String) = runOnUiThread { status.text = text }

    private companion object {
        const val CAMERA_REQUEST = 1
        const val QR_EVERY_FRAMES = 10L
        const val APP_NAME = "EZ2DIGITIZE Capture 0.1"
        const val CLOCK_TOLERANCE_NS = 1_000_000_000L // a frame is never a second old
    }
}
