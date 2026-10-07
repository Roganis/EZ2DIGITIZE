// SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
// SPDX-License-Identifier: GPL-3.0-or-later
package org.ez2digitize.capture

import java.io.Writer
import java.util.Locale

/**
 * The motion recorded during one capture, written as the desktop's motion
 * log (`<name>.motion.json`, format "ez2digitize-motion" version 1; see
 * src/ez2digitize/motion_log.py and docs/COMPANION.md).
 *
 * Times are seconds on the camera frames' clock; vectors and poses already
 * in the log's axes (see [Axes]). Thread-safe: sensors arrive on their own
 * thread, frames and poses on the render thread.
 */
class MotionLog(private val make: String, private val model: String) {
    private val gyroscope = ArrayList<DoubleArray>()
    private val accelerometer = ArrayList<DoubleArray>()
    private val gravity = ArrayList<DoubleArray>()
    private val poses = ArrayList<DoubleArray>()
    private val frames = ArrayList<Pair<String, Double>>()

    @Synchronized fun addGyroscope(t: Double, v: DoubleArray) = gyroscope.add(row(t, v))

    @Synchronized fun addAccelerometer(t: Double, v: DoubleArray) = accelerometer.add(row(t, v))

    @Synchronized fun addGravity(t: Double, v: DoubleArray) = gravity.add(row(t, v))

    /** A pose from [Axes.arcorePoseToLog]: 9 rotation values by rows, then the centre. */
    @Synchronized fun addPose(t: Double, pose: DoubleArray) {
        require(pose.size == 12) { "a pose is 12 numbers" }
        poses.add(row(t, pose))
    }

    @Synchronized fun addFrame(file: String, t: Double) = frames.add(file to t)

    /** Drop the sensor readings: they turned out to be on another clock than the frames. */
    @Synchronized
    fun clearSensors() {
        gyroscope.clear()
        accelerometer.clear()
        gravity.clear()
    }

    @get:Synchronized val frameCount: Int get() = frames.size

    @Synchronized
    fun write(out: Writer) {
        out.write("{\n")
        out.write("\"format\": \"ez2digitize-motion\", \"version\": 1,\n")
        out.write("\"device\": {\"make\": ${quote(make)}, \"model\": ${quote(model)}},\n")
        out.write("\"metric\": true,\n") // ARCore's unit is the metre
        rows(out, "gyroscope", gyroscope)
        rows(out, "accelerometer", accelerometer)
        rows(out, "gravity", gravity)
        rows(out, "poses", poses)
        out.write("\"frames\": [")
        frames.forEachIndexed { i, (file, t) ->
            if (i > 0) out.write(",")
            out.write("\n  {\"file\": ${quote(file)}, \"t\": ${number(t)}}")
        }
        out.write("\n]\n}\n")
        out.flush()
    }

    private fun rows(out: Writer, key: String, list: List<DoubleArray>) {
        out.write("\"$key\": [")
        list.forEachIndexed { i, values ->
            if (i > 0) out.write(",")
            out.write("\n  [")
            out.write(values.joinToString(", ") { number(it) })
            out.write("]")
        }
        out.write("\n],\n")
    }

    private fun row(t: Double, v: DoubleArray) = doubleArrayOf(t, *v)

    companion object {
        /** Nanoseconds (Android's timestamps) to seconds. */
        fun seconds(nanos: Long): Double = nanos / 1e9

        // Twelve significant digits: times since boot (up to 10^6 s) keep microseconds.
        internal fun number(x: Double): String {
            require(x.isFinite()) { "not a finite number: $x" }
            return String.format(Locale.ROOT, "%.12g", x + 0.0).let { trim(it) } // + 0.0: no "-0"
        }

        private fun trim(s: String): String {
            if ('e' in s || '.' !in s) return s
            return s.trimEnd('0').trimEnd('.')
        }

        internal fun quote(s: String): String {
            val escaped = StringBuilder("\"")
            for (c in s) {
                when {
                    c == '"' -> escaped.append("\\\"")
                    c == '\\' -> escaped.append("\\\\")
                    c < ' ' -> escaped.append(String.format(Locale.ROOT, "\\u%04x", c.code))
                    else -> escaped.append(c)
                }
            }
            return escaped.append('"').toString()
        }
    }
}
