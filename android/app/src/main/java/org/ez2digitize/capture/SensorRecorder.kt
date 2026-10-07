// SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
// SPDX-License-Identifier: GPL-3.0-or-later
package org.ez2digitize.capture

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

/**
 * Records the gyroscope, accelerometer and gravity sensors into a
 * [MotionLog], converted into the back camera's image axes.
 *
 * Sensor and camera timestamps share one clock on most phones (the
 * camera's SENSOR_INFO_TIMESTAMP_SOURCE is REALTIME); [MainActivity]
 * checks that before trusting the log.
 */
class SensorRecorder(
    private val sensors: SensorManager,
    private val log: MotionLog,
    private val sensorOrientation: Int,
) : SensorEventListener {
    fun start() {
        for (type in listOf(Sensor.TYPE_GYROSCOPE, Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GRAVITY)) {
            sensors.getDefaultSensor(type)?.let {
                sensors.registerListener(this, it, SAMPLING_US)
            }
        }
    }

    fun stop() = sensors.unregisterListener(this)

    override fun onSensorChanged(event: SensorEvent) {
        val t = MotionLog.seconds(event.timestamp)
        val v = Axes.deviceToImage(event.values, sensorOrientation)
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> log.addGyroscope(t, v)
            Sensor.TYPE_ACCELEROMETER -> log.addAccelerometer(t, v)
            Sensor.TYPE_GRAVITY -> log.addGravity(t, v)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) = Unit

    companion object {
        const val SAMPLING_US = 5_000 // 200 Hz asked; phones give what they can
    }
}
