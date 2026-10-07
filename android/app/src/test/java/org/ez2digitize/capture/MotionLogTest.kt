// SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
// SPDX-License-Identifier: GPL-3.0-or-later
package org.ez2digitize.capture

import java.io.StringWriter
import org.junit.Assert.assertEquals
import org.junit.Test

class MotionLogTest {
    /**
     * The log for fixed readings must equal sample.motion.json, which the
     * desktop's tests read with its own parser (tests/test_android_app.py):
     * the two sides agree on the format through this file.
     */
    @Test
    fun writesTheSampleLog() {
        val log = MotionLog("Google", "Pixel \"8\"")
        val start = 12_345.678_901
        for (i in 0 until 3) {
            val t = start + i * 0.005
            log.addGyroscope(t, doubleArrayOf(0.0, 0.5, 0.0))
            log.addAccelerometer(t, doubleArrayOf(-9.6, 0.0, 0.0))
            log.addGravity(t, doubleArrayOf(-9.81, 0.0, 0.0))
        }
        val columnMajor = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0.25f, 1.5f, -0.5f, 1f)
        log.addPose(start + 0.004, Axes.arcorePoseToLog(columnMajor))
        log.addFrame("IMG_0001.jpg", start + 0.004)
        val out = StringWriter()
        log.write(out)
        val expected = javaClass.getResource("/sample.motion.json")!!.readText()
        assertEquals(expected, out.toString())
    }

    @Test
    fun numbersAndStrings() {
        assertEquals("12345.678901", MotionLog.number(12_345.678_901))
        assertEquals("0", MotionLog.number(0.0))
        assertEquals("-9.81", MotionLog.number(-9.81))
        assertEquals("\"a\\\"b\\\\c\\u000a\"", MotionLog.quote("a\"b\\c\n"))
        assertEquals(1.5, MotionLog.seconds(1_500_000_000L), 0.0)
    }
}
