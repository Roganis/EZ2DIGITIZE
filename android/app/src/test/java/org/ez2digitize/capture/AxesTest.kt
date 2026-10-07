// SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
// SPDX-License-Identifier: GPL-3.0-or-later
package org.ez2digitize.capture

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class AxesTest {
    private fun image(v: FloatArray, orientation: Int) = Axes.deviceToImage(v, orientation)

    @Test
    fun uprightPortraitPhoneSeesGravityAlongImageX() {
        // Upright in portrait, at rest: the accelerometer pushes up the device's y.
        val push = floatArrayOf(0f, 9.81f, 0f)
        // The usual back camera (90°): the stored image is landscape and the
        // scene's up is its left side, so the push is along -x, down along +x
        // (docs/COMPANION.md: a portrait photo's "down" is close to [1, 0, 0]).
        assertArrayEquals(doubleArrayOf(-9.81, 0.0, 0.0), image(push, 90), 1e-6)
        // No rotation: the image is the screen, y down.
        assertArrayEquals(doubleArrayOf(0.0, -9.81, 0.0), image(push, 0), 1e-6)
        assertArrayEquals(doubleArrayOf(9.81, 0.0, 0.0), image(push, 270), 1e-6)
        assertArrayEquals(doubleArrayOf(0.0, 9.81, 0.0), image(push, 180), 1e-6)
    }

    @Test
    fun backCameraLooksOutOfTheBack() {
        // Out of the screen (towards the user) is backwards for the back camera.
        assertArrayEquals(doubleArrayOf(0.0, 0.0, -1.0), image(floatArrayOf(0f, 0f, 1f), 90), 1e-9)
    }

    @Test
    fun imageAxesStayRightHanded() {
        for (orientation in listOf(0, 90, 180, 270)) {
            val x = image(floatArrayOf(1f, 0f, 0f), orientation)
            val y = image(floatArrayOf(0f, 1f, 0f), orientation)
            val z = image(floatArrayOf(0f, 0f, 1f), orientation)
            assertArrayEquals(z, cross(x, y), 1e-9)
        }
    }

    @Test
    fun arcoreIdentityPoseFlipsToTheLogsAxes() {
        // ARCore's camera at (1, 2, 3), looking along the world's -z with y up.
        val columnMajor = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 1f, 2f, 3f, 1f)
        val pose = Axes.arcorePoseToLog(columnMajor)
        // F R F with R = I is I: the camera's OpenCV z (forward) maps to the
        // world's +z in the flipped world, the same direction as ARCore's -z.
        val identity = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assertArrayEquals(identity, pose.copyOfRange(0, 9), 1e-9)
        assertArrayEquals(doubleArrayOf(1.0, -2.0, -3.0), pose.copyOfRange(9, 12), 1e-9)
    }

    @Test
    fun arcoreTurnedPoseKeepsItsViewingDirection() {
        // Turned 90° about ARCore's world y (up): its -z (forward) points to world -x.
        val c = 0f
        val s = 1f
        // Rotation about y by +90°: x -> -z, z -> x (column-major, by columns).
        val columnMajor = floatArrayOf(c, 0f, -s, 0f, 0f, 1f, 0f, 0f, s, 0f, c, 0f, 0f, 0f, 0f, 1f)
        val pose = Axes.arcorePoseToLog(columnMajor)
        // In the log, the camera's forward (OpenCV z) is the rotation's third
        // column: world -x in ARCore's frame is -x in the log's too.
        val forward = doubleArrayOf(pose[2], pose[5], pose[8])
        assertArrayEquals(doubleArrayOf(-1.0, 0.0, 0.0), forward, 1e-6)
        // And the log's world y is down: ARCore's up (0, 1, 0) is (0, -1, 0).
        val up = doubleArrayOf(pose[1], pose[4], pose[7]) // the camera's y (down) in the world
        assertArrayEquals(doubleArrayOf(0.0, 1.0, 0.0), up, 1e-6)
    }

    private fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
}
