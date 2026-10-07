// SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
// SPDX-License-Identifier: GPL-3.0-or-later
package org.ez2digitize.capture

/**
 * Conversions into the axes the motion log uses (docs/COMPANION.md):
 * the camera's axes as the image is stored (x right, y down, z forward),
 * and a world with y down along gravity. Pure functions, unit-tested.
 */
object Axes {
    /**
     * A vector in Android's device axes (x right, y up, z out of the screen,
     * phone in its natural portrait) in the back camera's image axes.
     *
     * `sensorOrientation` is the camera's SENSOR_ORIENTATION (0, 90, 180 or
     * 270): how far the stored image must be turned clockwise to stand
     * upright on the screen. At 0 the image is the screen: x = x, y = -y;
     * each further quarter turn maps (a, b) to (b, -a). The back camera looks
     * out of the back, so z = -z.
     */
    fun deviceToImage(v: FloatArray, sensorOrientation: Int): DoubleArray {
        var a = v[0].toDouble()
        var b = -v[1].toDouble()
        repeat(Math.floorMod(sensorOrientation, 360) / 90) {
            val turned = b to -a
            a = turned.first
            b = turned.second
        }
        return doubleArrayOf(a, b, -v[2].toDouble())
    }

    /**
     * ARCore's camera pose (camera to world, by rows, as `Pose.toMatrix`'s
     * column-major 4 x 4 gives it) in the log's axes: rotation by rows, then
     * the camera centre.
     *
     * ARCore's world has y up, and its camera looks along -z with y up;
     * with F = diag(1, -1, -1) the log's rotation is F R F and its centre F c.
     */
    fun arcorePoseToLog(columnMajor: FloatArray): DoubleArray {
        fun r(row: Int, col: Int) = columnMajor[col * 4 + row].toDouble()
        val f = doubleArrayOf(1.0, -1.0, -1.0)
        val out = DoubleArray(12)
        for (row in 0 until 3) {
            for (col in 0 until 3) {
                out[row * 3 + col] = f[row] * r(row, col) * f[col]
            }
            out[9 + row] = f[row] * r(row, 3)
        }
        return out
    }
}
