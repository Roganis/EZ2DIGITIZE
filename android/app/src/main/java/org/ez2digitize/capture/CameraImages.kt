// SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
// SPDX-License-Identifier: GPL-3.0-or-later
package org.ez2digitize.capture

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import com.google.zxing.BinaryBitmap
import com.google.zxing.MultiFormatReader
import com.google.zxing.NotFoundException
import com.google.zxing.PlanarYUVLuminanceSource
import com.google.zxing.common.HybridBinarizer
import java.io.File
import java.io.FileOutputStream

/** What is done with ARCore's CPU camera images (YUV_420_888): photos and QR codes. */
object CameraImages {
    /**
     * Save a camera image (see [toNv21]) as a JPEG, as the sensor delivered it (not turned upright:
     * the motion log's axes are the stored image's).
     *
     * TODO: ARCore's CPU image is at most the camera config's CPU size (often
     * 1920 x 1080 or less). Full-resolution photos need ARCore's shared
     * camera (Camera2 still captures while ARCore tracks); see README.md.
     */
    fun saveJpeg(nv21: Nv21, file: File, quality: Int = 95) {
        val yuv = YuvImage(nv21.data, ImageFormat.NV21, nv21.width, nv21.height, null)
        FileOutputStream(file).use { yuv.compressToJpeg(Rect(0, 0, nv21.width, nv21.height), quality, it) }
    }

    /** A copy of a camera image's pixels, so the image can go back to ARCore at once. */
    class Nv21(val data: ByteArray, val width: Int, val height: Int)

    /** The text of a QR code in `image`, or null: the desktop's upload URL. */
    fun readQrCode(image: Image): String? {
        val plane = image.planes[0]
        val width = image.width
        val height = image.height
        val luminance = ByteArray(width * height)
        val buffer = plane.buffer.duplicate()
        for (row in 0 until height) {
            buffer.position(row * plane.rowStride)
            if (plane.pixelStride == 1) {
                buffer.get(luminance, row * width, width)
            } else {
                for (col in 0 until width) {
                    luminance[row * width + col] = buffer.get(row * plane.rowStride + col * plane.pixelStride)
                }
            }
        }
        val source = PlanarYUVLuminanceSource(luminance, width, height, 0, 0, width, height, false)
        return try {
            MultiFormatReader().decode(BinaryBitmap(HybridBinarizer(source))).text
        } catch (e: NotFoundException) {
            null
        }
    }

    /** YUV_420_888 (any strides) to NV21: Y, then interleaved V and U. */
    fun toNv21(image: Image): Nv21 {
        val width = image.width
        val height = image.height
        val out = ByteArray(width * height * 3 / 2)
        val (y, u, v) = image.planes
        var at = 0
        val yBuffer = y.buffer.duplicate()
        for (row in 0 until height) {
            for (col in 0 until width) {
                out[at++] = yBuffer.get(row * y.rowStride + col * y.pixelStride)
            }
        }
        val uBuffer = u.buffer.duplicate()
        val vBuffer = v.buffer.duplicate()
        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                out[at++] = vBuffer.get(row * v.rowStride + col * v.pixelStride)
                out[at++] = uBuffer.get(row * u.rowStride + col * u.pixelStride)
            }
        }
        return Nv21(out, width, height)
    }
}
