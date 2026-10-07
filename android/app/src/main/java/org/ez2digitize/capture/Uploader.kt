// SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
// SPDX-License-Identifier: GPL-3.0-or-later
package org.ez2digitize.capture

import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.UUID
import org.json.JSONObject

/**
 * Sends a capture to the desktop's upload page, as docs/COMPANION.md
 * describes and tools/companion/send_capture.py does: check the API
 * version, describe the capture, send each file in resumable chunks, say
 * it is done. Blocking: call it off the main thread.
 *
 * `base` is the URL in the desktop's QR code, ending in `/`.
 */
class Uploader(base: String, private val progress: (String) -> Unit = {}) {
    private val base = if (base.endsWith("/")) base else "$base/"

    class UploadError(message: String) : IOException(message)

    /** The server's description; [UploadError] if this app can't talk to it. */
    fun checkApi(): JSONObject {
        val (status, body) = request("GET", "api")
        if (status == 404) throw UploadError("No upload page there, or a desktop app too old for this one")
        val api = parse(body) ?: throw UploadError("Not an EZ2DIGITIZE upload page (HTTP $status)")
        if (api.optString("app") != "EZ2DIGITIZE") throw UploadError("Not an EZ2DIGITIZE upload page")
        if (api.optInt("api") != API_VERSION) {
            throw UploadError("The desktop speaks API ${api.optInt("api")}, this app $API_VERSION: update one of them")
        }
        return api
    }

    fun describeCapture(make: String, model: String, app: String, flipped: Boolean) {
        val capture = JSONObject()
            .put("source", "android")
            .put("device", JSONObject().put("make", make).put("model", model))
            .put("app", app)
            .put("flipped", flipped)
        val (status, body) = request("POST", "capture", capture.toString().toByteArray())
        if (status != 200) throw UploadError("Capture description refused: ${parse(body)?.optString("error")}")
    }

    fun sendAll(files: List<File>, chunkLimit: Int) {
        val chunk = minOf(CHUNK, chunkLimit)
        files.forEachIndexed { i, file ->
            progress("Sending ${i + 1} of ${files.size}: ${file.name}")
            send(file, chunk)
        }
        request("POST", "done", ByteArray(0))
    }

    /** One file, from wherever the server says it has got to after a failure. */
    private fun send(file: File, chunk: Int) {
        val id = UUID.randomUUID().toString().replace("-", "")
        val size = file.length()
        val query = "name=${URLEncoder.encode(file.name, "UTF-8")}&size=$size"
        var offset = 0L
        var failures = 0
        RandomAccessFile(file, "r").use { input ->
            val buffer = ByteArray(chunk)
            while (offset < size) {
                input.seek(offset)
                val count = input.read(buffer, 0, minOf(chunk.toLong(), size - offset).toInt())
                val data = buffer.copyOf(count)
                val answer = try {
                    request("PUT", "files/$id?offset=$offset&$query", data)
                } catch (e: IOException) {
                    0 to ""
                }
                val (status, body) = answer
                val received = parse(body)?.optLong("received", -1) ?: -1
                when {
                    (status == 200 || status == 409) && received >= 0 -> {
                        offset = received
                        failures = if (status == 200) 0 else failures + 1
                    }
                    status == 400 -> throw UploadError("${file.name} refused: ${parse(body)?.optString("error")}")
                    else -> {
                        failures++
                        Thread.sleep(minOf(1000L shl failures, 30_000L))
                        offset = runCatching { parse(request("GET", "files/$id").second)?.optLong("received") }
                            .getOrNull() ?: offset
                    }
                }
                if (failures > RETRIES) throw UploadError("${file.name}: gave up after $RETRIES failures")
            }
        }
    }

    private fun request(method: String, path: String, body: ByteArray? = null): Pair<Int, String> {
        val connection = URL(base + path).openConnection() as HttpURLConnection
        try {
            connection.requestMethod = method
            connection.connectTimeout = 10_000
            connection.readTimeout = 60_000
            if (body != null) {
                connection.doOutput = true
                connection.setFixedLengthStreamingMode(body.size)
                if (method == "POST") connection.setRequestProperty("Content-Type", "application/json")
                connection.outputStream.use { it.write(body) }
            }
            val status = connection.responseCode
            val stream = if (status < 400) connection.inputStream else connection.errorStream
            val text = stream?.use { it.readBytes().toString(Charsets.UTF_8) } ?: ""
            return status to text
        } finally {
            connection.disconnect()
        }
    }

    private fun parse(text: String): JSONObject? = runCatching { JSONObject(text) }.getOrNull()

    companion object {
        const val API_VERSION = 1
        const val CHUNK = 4 * 1024 * 1024
        const val RETRIES = 5
    }
}
