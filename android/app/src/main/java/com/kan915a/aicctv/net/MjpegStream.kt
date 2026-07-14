package com.kan915a.aicctv.net

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * Reads a multipart MJPEG stream (the server's /video_feed) on a background
 * thread and delivers decoded frames. Frames arrive already annotated with
 * detection boxes drawn by the server.
 */
class MjpegStream(
    private val url: String,
    private val onFrame: (Bitmap) -> Unit,
    private val onError: (String) -> Unit,
) {
    @Volatile private var running = false
    private var thread: Thread? = null

    fun start() {
        if (running) return
        running = true
        thread = Thread(::run, "mjpeg-reader").also {
            it.isDaemon = true
            it.start()
        }
    }

    fun stop() {
        running = false
        thread = null
    }

    private fun run() {
        val client = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.MILLISECONDS) // endless stream
            .build()
        try {
            val req = Request.Builder().url(url).build()
            client.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) throw java.io.IOException("HTTP ${resp.code}")
                val input = resp.body?.byteStream()
                    ?: throw java.io.IOException("empty body")
                val buf = ByteArrayOutputStream(64 * 1024)
                val chunk = ByteArray(16 * 1024)
                while (running) {
                    val n = input.read(chunk)
                    if (n < 0) break
                    buf.write(chunk, 0, n)
                    extractFrames(buf)
                    if (buf.size() > 4 * 1024 * 1024) buf.reset() // corrupt guard
                }
            }
        } catch (e: Exception) {
            if (running) onError(e.message ?: "stream error")
        }
    }

    /** Pull every complete JPEG (SOI..EOI) out of the buffer, decode the last one. */
    private fun extractFrames(buf: ByteArrayOutputStream) {
        val data = buf.toByteArray()
        var searchFrom = 0
        var lastFrameEnd = -1
        var latest: Bitmap? = null
        while (true) {
            val soi = indexOfMarker(data, searchFrom, 0xD8)
            if (soi < 0) break
            val eoi = indexOfMarker(data, soi + 2, 0xD9)
            if (eoi < 0) break
            val len = eoi + 2 - soi
            val bmp = BitmapFactory.decodeByteArray(data, soi, len)
            if (bmp != null) latest = bmp
            lastFrameEnd = eoi + 2
            searchFrom = eoi + 2
        }
        if (lastFrameEnd > 0) {
            buf.reset()
            buf.write(data, lastFrameEnd, data.size - lastFrameEnd)
        }
        latest?.let { if (running) onFrame(it) }
    }

    /** Find 0xFF followed by [second] starting at [from]. */
    private fun indexOfMarker(data: ByteArray, from: Int, second: Int): Int {
        var i = maxOf(from, 0)
        while (i < data.size - 1) {
            if (data[i] == 0xFF.toByte() && data[i + 1] == second.toByte()) return i
            i++
        }
        return -1
    }
}
