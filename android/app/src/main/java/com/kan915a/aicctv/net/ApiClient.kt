package com.kan915a.aicctv.net

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

data class ThreatItem(val level: String, val kind: String, val message: String)

data class Status(
    val running: Boolean,
    val threatLevel: String?,
    val threats: List<ThreatItem>,
    val counts: Map<String, Int>,
    val fps: Double,
    val inferenceMs: Double,
    val error: String?,
)

data class EventItem(
    val ts: String,
    val level: String,
    val kind: String,
    val message: String,
    val snapshot: String?,
)

/** Thin blocking client for the FastAPI detection server. Call from Dispatchers.IO. */
class ApiClient(private val baseUrl: String) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    private val json = "application/json".toMediaType()

    private fun get(path: String): String {
        val req = Request.Builder().url("$baseUrl$path").build()
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw java.io.IOException("HTTP ${resp.code}")
            return resp.body?.string() ?: ""
        }
    }

    private fun post(path: String, body: String): String {
        val req = Request.Builder()
            .url("$baseUrl$path")
            .post(body.toRequestBody(json))
            .build()
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw java.io.IOException("HTTP ${resp.code}")
            return resp.body?.string() ?: ""
        }
    }

    fun status(): Status = parseStatus(JSONObject(get("/api/status")))

    fun start(source: String): Status {
        val payload = JSONObject().put("source", source).toString()
        val obj = JSONObject(post("/api/start", payload))
        return Status(
            running = obj.optBoolean("running", false),
            threatLevel = null, threats = emptyList(), counts = emptyMap(),
            fps = 0.0, inferenceMs = 0.0,
            error = if (obj.isNull("error")) null else obj.optString("error"),
        )
    }

    fun stop() {
        post("/api/stop", "{}")
    }

    fun events(limit: Int = 50): List<EventItem> {
        val arr = org.json.JSONArray(get("/api/events?limit=$limit"))
        val out = ArrayList<EventItem>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            out.add(
                EventItem(
                    ts = o.opt("ts")?.toString() ?: "",
                    level = o.optString("level", "?"),
                    kind = o.optString("kind", ""),
                    message = o.optString("message", ""),
                    snapshot = if (o.isNull("snapshot")) null else o.optString("snapshot"),
                )
            )
        }
        return out
    }

    fun snapshotUrl(name: String): String = "$baseUrl/snapshots/$name"

    fun videoFeedUrl(): String = "$baseUrl/video_feed"

    companion object {
        fun parseStatus(obj: JSONObject): Status {
            val threats = ArrayList<ThreatItem>()
            obj.optJSONArray("threats")?.let { arr ->
                for (i in 0 until arr.length()) {
                    val t = arr.getJSONObject(i)
                    threats.add(
                        ThreatItem(
                            level = t.optString("level", "?"),
                            kind = t.optString("kind", ""),
                            message = t.optString("message", ""),
                        )
                    )
                }
            }
            val counts = LinkedHashMap<String, Int>()
            obj.optJSONObject("counts")?.let { c ->
                for (key in c.keys()) counts[key] = c.optInt(key)
            }
            return Status(
                running = obj.optBoolean("running", false),
                threatLevel = if (obj.isNull("threat_level")) null
                              else obj.optString("threat_level"),
                threats = threats,
                counts = counts,
                fps = obj.optDouble("fps", 0.0),
                inferenceMs = obj.optDouble("inference_ms", 0.0),
                error = if (obj.isNull("error")) null else obj.optString("error"),
            )
        }
    }
}
