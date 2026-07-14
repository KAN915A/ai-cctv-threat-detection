package com.kan915a.aicctv.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.kan915a.aicctv.R
import com.kan915a.aicctv.databinding.ItemEventBinding
import com.kan915a.aicctv.net.ApiClient
import com.kan915a.aicctv.net.EventItem
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class EventAdapter(private val api: ApiClient) :
    RecyclerView.Adapter<EventAdapter.Holder>() {

    private val items = ArrayList<EventItem>()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val http = OkHttpClient()
    private val cache = LruCache<String, Bitmap>(32)
    private val timeFormat = SimpleDateFormat("MMM d, HH:mm:ss", Locale.getDefault())

    fun submit(events: List<EventItem>) {
        items.clear()
        items.addAll(events)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder {
        val b = ItemEventBinding.inflate(
            LayoutInflater.from(parent.context), parent, false)
        return Holder(b)
    }

    override fun getItemCount() = items.size

    override fun onBindViewHolder(holder: Holder, position: Int) =
        holder.bind(items[position])

    inner class Holder(private val b: ItemEventBinding) :
        RecyclerView.ViewHolder(b.root) {

        fun bind(e: EventItem) {
            b.levelChip.text = e.level
            b.levelChip.setBackgroundColor(
                ContextCompat.getColor(b.root.context, levelColor(e.level)))
            b.messageText.text = e.message
            b.timeText.text = formatTs(e.ts)

            b.thumb.setImageResource(R.drawable.ic_videocam)
            b.thumb.tag = e.snapshot
            val snap = e.snapshot ?: return
            cache.get(snap)?.let { b.thumb.setImageBitmap(it); return }
            scope.launch {
                val bmp = withContext(Dispatchers.IO) { fetch(snap) }
                if (bmp != null && b.thumb.tag == snap) {
                    b.thumb.setImageBitmap(bmp)
                }
            }
        }
    }

    private fun fetch(name: String): Bitmap? = try {
        val req = Request.Builder().url(api.snapshotUrl(name)).build()
        http.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) null
            else resp.body?.bytes()?.let { bytes ->
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    ?.also { cache.put(name, it) }
            }
        }
    } catch (e: Exception) { null }

    private fun formatTs(ts: String): String {
        val secs = ts.toDoubleOrNull() ?: return ts
        return timeFormat.format(Date((secs * 1000).toLong()))
    }

    private fun levelColor(level: String): Int = when (level.uppercase()) {
        "CRITICAL" -> R.color.level_critical
        "HIGH" -> R.color.level_high
        "MEDIUM" -> R.color.level_medium
        "LOW" -> R.color.level_low
        else -> R.color.level_ok
    }
}
