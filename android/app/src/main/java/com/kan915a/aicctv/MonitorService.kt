package com.kan915a.aicctv

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.kan915a.aicctv.net.ApiClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Foreground service that polls the detection server and raises a
 * notification whenever the threat level escalates to HIGH or CRITICAL.
 */
class MonitorService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var job: Job? = null
    private var lastNotifiedLevel: String? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createChannels()
        startForeground(ONGOING_ID, ongoingNotification())
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (job == null) job = scope.launch { poll() }
        return START_STICKY
    }

    private suspend fun poll() {
        while (scope.isActive) {
            try {
                val api = ApiClient(Prefs.serverUrl(this))
                val status = api.status()
                val level = status.threatLevel?.uppercase()
                if (level in ALERT_LEVELS && level != lastNotifiedLevel &&
                    Prefs.notificationsEnabled(this)
                ) {
                    val msg = status.threats.firstOrNull()?.message
                        ?: getString(R.string.notif_generic, level)
                    notifyThreat(level!!, msg)
                }
                lastNotifiedLevel = if (level in ALERT_LEVELS) level else null
            } catch (_: Exception) {
                // server unreachable; keep trying
            }
            delay(3000)
        }
    }

    private fun notifyThreat(level: String, message: String) {
        val intent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notif = NotificationCompat.Builder(this, CHANNEL_ALERTS)
            .setSmallIcon(R.drawable.ic_shield)
            .setContentTitle(getString(R.string.notif_title, level))
            .setContentText(message)
            .setStyle(NotificationCompat.BigTextStyle().bigText(message))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setContentIntent(intent)
            .setAutoCancel(true)
            .build()
        manager().notify(ALERT_ID, notif)
    }

    private fun ongoingNotification(): Notification =
        NotificationCompat.Builder(this, CHANNEL_MONITOR)
            .setSmallIcon(R.drawable.ic_shield)
            .setContentTitle(getString(R.string.notif_monitoring))
            .setContentText(Prefs.serverUrl(this))
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

    private fun createChannels() {
        if (Build.VERSION.SDK_INT < 26) return
        val m = manager()
        m.createNotificationChannel(
            NotificationChannel(
                CHANNEL_MONITOR, getString(R.string.channel_monitor),
                NotificationManager.IMPORTANCE_LOW,
            )
        )
        m.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ALERTS, getString(R.string.channel_alerts),
                NotificationManager.IMPORTANCE_HIGH,
            )
        )
    }

    private fun manager() =
        getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        private const val CHANNEL_MONITOR = "monitor"
        private const val CHANNEL_ALERTS = "alerts"
        private const val ONGOING_ID = 1
        private const val ALERT_ID = 2
        private val ALERT_LEVELS = setOf("HIGH", "CRITICAL")

        fun start(ctx: Context) {
            val intent = Intent(ctx, MonitorService::class.java)
            if (Build.VERSION.SDK_INT >= 26) ctx.startForegroundService(intent)
            else ctx.startService(intent)
        }

        fun stop(ctx: Context) {
            ctx.stopService(Intent(ctx, MonitorService::class.java))
        }
    }
}
