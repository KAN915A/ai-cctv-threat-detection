package com.kan915a.aicctv

import android.content.Context
import android.content.SharedPreferences

object Prefs {
    private const val FILE = "aicctv_prefs"

    private fun sp(ctx: Context): SharedPreferences =
        ctx.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    fun serverUrl(ctx: Context): String =
        sp(ctx).getString("server_url", "http://192.168.1.100:8000")!!.trimEnd('/')

    fun setServerUrl(ctx: Context, v: String) =
        sp(ctx).edit().putString("server_url", v.trim().trimEnd('/')).apply()

    fun cameraSource(ctx: Context): String =
        sp(ctx).getString("camera_source", "0")!!

    fun setCameraSource(ctx: Context, v: String) =
        sp(ctx).edit().putString("camera_source", v.trim()).apply()

    fun notificationsEnabled(ctx: Context): Boolean =
        sp(ctx).getBoolean("notifications_enabled", true)

    fun setNotificationsEnabled(ctx: Context, v: Boolean) =
        sp(ctx).edit().putBoolean("notifications_enabled", v).apply()

    fun backgroundMonitor(ctx: Context): Boolean =
        sp(ctx).getBoolean("background_monitor", false)

    fun setBackgroundMonitor(ctx: Context, v: Boolean) =
        sp(ctx).edit().putBoolean("background_monitor", v).apply()
}
