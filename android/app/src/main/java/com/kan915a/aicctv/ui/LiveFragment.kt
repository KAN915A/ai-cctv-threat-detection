package com.kan915a.aicctv.ui

import android.graphics.Bitmap
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import com.kan915a.aicctv.Prefs
import com.kan915a.aicctv.R
import com.kan915a.aicctv.databinding.FragmentLiveBinding
import com.kan915a.aicctv.net.ApiClient
import com.kan915a.aicctv.net.MjpegStream
import com.kan915a.aicctv.net.Status
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class LiveFragment : Fragment() {

    private var _binding: FragmentLiveBinding? = null
    private val binding get() = _binding!!

    private var stream: MjpegStream? = null
    private var pollJob: Job? = null

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentLiveBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        binding.sourceInput.setText(Prefs.cameraSource(requireContext()))

        binding.startButton.setOnClickListener {
            val source = binding.sourceInput.text.toString().trim()
            if (source.isEmpty()) {
                Snackbar.make(binding.root, R.string.err_no_source, Snackbar.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            Prefs.setCameraSource(requireContext(), source)
            controlCamera { it.start(source) }
        }
        binding.stopButton.setOnClickListener {
            controlCamera { api -> api.stop(); null }
        }
    }

    private fun controlCamera(block: (ApiClient) -> Status?) {
        val api = ApiClient(Prefs.serverUrl(requireContext()))
        binding.startButton.isEnabled = false
        binding.stopButton.isEnabled = false
        viewLifecycleOwner.lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                try {
                    block(api); null
                } catch (e: Exception) {
                    e.message ?: "request failed"
                }
            }
            _binding ?: return@launch
            binding.startButton.isEnabled = true
            binding.stopButton.isEnabled = true
            if (result != null) {
                Snackbar.make(binding.root,
                    getString(R.string.err_server, result), Snackbar.LENGTH_LONG).show()
            }
            restartStream()
        }
    }

    override fun onResume() {
        super.onResume()
        restartStream()
        startPolling()
    }

    override fun onPause() {
        super.onPause()
        stream?.stop()
        stream = null
        pollJob?.cancel()
    }

    private fun restartStream() {
        stream?.stop()
        val api = ApiClient(Prefs.serverUrl(requireContext()))
        stream = MjpegStream(
            url = api.videoFeedUrl(),
            onFrame = ::postFrame,
            onError = { /* feed not running yet; polling shows state */ },
        ).also { it.start() }
    }

    private fun postFrame(bmp: Bitmap) {
        val b = _binding ?: return
        b.videoView.post {
            _binding?.videoView?.setImageBitmap(bmp)
            _binding?.placeholder?.visibility = View.GONE
        }
    }

    private fun startPolling() {
        pollJob?.cancel()
        pollJob = viewLifecycleOwner.lifecycleScope.launch {
            val api = ApiClient(Prefs.serverUrl(requireContext()))
            while (isActive) {
                val status = withContext(Dispatchers.IO) {
                    try { api.status() } catch (e: Exception) { null }
                }
                _binding?.let { render(it, status) }
                delay(1000)
            }
        }
    }

    private fun render(b: FragmentLiveBinding, status: Status?) {
        val ctx = context ?: return
        if (status == null) {
            b.statusBanner.text = getString(R.string.status_unreachable)
            b.statusBanner.setBackgroundColor(ContextCompat.getColor(ctx, R.color.level_offline))
            b.statsText.text = ""
            return
        }
        val level = status.threatLevel
        val (label, color) = when {
            !status.running -> getString(R.string.status_idle) to R.color.level_offline
            level == null -> getString(R.string.status_all_clear) to R.color.level_ok
            else -> getString(R.string.status_threat, level) to levelColor(level)
        }
        b.statusBanner.text = label
        b.statusBanner.setBackgroundColor(ContextCompat.getColor(ctx, color))

        val counts = status.counts.entries.joinToString("  ") { "${it.key}: ${it.value}" }
        val threats = status.threats.joinToString("\n") { "• [${it.level}] ${it.message}" }
        b.statsText.text = buildString {
            if (status.running) {
                append(getString(R.string.stats_line, status.fps, status.inferenceMs))
                if (counts.isNotEmpty()) append("\n").append(counts)
            }
            if (threats.isNotEmpty()) append("\n").append(threats)
            status.error?.let { append("\n").append(getString(R.string.err_server, it)) }
        }.trim()
        if (!status.running) b.placeholder.visibility = View.VISIBLE
    }

    private fun levelColor(level: String): Int = when (level.uppercase()) {
        "CRITICAL" -> R.color.level_critical
        "HIGH" -> R.color.level_high
        "MEDIUM" -> R.color.level_medium
        "LOW" -> R.color.level_low
        else -> R.color.level_ok
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
