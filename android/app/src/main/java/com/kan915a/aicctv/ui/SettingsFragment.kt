package com.kan915a.aicctv.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.google.android.material.snackbar.Snackbar
import com.kan915a.aicctv.MonitorService
import com.kan915a.aicctv.Prefs
import com.kan915a.aicctv.R
import com.kan915a.aicctv.databinding.FragmentSettingsBinding

class SettingsFragment : Fragment() {

    private var _binding: FragmentSettingsBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSettingsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        val ctx = requireContext()
        binding.serverUrlInput.setText(Prefs.serverUrl(ctx))
        binding.sourceInput.setText(Prefs.cameraSource(ctx))
        binding.notifSwitch.isChecked = Prefs.notificationsEnabled(ctx)
        binding.monitorSwitch.isChecked = Prefs.backgroundMonitor(ctx)

        binding.saveButton.setOnClickListener {
            val url = binding.serverUrlInput.text.toString().trim()
            if (!url.startsWith("http://") && !url.startsWith("https://")) {
                Snackbar.make(binding.root, R.string.err_bad_url, Snackbar.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            Prefs.setServerUrl(ctx, url)
            Prefs.setCameraSource(ctx, binding.sourceInput.text.toString())
            Prefs.setNotificationsEnabled(ctx, binding.notifSwitch.isChecked)

            val monitor = binding.monitorSwitch.isChecked
            Prefs.setBackgroundMonitor(ctx, monitor)
            if (monitor) MonitorService.start(ctx) else MonitorService.stop(ctx)

            Snackbar.make(binding.root, R.string.saved, Snackbar.LENGTH_SHORT).show()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
