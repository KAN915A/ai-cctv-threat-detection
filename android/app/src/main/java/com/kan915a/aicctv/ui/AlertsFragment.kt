package com.kan915a.aicctv.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.kan915a.aicctv.Prefs
import com.kan915a.aicctv.databinding.FragmentAlertsBinding
import com.kan915a.aicctv.net.ApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AlertsFragment : Fragment() {

    private var _binding: FragmentAlertsBinding? = null
    private val binding get() = _binding!!
    private lateinit var adapter: EventAdapter

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentAlertsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        adapter = EventAdapter(ApiClient(Prefs.serverUrl(requireContext())))
        binding.recycler.layoutManager = LinearLayoutManager(requireContext())
        binding.recycler.adapter = adapter
        binding.swipeRefresh.setOnRefreshListener { load() }
        load()
    }

    private fun load() {
        val api = ApiClient(Prefs.serverUrl(requireContext()))
        viewLifecycleOwner.lifecycleScope.launch {
            val events = withContext(Dispatchers.IO) {
                try { api.events(100) } catch (e: Exception) { null }
            }
            val b = _binding ?: return@launch
            b.swipeRefresh.isRefreshing = false
            if (events == null) {
                b.emptyText.visibility = View.VISIBLE
                b.emptyText.text = getString(com.kan915a.aicctv.R.string.status_unreachable)
            } else {
                b.emptyText.visibility = if (events.isEmpty()) View.VISIBLE else View.GONE
                adapter.submit(events)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
