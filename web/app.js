/*
	SeismoGuard Frontend Glue Layer
	- Desktop-only integration, no rewrites of existing modules
	- Maps WS data to canvases and metrics; exposes minimal hooks
*/

(function(){
	const state = {
		waveform: [],
		lastSpectrogram: null,
	};

	// Expose hooks expected by WebSocketClient and other modules
	window.updateWaveform = function(samples){
		state.waveform = samples.slice(-1000);
		window.currentRawData = state.waveform;
		drawWaveform();
		if (window.statisticsPanel) window.statisticsPanel.updateStatistics(state.waveform);
		updateMetricsFromData(state.waveform);
			// Notify optional filter bank module
			if (window.filterBank && typeof window.filterBank.onWaveform === 'function') {
				window.filterBank.onWaveform(state.waveform);
			}
			// Notify predictive baseline overlay
			if (window.predictiveBaseline && typeof window.predictiveBaseline.onWaveform === 'function'){
				window.predictiveBaseline.onWaveform(state.waveform);
			}
	};

	window.addEventToList = function(event){
		const list = document.getElementById('eventList');
		if (!list) return;
		const item = document.createElement('div');
		item.className = 'event-item';
		const ts = Number(event.timestamp || Date.now());
			const id = `ev_${ts}_${Math.floor(Math.random()*1e6)}`;
			item.setAttribute('data-event-id', id);
			item.innerHTML = `<input type="checkbox" class="event-select" title="Select for comparison" style="margin-right:8px;">
				<span class="event-time" data-timestamp="${ts}">${new Date(ts).toLocaleTimeString()}</span>
			<span class="event-magnitude">M${(event.magnitude ?? 0).toFixed ? event.magnitude.toFixed(1) : event.magnitude}</span>
			<div style="margin-top:10px; color:#ccc;">Type: ${event.type || event.event_type || 'unknown'} • Confidence: ${(event.confidence ?? 0).toFixed ? event.confidence.toFixed(1) : event.confidence}%</div>`;
		list.prepend(item);
		if (!window.detectedEvents) window.detectedEvents = [];
		window.detectedEvents.unshift({
			id,
			timestamp: ts,
			time: item.querySelector('.event-time').textContent,
			magnitude: Number(event.magnitude ?? 0),
			type: event.type || event.event_type || 'unknown',
			confidence: Number(event.confidence ?? 0)
		});
		const det = document.getElementById('detectionStatus');
		if (det) det.textContent = `${window.detectedEvents.length} Events Detected`;
		// Notify clustering to recompute labels when enabled
		if (window.eventClustering && typeof window.eventClustering.onEventsUpdated==='function'){
			window.eventClustering.onEventsUpdated(window.detectedEvents);
		}
	};

	// Canvas helpers
	function getCtx(id){ const c = document.getElementById(id); return c ? c.getContext('2d') : null; }
	function clearCanvas(ctx){ if (!ctx) return; ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); }

	function drawWaveform(){
		const ctx = getCtx('waveformCanvas'); if (!ctx) return;
		const { width, height } = ctx.canvas;
		clearCanvas(ctx);
		// Grid
		ctx.strokeStyle = 'rgba(0,255,255,0.1)'; ctx.lineWidth = 1;
		for (let i=0;i<10;i++){ ctx.beginPath(); ctx.moveTo(0, i*height/10); ctx.lineTo(width, i*height/10); ctx.stroke(); }
		// Data
		const data = state.waveform; if (!data || data.length === 0) return;
		const min = Math.min(...data), max = Math.max(...data);
		const range = (max - min) || 1;
		ctx.strokeStyle = '#00ffff'; ctx.lineWidth = 2; ctx.beginPath();
		for (let x=0; x<width; x++){
			const idx = Math.floor((x/width) * (data.length-1));
			const v = (data[idx] - min) / range; // 0..1
			const y = height - v * (height*0.9) - height*0.05;
			if (x===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
		}
		ctx.stroke();
		animateProgress();
	}

	function drawSpectrogram(){
		const ctx = getCtx('spectrogramCanvas'); if (!ctx) return;
		const { width, height } = ctx.canvas;
		clearCanvas(ctx);
		// Simple synthetic spectrogram based on waveform variability
		const data = state.waveform; if (!data || data.length < 10) return;
		const segments = Math.max(50, Math.floor(width/4));
		for (let x = 0; x < segments; x++){
			const start = Math.floor((x/segments) * data.length);
			const end = Math.min(data.length, start + Math.floor(data.length/segments));
			const slice = data.slice(start, end);
			const amp = slice.reduce((s,v)=> s + Math.abs(v), 0) / (slice.length || 1);
			const intensity = Math.min(1, amp * 1e9 * 0.5);
			for (let y=0; y<height; y+=2){
				const rowFactor = y/height;
				const val = Math.max(0, intensity - Math.abs(rowFactor-0.5));
				if (val > 0.05){
					const hue = 180 + val*180; ctx.fillStyle = `hsla(${hue},100%,50%,${val})`;
					ctx.fillRect(Math.floor(x*(width/segments)), y, Math.ceil(width/segments), 2);
				}
			}
		}
	}

	function updateMetricsFromData(data){
		if (!data || !data.length) return;
		const absMax = Math.max(...data.map(Math.abs));
		const mean = data.reduce((s,v)=> s+v,0)/data.length;
		const variance = data.reduce((s,v)=> s + (v-mean)*(v-mean),0)/data.length;
		const std = Math.sqrt(variance);
		const snr = std > 0 ? Math.max(0, 10 * Math.log10((std*std)/(1e-18))) : 0; // heuristic vs. 1e-9 noise
		const freq = 0.5 + (absMax*1e9) * 5; // heuristic for demo
		setText('snrValue', snr.toFixed(1));
		setText('frequencyValue', freq.toFixed(2));
		setText('amplitudeValue', (absMax*1e9).toFixed(0));
		setText('compressionRatio', '85:1');
		setText('dataSaved', '98.8');
		drawSpectrogram();
	}

	function setText(id, text){ const el = document.getElementById(id); if (el) el.textContent = text; }
	function animateProgress(){ const fill = document.getElementById('processingProgress'); if (!fill) return; let p = 0; const i = setInterval(()=>{ p+=5; fill.style.width = p+'%'; fill.textContent = p+'%'; if (p>=100) clearInterval(i); }, 60); }

	// Initialize after DOM ready
	document.addEventListener('DOMContentLoaded', () => {
		// Ensure canvases have proper dimensions (handled in inline script too)
		const wf = document.getElementById('waveformCanvas'); const sp = document.getElementById('spectrogramCanvas');
		if (wf) { wf.width = wf.offsetWidth; wf.height = 300; }
		if (sp) { sp.width = sp.offsetWidth; sp.height = 300; }

		// If wsClient is present, start streaming; else, we rely on simulation buttons
		if (window.wsClient && typeof window.wsClient.startStreaming === 'function'){
			window.wsClient.startStreaming('seismic_data');
		}

		// Light binding to improve alignment with server
		if (window.wsClient){
			window.wsClient.subscribe('seismic_data', (buffer)=>{
				// Already handled via updateWaveform called inside ws client if defined
			});
			window.wsClient.subscribe('processing_result', (payload)=>{
				const list = payload?.events || [];
				list.forEach(ev => window.addEventToList(ev));
			});
		}

		// Route Export Results button to ExportManager modal if available
		window.exportResults = function(){
			if (window.exportManager && typeof window.exportManager.openModal === 'function'){
				window.exportManager.openModal();
			} else {
				// Fallback to basic JSON download
				const payload = { planet: window.currentPlanet || 'moon', timestamp: new Date().toISOString(), events: window.detectedEvents || [] };
				const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
				const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `seismoguard_events_${Date.now()}.json`; a.click(); URL.revokeObjectURL(a.href);
			}
		};
	});
})();

