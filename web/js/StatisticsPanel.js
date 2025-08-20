/* eslint-disable */
// StatisticsPanel.js - Enhanced live stats and charts (Chart.js)
/**
 * Displays live statistics and manages charts for signal/event/performance.
 * @class StatisticsPanel
 */
class StatisticsPanel {
    constructor() {
        this.stats = {
            raw: { mean: 0, median: 0, std: 0, min: 0, max: 0, rms: 0, kurtosis: 0, skewness: 0 },
            processed: { snr: 0, peakFrequency: 0, dominantPeriod: 0, energy: 0, zeroCrossings: 0, spectralCentroid: 0 },
            events: { total: 0, byType: {}, avgMagnitude: 0, avgDuration: 0, detectionRate: 0, falsePositiveRate: 0 },
            performance: { processingTime: 0, dataPoints: 0, compressionRatio: 0, memoryUsage: 0, cpuUsage: 0 }
        };
        this.charts = {};
        this.updateInterval = null;
        this.init();
    }
    init() { this.createPanel(); this.initCharts(); this.startMonitoring(); }
    createPanel() {
        const html = `
        <div class="statistics-panel" id="statisticsPanel">
            <div class="stats-header"><h3>📊 Advanced Statistics</h3><button class="stats-toggle" id="statsToggle">▼</button></div>
            <div class="stats-content" id="statsContent">
                <div class="stats-tabs">
                    <button class="tab-btn active" data-tab="signal">Signal</button>
                    <button class="tab-btn" data-tab="spectral">Spectral</button>
                    <button class="tab-btn" data-tab="events">Events</button>
                    <button class="tab-btn" data-tab="performance">Performance</button>
                </div>
                <div class="tab-content active" id="signal-tab">
                    <div class="stat-grid">
                        <div class="stat-item"><label>Mean</label><span id="stat-mean">--</span></div>
                        <div class="stat-item"><label>Median</label><span id="stat-median">--</span></div>
                        <div class="stat-item"><label>Std Dev</label><span id="stat-std">--</span></div>
                        <div class="stat-item"><label>Min/Max</label><span id="stat-range">--</span></div>
                        <div class="stat-item"><label>RMS</label><span id="stat-rms">--</span></div>
                        <div class="stat-item"><label>Kurtosis</label><span id="stat-kurtosis">--</span></div>
                        <div class="stat-item"><label>Skewness</label><span id="stat-skewness">--</span></div>
                        <div class="stat-item"><label>Zero Crossings</label><span id="stat-zero">--</span></div>
                    </div>
                    <canvas id="histogramChart" width="300" height="150"></canvas>
                </div>
                <div class="tab-content" id="spectral-tab">
                    <div class="stat-grid">
                        <div class="stat-item"><label>Peak Frequency</label><span id="stat-peak-freq">-- Hz</span></div>
                        <div class="stat-item"><label>Spectral Centroid</label><span id="stat-centroid">-- Hz</span></div>
                        <div class="stat-item"><label>Bandwidth</label><span id="stat-bandwidth">-- Hz</span></div>
                        <div class="stat-item"><label>Energy</label><span id="stat-energy">--</span></div>
                    </div>
                    <canvas id="spectrumChart" width="300" height="150"></canvas>
                </div>
                <div class="tab-content" id="events-tab">
                    <div class="stat-grid">
                        <div class="stat-item"><label>Total Events</label><span id="stat-total-events">0</span></div>
                        <div class="stat-item"><label>Detection Rate</label><span id="stat-detection-rate">--%</span></div>
                        <div class="stat-item"><label>Avg Magnitude</label><span id="stat-avg-mag">--</span></div>
                        <div class="stat-item"><label>Avg Duration</label><span id="stat-avg-duration">-- s</span></div>
                    </div>
                    <canvas id="eventTypeChart" width="300" height="150"></canvas>
                </div>
                <div class="tab-content" id="performance-tab">
                    <div class="stat-grid">
                        <div class="stat-item"><label>Processing Time</label><span id="stat-proc-time">-- ms</span></div>
                        <div class="stat-item"><label>Data Points</label><span id="stat-data-points">--</span></div>
                        <div class="stat-item"><label>Compression</label><span id="stat-compression">--:1</span></div>
                        <div class="stat-item"><label>Memory</label><span id="stat-memory">-- MB</span></div>
                    </div>
                    <canvas id="performanceChart2" width="300" height="150"></canvas>
                </div>
            </div>
        </div>`;
        const style = document.createElement('style');
        style.textContent = `
            .statistics-panel { position: fixed; right: 20px; top: 80px; width: 350px; background: var(--card-bg, rgba(255,255,255,0.08)); backdrop-filter: blur(10px); border-radius: 15px; padding: 20px; z-index: 1000; transition: all 0.3s ease; max-height: 600px; overflow-y: auto; }
            .stats-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; color: var(--text-color, #fff); }
            .stats-toggle { background: none; border: none; color: var(--text-color, #fff); cursor: pointer; font-size: 20px; transition: transform 0.3s ease; }
            .stats-toggle.collapsed { transform: rotate(-90deg); }
            .stats-tabs { display: flex; gap: 5px; margin-bottom: 15px; }
            .tab-btn { flex: 1; padding: 8px; background: rgba(0,0,0,0.2); border: 1px solid var(--accent-color, #00ffff); color: var(--text-color, #fff); cursor: pointer; border-radius: 5px; transition: all 0.3s ease; }
            .tab-btn.active { background: var(--accent-color, #00ffff); color: #000; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 15px; }
            .stat-item { background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; display: flex; flex-direction: column; }
            .stat-item label { font-size: 0.8em; color: var(--text-color, #888); margin-bottom: 5px; }
            .stat-item span { font-size: 1.2em; font-weight: bold; color: var(--accent-color, #00ffff); }
            .statistics-panel canvas { background: rgba(0, 0, 0, 0.3); border-radius: 8px; padding: 10px; }
        `;
        document.head.appendChild(style);
        document.body.insertAdjacentHTML('beforeend', html);
        this.attachListeners();
    }
    attachListeners() {
        document.getElementById('statsToggle').addEventListener('click', (e) => {
            const content = document.getElementById('statsContent');
            const t = e.target; if (content.style.display === 'none') { content.style.display = 'block'; t.classList.remove('collapsed'); } else { content.style.display = 'none'; t.classList.add('collapsed'); }
        });
        document.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active'); document.getElementById(`${btn.dataset.tab}-tab`).classList.add('active');
        }));
    }
    updateStatistics(data) {
        if (!data || !data.length) return;
        const sorted = [...data].sort((a,b)=>a-b); const n = data.length;
        this.stats.raw.mean = data.reduce((a,b)=>a+b,0) / n;
        this.stats.raw.median = sorted[Math.floor(n/2)];
        this.stats.raw.min = sorted[0]; this.stats.raw.max = sorted[n-1];
        const variance = data.reduce((s,x)=> s + Math.pow(x - this.stats.raw.mean,2),0) / n; this.stats.raw.std = Math.sqrt(variance);
        this.stats.raw.rms = Math.sqrt(data.reduce((s,x)=> s + x*x,0) / n);
        const m4 = data.reduce((s,x)=> s + Math.pow((x - this.stats.raw.mean)/this.stats.raw.std,4),0) / n; this.stats.raw.kurtosis = m4 - 3;
        const m3 = data.reduce((s,x)=> s + Math.pow((x - this.stats.raw.mean)/this.stats.raw.std,3),0) / n; this.stats.raw.skewness = m3;
        this.stats.processed.zeroCrossings = 0; for (let i=1;i<data.length;i++){ if ((data[i]>=0 && data[i-1]<0) || (data[i]<0 && data[i-1]>=0)) this.stats.processed.zeroCrossings++; }
        // P1: Lightweight spectral metrics (naive DFT on downsample)
        try {
            const sampleRate = 100; // UI heuristic to match backend default
            const maxBins = 128; // compute on a small subset for speed
            const step = Math.max(1, Math.floor(data.length / maxBins));
            const xs = []; for (let i=0;i<maxBins && i*step < data.length; i++) xs.push(data[i*step]);
            const N = xs.length; if (N > 8) {
                const mags = new Array(N).fill(0);
                for (let k=0;k<N;k++){
                    let re=0, im=0; for (let n=0;n<N;n++){ const angle = -2*Math.PI*k*n/N; const v = xs[n]; re += v*Math.cos(angle); im += v*Math.sin(angle); }
                    mags[k] = Math.sqrt(re*re + im*im);
                }
                const freqs = mags.map((_,k)=> k * (sampleRate / (N*step)));
                let peakIdx = 0; for (let i=1;i<mags.length;i++){ if (mags[i] > mags[peakIdx]) peakIdx = i; }
                const sumMag = mags.reduce((a,b)=>a+b,0) || 1;
                const centroid = mags.reduce((s,m,i)=> s + m*freqs[i], 0) / sumMag;
                this.stats.processed.peakFrequency = freqs[peakIdx] || 0;
                this.stats.processed.spectralCentroid = centroid || 0;
                this.stats.processed.energy = mags.reduce((s,m)=> s + m*m, 0);
            }
        } catch {}
        this.updateUI(); this.updateCharts(data);
    }
    updateUI() {
        const s = this.stats;
        document.getElementById('stat-mean').textContent = s.raw.mean.toExponential(2);
        document.getElementById('stat-median').textContent = s.raw.median.toExponential(2);
        document.getElementById('stat-std').textContent = s.raw.std.toExponential(2);
        document.getElementById('stat-range').textContent = `${s.raw.min.toExponential(2)} / ${s.raw.max.toExponential(2)}`;
        document.getElementById('stat-rms').textContent = s.raw.rms.toExponential(2);
        document.getElementById('stat-kurtosis').textContent = s.raw.kurtosis.toFixed(2);
        document.getElementById('stat-skewness').textContent = s.raw.skewness.toFixed(2);
        document.getElementById('stat-zero').textContent = s.processed.zeroCrossings;
        document.getElementById('stat-peak-freq').textContent = `${s.processed.peakFrequency.toFixed(2)} Hz`;
        document.getElementById('stat-centroid').textContent = `${s.processed.spectralCentroid.toFixed(2)} Hz`;
        document.getElementById('stat-energy').textContent = s.processed.energy.toExponential(2);
        document.getElementById('stat-total-events').textContent = s.events.total;
        document.getElementById('stat-detection-rate').textContent = `${s.events.detectionRate.toFixed(1)}%`;
        document.getElementById('stat-avg-mag').textContent = s.events.avgMagnitude.toFixed(2);
        document.getElementById('stat-avg-duration').textContent = `${s.events.avgDuration.toFixed(1)} s`;
        document.getElementById('stat-proc-time').textContent = `${s.performance.processingTime.toFixed(1)} ms`;
        document.getElementById('stat-data-points').textContent = s.performance.dataPoints.toLocaleString();
        document.getElementById('stat-compression').textContent = `${s.performance.compressionRatio.toFixed(1)}:1`;
        document.getElementById('stat-memory').textContent = `${(s.performance.memoryUsage/1024/1024).toFixed(1)} MB`;
    }
    initCharts() {
        if (!window.Chart) return;
        const ctx = document.getElementById('histogramChart').getContext('2d');
        this.charts.histogram = new Chart(ctx, { type: 'bar', data: { labels: [], datasets: [{ label: 'Amplitude Distribution', data: [], backgroundColor: 'rgba(0,255,255,0.5)', borderColor: 'rgba(0,255,255,1)', borderWidth: 1 }]}, options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } } });
    }
    updateCharts(data) {
        if (!this.charts.histogram) return;
        const bins = 20; const min = Math.min(...data); const max = Math.max(...data); const binWidth = (max - min) / bins; const histogram = new Array(bins).fill(0);
        data.forEach(v => { const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1); histogram[idx]++; });
        this.charts.histogram.data.labels = histogram.map((_,i)=> (min + i*binWidth).toExponential(1));
        this.charts.histogram.data.datasets[0].data = histogram; this.charts.histogram.update();
    }
    startMonitoring() { this.updateInterval = setInterval(() => { if (performance.memory) this.stats.performance.memoryUsage = performance.memory.usedJSHeapSize; }, 1000); }
    updateFromServer(stats) {
        // Map backend stats payload to panel structure safely
        try {
            if (!stats || typeof stats !== 'object') return;
            // Events
            if (typeof stats.events_detected === 'number') {
                this.stats.events.total = stats.events_detected;
            }
            // Data points
            if (typeof stats.total_data_points === 'number') {
                this.stats.performance.dataPoints = stats.total_data_points;
            }
            // Compression (bytes-aware preferred)
            const rawB = Number(stats.raw_bytes) || 0;
            const compB = Number(stats.compressed_bytes) || 0;
            let ratio = Number(stats.compression_ratio) || 0;
            if (rawB > 0 && compB > 0) {
                ratio = rawB / compB;
            }
            if (ratio && isFinite(ratio)) {
                this.stats.performance.compressionRatio = ratio;
            }
            // Update UI elements also used outside the panel
            const crEl = document.getElementById('compressionRatio');
            if (crEl && ratio && isFinite(ratio)) crEl.textContent = `${ratio.toFixed(1)}:1`;
            const dsEl = document.getElementById('dataSaved');
            if (dsEl && rawB > 0 && compB >= 0) {
                const savedPct = Math.max(0, Math.min(100, 100 * (1 - (compB / rawB))));
                dsEl.textContent = savedPct.toFixed(1);
            }
        } catch (e) {
            // Keep UI resilient on malformed payloads
            // console.error('Stats mapping error', e);
        }
        this.updateUI();
    }
}

document.addEventListener('DOMContentLoaded', () => { window.statisticsPanel = new StatisticsPanel(); });
