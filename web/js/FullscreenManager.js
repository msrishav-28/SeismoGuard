/* eslint-disable */
// FullscreenManager.js - Safe fullscreen toggle with overlay and clock
class FullscreenManager {
    constructor(){ this.isFullscreen = false; this.prevStyles = {}; this.clockTimer = null; this.init(); }
    init(){ this.injectStyles(); this.injectUI(); this.attach(); this.tickClock(); }
    injectStyles(){ const style = document.createElement('style'); style.textContent = `
        .fullscreen-toggle { position: fixed; top: 20px; right: 300px; width: 40px; height: 40px; border: none; background: var(--card-bg, rgba(255,255,255,0.1)); border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; color: var(--text-color,#fff); transition: all .3s ease; z-index: 9999; backdrop-filter: blur(10px); }
        .fullscreen-toggle:hover { transform: scale(1.08); background: var(--accent-color,#00ffff); color:#000; }
        .fullscreen-overlay { position: fixed; inset: 0 0 auto 0; height: 40px; background: rgba(0,0,0,.75); z-index: 10001; backdrop-filter: blur(10px); display: none; }
        .fullscreen-controls { height: 100%; display: flex; align-items: center; justify-content: space-between; padding: 5px 16px; color: var(--text-color,#fff); font-size: 12px; }
        .fullscreen-controls .control-btn { padding: 6px 12px; background: var(--accent-color,#00ffff); color:#000; border:0; border-radius:18px; cursor:pointer; font-weight:600; }
        body.fullscreen-mode { overflow: hidden; }
    `; document.head.appendChild(style); }
    injectUI(){ const html = `
        <button class="fullscreen-toggle" id="fullscreenToggle" title="Toggle Fullscreen (F11)">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
        <div class="fullscreen-overlay" id="fullscreenOverlay">
            <div class="fullscreen-controls">
                <button class="control-btn" id="exitFullscreenBtn">Exit Fullscreen (Esc)</button>
                <div class="fullscreen-info"><span id="fullscreenClock">--:--:--</span> <span id="fullscreenStats"></span></div>
            </div>
        </div>`; document.body.insertAdjacentHTML('beforeend', html); }
    attach(){
        const toggle = document.getElementById('fullscreenToggle'); const overlay = document.getElementById('fullscreenOverlay');
        if (toggle) toggle.addEventListener('click', ()=> this.toggle());
        const exitBtn = document.getElementById('exitFullscreenBtn'); if (exitBtn) exitBtn.addEventListener('click', ()=> this.exit());
        document.addEventListener('keydown', (e)=>{ if (e.key === 'F11'){ e.preventDefault(); this.toggle(); } });
        const onChange = ()=> { if (!document.fullscreenElement && !document.webkitFullscreenElement) this._afterExit(); };
        document.addEventListener('fullscreenchange', onChange); document.addEventListener('webkitfullscreenchange', onChange);
        this.overlay = overlay;
    }
    toggle(){ this.isFullscreen ? this.exit() : this.enter(); }
    enter(){ const docEl = document.documentElement; if (docEl.requestFullscreen) docEl.requestFullscreen(); else if (docEl.webkitRequestFullscreen) docEl.webkitRequestFullscreen();
        document.body.classList.add('fullscreen-mode'); if (this.overlay) this.overlay.style.display = 'block'; this.isFullscreen = true; this.updateStats(); }
    exit(){ if (document.exitFullscreen) document.exitFullscreen(); else if (document.webkitExitFullscreen) document.webkitExitFullscreen(); this._afterExit(); }
    _afterExit(){ document.body.classList.remove('fullscreen-mode'); if (this.overlay) this.overlay.style.display = 'none'; this.isFullscreen = false; }
    tickClock(){ const clock = ()=>{ const el = document.getElementById('fullscreenClock'); if (el) el.textContent = new Date().toLocaleTimeString(); this.updateStats(); }; clock(); this.clockTimer = setInterval(clock, 1000); }
    updateStats(){ try{ const s = window.statisticsPanel?.stats; if (!s) return; const cr = s.performance?.compressionRatio; const ev = s.events?.total; const span = document.getElementById('fullscreenStats'); if (span) span.textContent = `Events: ${ev ?? 0} | Compression: ${cr?cr.toFixed(1):'--'}:1`; }catch{} }
}

document.addEventListener('DOMContentLoaded', ()=> { window.fullscreenManager = new FullscreenManager(); });
