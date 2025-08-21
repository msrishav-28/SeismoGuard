/* eslint-disable */
// TimestampManager.js - Unified timestamp formatting and UI control
class TimestampManager {
    constructor(){
        this.modes = ['local','utc','iso','unix','relative'];
        this.mode = localStorage.getItem('sg_timestamp_mode') || 'local';
        this.missionStart = Number(localStorage.getItem('sg_mission_start')) || Date.now();
        this.init();
    }
    init(){ this.injectStyles(); this.injectUI(); this.attach(); }
    injectStyles(){ const style = document.createElement('style'); style.textContent = `
        .timestamp-toggle { position: fixed; top: 20px; right: 240px; height: 40px; background: var(--card-bg, rgba(255,255,255,0.1)); color: var(--text-color,#fff); border: 1px solid var(--accent-color,#00ffff); border-radius: 20px; padding: 6px 10px; z-index: 9999; backdrop-filter: blur(10px); }
        .timestamp-toggle select { background: transparent; border: none; color: inherit; font-weight: 600; cursor: pointer; }
    `; document.head.appendChild(style); }
    injectUI(){ const html = `
        <div class="timestamp-toggle" id="timestampToggle" title="Timestamp display">
            <label style="margin-right:8px; opacity:0.8">Time</label>
            <select id="timestampMode">
                <option value="local">Local</option>
                <option value="utc">UTC</option>
                <option value="iso">ISO</option>
                <option value="unix">Unix</option>
                <option value="relative">T+ (relative)</option>
            </select>
        </div>`; document.body.insertAdjacentHTML('beforeend', html); const sel = document.getElementById('timestampMode'); if (sel) sel.value = this.mode; }
    attach(){ const sel = document.getElementById('timestampMode'); if (sel) sel.addEventListener('change', (e)=>{ this.setMode(e.target.value); });
        document.addEventListener('keydown', (e)=>{ if (e.key.toLowerCase() === 't'){ this.cycleMode(); }});
    }
    setMode(mode){ if (!this.modes.includes(mode)) return; this.mode = mode; localStorage.setItem('sg_timestamp_mode', mode); this.reformatEventList(); }
    cycleMode(){ const i = this.modes.indexOf(this.mode); const next = this.modes[(i+1) % this.modes.length]; this.setMode(next); const sel = document.getElementById('timestampMode'); if (sel) sel.value = next; }
    setMissionStart(ts){ this.missionStart = Number(ts) || Date.now(); localStorage.setItem('sg_mission_start', String(this.missionStart)); this.reformatEventList(); }
    formatTimestamp(input){ let ts = input; if (input && typeof input === 'object'){ if ('timestamp' in input) ts = input.timestamp; else if ('time' in input) ts = input.time; }
        if (typeof ts === 'string' && /^T\+/.test(ts)) return ts; // already relative
        if (typeof ts === 'string' && !/\d{4}-\d{2}-\d{2}T/.test(ts)){ const d = new Date(ts); ts = d.getTime(); }
        if (typeof ts !== 'number') ts = Date.now(); const d = new Date(ts);
        switch(this.mode){
            case 'local': return d.toLocaleTimeString();
            case 'utc': return d.toUTCString().split(' ')[4] + ' UTC';
            case 'iso': return d.toISOString();
            case 'unix': return Math.floor(ts/1000).toString();
            case 'relative': default: { const sec = Math.max(0, Math.round((ts - this.missionStart)/1000)); return `T+${sec}s`; }
        }
    }
    reformatEventList(){ try { const items = document.querySelectorAll('#eventList .event-item .event-time'); items.forEach(span => { const ts = Number(span.dataset.timestamp); if (ts) span.textContent = this.formatTimestamp(ts); }); } catch {}
    }
}

document.addEventListener('DOMContentLoaded', ()=> { window.timestampManager = new TimestampManager(); });
