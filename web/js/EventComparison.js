/* eslint-disable */
// EventComparison.js - Simple comparison table and overlay
class EventComparison {
    constructor(){ this.selected = new Set(); this.init(); }
    init(){ this.injectStyles(); this.injectUI(); this.attach(); }
    injectStyles(){ const s = document.createElement('style'); s.textContent = `
        .compare-panel{ position:fixed; right:20px; bottom:80px; width:360px; background:var(--card-bg,rgba(255,255,255,.08)); border-radius:12px; padding:12px; z-index:999; backdrop-filter:blur(10px);} 
        .compare-panel h4{ margin:0 0 8px; color:var(--text-color,#fff);} 
        .compare-table{ width:100%; font-size:12px; border-collapse:collapse; }
        .compare-table th,.compare-table td{ border:1px solid rgba(255,255,255,.15); padding:4px 6px; color:var(--text-color,#fff);} 
        .compare-actions{ display:flex; gap:8px; margin-top:8px; }
        .overlay-canvas{ position:fixed; left:20px; bottom:20px; width:420px; height:160px; background:rgba(0,0,0,.3); border-radius:8px; padding:8px; }
    `; document.head.appendChild(s); }
    injectUI(){ const html = `
        <div class="compare-panel" id="comparePanel">
            <h4>Event Comparison</h4>
            <table class="compare-table"><thead><tr><th>Sel</th><th>Time</th><th>Mag</th><th>Type</th></tr></thead><tbody id="compareBody"></tbody></table>
            <div class="compare-actions"><button class="btn" id="compareOverlayBtn">Overlay</button><button class="btn btn-secondary" id="compareExportBtn">Export JSON</button></div>
        </div>
        <canvas id="compareOverlay" class="overlay-canvas"></canvas>`; document.body.insertAdjacentHTML('beforeend', html); this.canvas = document.getElementById('compareOverlay'); this.ctx = this.canvas.getContext('2d'); }
    attach(){
        document.addEventListener('change', (e)=>{
            if (e.target && e.target.classList.contains('event-select')){ const row = e.target.closest('.event-item'); const id = row?.getAttribute('data-event-id'); if (!id) return; if (e.target.checked) this.selected.add(id); else this.selected.delete(id); this.refreshTable(); }
        });
        document.getElementById('compareOverlayBtn').addEventListener('click', ()=> this.drawOverlay());
        document.getElementById('compareExportBtn').addEventListener('click', ()=> this.exportJSON());
    }
    refreshTable(){ const body = document.getElementById('compareBody'); if (!body) return; const evs = (window.detectedEvents||[]).filter(e=> this.selected.has(e.id)); body.innerHTML = evs.map(e=> `<tr><td>✓</td><td>${window.timestampManager? window.timestampManager.formatTimestamp(e.timestamp): e.time}</td><td>${e.magnitude?.toFixed? e.magnitude.toFixed(1): e.magnitude}</td><td>${e.type}</td></tr>`).join(''); }
    drawOverlay(){ try{ const ctx = this.ctx; if (!ctx) return; const W = this.canvas.width = this.canvas.offsetWidth; const H = this.canvas.height = this.canvas.offsetHeight; ctx.clearRect(0,0,W,H); const data = (window.currentRawData||[]); if (data.length<10) return; const series = Array.from(this.selected).map(()=> this.normalize(data)); const colors = ['#00ffff','#ff00aa','#ffaa00','#00ff88']; series.forEach((arr,i)=>{ ctx.strokeStyle=colors[i%colors.length]; ctx.beginPath(); for(let x=0;x<W;x++){ const idx = Math.floor(x/W*arr.length); const v = arr[idx]; const y = H - (v*H*0.8 + H*0.1); if(x===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);} ctx.stroke(); }); }catch{}
    }
    normalize(data){ const min=Math.min(...data), max=Math.max(...data); const r = (max-min)||1; return data.map(v=> (v-min)/r ); }
    exportJSON(){ const out = (window.detectedEvents||[]).filter(e=> this.selected.has(e.id)); const blob = new Blob([JSON.stringify(out,null,2)], {type:'application/json'}); const a = document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=`event_comparison_${Date.now()}.json`; a.click(); URL.revokeObjectURL(a.href); }
}

document.addEventListener('DOMContentLoaded', ()=> { window.eventComparison = new EventComparison(); });
