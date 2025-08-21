/* eslint-disable */
// PrintManager.js - Generate printable report from current dashboard state
class PrintManager {
    constructor(){ this.init(); }
    init(){ this.injectStyles(); this.injectButton(); this.attach(); }
    injectStyles(){ const style = document.createElement('style'); style.textContent = `
        .print-btn { position: fixed; bottom: 20px; right: 20px; width: 48px; height: 48px; border-radius: 50%; border: 0; background: var(--accent-color,#00ffff); color:#000; font-size: 20px; cursor: pointer; z-index: 9999; box-shadow: 0 6px 20px rgba(0,0,0,.3); }
        .print-btn:hover { transform: translateY(-2px); }
    `; document.head.appendChild(style); }
    injectButton(){ if (document.getElementById('printBtn')) return; const html = `<button id="printBtn" class="print-btn" title="Print Report">🖨️</button>`; document.body.insertAdjacentHTML('beforeend', html); }
    attach(){ const btn = document.getElementById('printBtn'); if (btn) btn.addEventListener('click', ()=> this.printReport()); }
    buildReportHTML(){
        const planet = window.currentPlanet || 'moon';
        const now = new Date();
        const events = window.detectedEvents || [];
        const stats = window.statisticsPanel?.stats || {};
        const wave = document.getElementById('waveformCanvas');
        const spec = document.getElementById('spectrogramCanvas');
        const perf = document.getElementById('performanceChart');
        const imgs = [wave, spec, perf].filter(Boolean).map(c => `<img src="${c.toDataURL('image/png')}" style="width:100%;max-height:300px;object-fit:contain;"/>`).join('\n');
        const styles = `
            <style>
                body{ font-family: Arial, sans-serif; color:#000; }
                h1{ margin: 0 0 10px; }
                .muted{ color:#555; }
                .grid{ display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
                .card{ border:1px solid #ddd; padding:12px; border-radius:8px; }
                table{ width:100%; border-collapse:collapse; }
                th,td{ border:1px solid #ccc; padding:6px 8px; font-size:12px; }
                .section{ margin: 16px 0; }
                @media print { .pagebreak { page-break-before: always; } }
            </style>`;
        const flatten = (o,prefix='')=>{ const out={}; for(const [k,v] of Object.entries(o||{})){ const n=prefix?`${prefix}.${k}`:k; if(v&&typeof v==='object'&&!Array.isArray(v)) Object.assign(out, flatten(v,n)); else out[n]=v; } return out; };
        const flatStats = flatten(stats);
        const statsRows = Object.entries(flatStats).map(([k,v])=>`<tr><td>${k}</td><td>${typeof v==='number'? v.toFixed? v.toFixed(3):v : String(v)}</td></tr>`).join('\n');
        const eventRows = events.map(e=>`<tr><td>${e.time||e.timestamp||''}</td><td>${e.type||''}</td><td>${e.magnitude||''}</td><td>${e.confidence||''}</td></tr>`).join('\n');
        return `<!doctype html><html><head><meta charset='utf-8'>${styles}<title>SeismoGuard Report</title></head>
        <body>
            <h1>SeismoGuard Report</h1>
            <div class='muted'>Generated ${now.toISOString()} • Planet: ${planet} • Events: ${events.length}</div>
            <div class='section grid'>
                <div class='card'><h3>Waveform</h3>${imgs.split('\n')[0]||''}</div>
                <div class='card'><h3>Spectrogram</h3>${imgs.split('\n')[1]||''}</div>
            </div>
            <div class='section card'><h3>Performance</h3>${imgs.split('\n')[2]||''}</div>
            <div class='section card pagebreak'>
                <h3>Statistics</h3>
                <table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>${statsRows}</tbody></table>
            </div>
            <div class='section card'>
                <h3>Detected Events</h3>
                <table><thead><tr><th>Time</th><th>Type</th><th>Magnitude</th><th>Confidence</th></tr></thead><tbody>${eventRows || '<tr><td colspan="4" class="muted">No events</td></tr>'}</tbody></table>
            </div>
        </body></html>`;
    }
    printReport(){ try { const html = this.buildReportHTML(); const w = window.open('', 'seismoguard_report'); if (!w) return; w.document.open(); w.document.write(html); w.document.close(); w.focus(); setTimeout(()=>{ try { w.print(); } catch {} }, 300); } catch(e) { console.error('Print failed', e); } }
}

document.addEventListener('DOMContentLoaded', ()=> { window.printManager = new PrintManager(); });
