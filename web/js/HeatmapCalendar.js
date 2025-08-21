/* eslint-disable */
// HeatmapCalendar.js - CSS grid daily counts
class HeatmapCalendar {
    constructor(){ this.counts = {}; this.init(); }
    init(){ this.injectStyles(); this.injectUI(); this.attach(); this.refresh(); }
    injectStyles(){ const s=document.createElement('style'); s.textContent=`
        .heatmap{ position:fixed; right:20px; top:460px; width:360px; background:var(--card-bg); border-radius:12px; padding:10px; z-index:900}
        .heatgrid{ display:grid; grid-template-columns: repeat(7, 1fr); gap:4px; }
        .heatcell{ width:100%; padding-top:100%; position:relative; border-radius:4px; }
        .heatcell>div{ position:absolute; inset:0; border-radius:4px; }
    `; document.head.appendChild(s); }
    injectUI(){ const html=`<div class='heatmap'><h4 style='margin:0 0 8px;color:var(--text-color,#fff)'>Event Heatmap</h4><div id='heatgrid' class='heatgrid'></div></div>`; document.body.insertAdjacentHTML('beforeend', html); }
    attach(){ document.addEventListener('DOMNodeInserted', (e)=>{ if (e.target && e.target.classList && e.target.classList.contains('event-item')){ const ts = Number(e.target.querySelector('.event-time')?.dataset.timestamp)||Date.now(); const d = new Date(ts); const k = d.toISOString().slice(0,10); this.counts[k] = (this.counts[k]||0)+1; this.refresh(); } }); }
    refresh(){ const grid=document.getElementById('heatgrid'); if (!grid) return; grid.innerHTML=''; const today=new Date(); const start=new Date(today.getFullYear(), today.getMonth(), 1); const days=new Date(today.getFullYear(), today.getMonth()+1, 0).getDate(); for(let i=0;i<days;i++){ const d=new Date(today.getFullYear(), today.getMonth(), i+1); const k=d.toISOString().slice(0,10); const v = this.counts[k]||0; const c = Math.min(1, v/5); const cell=document.createElement('div'); cell.className='heatcell'; const inner=document.createElement('div'); inner.style.background = `rgba(0,255,255,${0.15 + 0.7*c})`; inner.title = `${k}: ${v}`; cell.appendChild(inner); grid.appendChild(cell); }
    }
}

document.addEventListener('DOMContentLoaded', ()=> { window.heatmapCalendar = new HeatmapCalendar(); });
