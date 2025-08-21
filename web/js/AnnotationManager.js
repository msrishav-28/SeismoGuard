/* eslint-disable */
// AnnotationManager.js - MVP annotations on waveform canvas
class AnnotationManager {
    constructor(){ this.ann=[]; this.active=null; this.init(); }
    init(){ this.injectStyles(); this.attach(); }
    injectStyles(){ const s=document.createElement('style'); s.textContent=`.ann-badge{position:fixed;left:20px;top:80px;background:var(--card-bg);color:var(--text-color);padding:6px 10px;border-radius:12px;z-index:999}`; document.head.appendChild(s); document.body.insertAdjacentHTML('beforeend', '<div class="ann-badge">Annotation: hold Shift then drag on waveform</div>'); }
    attach(){ const c=document.getElementById('waveformCanvas'); if(!c) return; c.addEventListener('pointerdown', (e)=>{ if(!e.shiftKey) return; const r=c.getBoundingClientRect(); const x=e.clientX-r.left; this.active={x0:x,x1:x, note:''}; }); c.addEventListener('pointermove',(e)=>{ if(!this.active) return; const r=c.getBoundingClientRect(); this.active.x1=e.clientX-r.left; this.render(); }); c.addEventListener('pointerup',()=>{ if(!this.active) return; const seg={...this.active, id:`a_${Date.now()}`}; this.ann.push(seg); this.active=null; this.save(); this.render(true); }); window.addEventListener('resize',()=> this.render(true)); this.load(); this.render(true); }
    toTime(x){ const c=document.getElementById('waveformCanvas'); const W=c?c.width:1; const base=(window.currentRawData||[]).length; const idx=Math.floor(x/W*base); const ts=Date.now()+idx*10; return window.timestampManager? window.timestampManager.formatTimestamp(ts): new Date(ts).toLocaleTimeString(); }
    render(full=false){ const c=document.getElementById('waveformCanvas'); if(!c) return; const ctx=c.getContext('2d'); if(full){ // redraw waveform via app hook
        if (window.updateWaveform) window.updateWaveform(window.currentRawData||[]);
    }
    // overlay annotations
    try{ const W=c.width, H=c.height; ctx.save(); ctx.globalAlpha=0.3; ctx.fillStyle='#ff00aa'; [...this.ann, ...(this.active?[this.active]:[])].forEach(a=>{ const x0=Math.min(a.x0,a.x1), x1=Math.max(a.x0,a.x1); ctx.fillRect(x0,0,Math.max(2,x1-x0),H); }); ctx.restore(); }catch{}
    }
    save(){ try{ localStorage.setItem('sg_annotations', JSON.stringify(this.ann)); }catch{} }
    load(){ try{ this.ann = JSON.parse(localStorage.getItem('sg_annotations')||'[]'); }catch{ this.ann=[]; } }
    export(){ const data=this.ann.map(a=>({start:this.toTime(a.x0), end:this.toTime(a.x1)})); const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=`annotations_${Date.now()}.json`; a.click(); URL.revokeObjectURL(a.href); }
}

document.addEventListener('DOMContentLoaded', ()=> { window.annotationManager = new AnnotationManager(); });
