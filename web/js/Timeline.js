/* eslint-disable */
// Timeline.js - Lightweight SVG mission timeline
class Timeline {
    constructor(){ this.palette=['#00ffff','#ff6b6b','#ffd166','#06d6a0','#a26bff']; this.init(); }
    init(){ this.injectUI(); this.render(); }
    injectUI(){ const html = `<div id="timeline" style="position:fixed;left:20px;top:140px;right:20px;height:80px;background:var(--card-bg);border:1px solid var(--card-border);border-radius:8px;padding:6px;z-index:900"><svg id="timelineSvg" width="100%" height="100%"></svg></div>`; document.body.insertAdjacentHTML('beforeend', html); }
    render(){ try{ const svg = document.getElementById('timelineSvg'); if(!svg) return; const w=svg.clientWidth||800, h=svg.clientHeight||60; svg.innerHTML=''; const ns='http://www.w3.org/2000/svg'; const evs=(window.detectedEvents||[]).slice(0,100); if(!evs.length) return; const minT=Math.min(...evs.map(e=> e.timestamp||Date.now())); const maxT=Math.max(...evs.map(e=> e.timestamp||Date.now())); const span = (maxT-minT)||1; evs.forEach((e,i)=>{ const x = ((e.timestamp - minT)/span) * w; const y = h/2 + Math.sin(i)*10; const c = document.createElementNS(ns,'circle'); c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', 4); const col = (typeof e.cluster==='number') ? this.palette[e.cluster % this.palette.length] : '#00ffff'; c.setAttribute('fill', col); c.setAttribute('opacity','0.9'); c.setAttribute('title', (e.type||'event') + (typeof e.cluster==='number' ? ` • C${e.cluster+1}`:'')); svg.appendChild(c); }); }catch{}
    }
}

document.addEventListener('DOMContentLoaded', ()=> { window.timeline = new Timeline(); });
