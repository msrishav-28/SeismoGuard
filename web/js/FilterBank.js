/* eslint-disable */
// FilterBank.js - Visualization-only frequency filters
class FilterBank {
    constructor(){ this.bands=[{name:'Low',f:[0,1]},{name:'Mid',f:[1,5]},{name:'High',f:[5,20]}]; this.active=new Set(); this.lastWave=[]; this.init(); }
    init(){ this.injectUI(); this.attach(); }
    injectUI(){ const html=`<div id="filterBank" style="position:fixed;left:200px;bottom:80px;background:var(--card-bg);padding:10px;border-radius:8px;z-index:999">${this.bands.map((b,i)=>`<label style='margin-right:8px;color:var(--text-color,#fff)'><input type='checkbox' class='fb' data-i='${i}'> ${b.name}</label>`).join('')}</div>`; document.body.insertAdjacentHTML('beforeend', html); }
    attach(){ document.querySelectorAll('#filterBank .fb').forEach(cb=> cb.addEventListener('change',(e)=>{ const i=Number(e.target.dataset.i); if(e.target.checked) this.active.add(i); else this.active.delete(i); this.render(); })); }
    onWaveform(samples){ this.lastWave = samples.slice(); this.render(); }
    render(){ try{ const c=document.getElementById('spectrogramCanvas'); if(!c) return; const ctx=c.getContext('2d'); const W=c.width, H=c.height; const data=this.lastWave; if(!data || data.length<8){ return; } // compute naive DFT
        const N = Math.min(256, data.length); const step = Math.floor(data.length/N); const xs = Array.from({length:N}, (_,k)=> data[k*step]); const mags = new Array(N).fill(0); for(let k=0;k<N;k++){ let re=0,im=0; for(let n=0;n<N;n++){ const angle=-2*Math.PI*k*n/N; const v=xs[n]; re+=v*Math.cos(angle); im+=v*Math.sin(angle);} mags[k]=Math.sqrt(re*re+im*im);} const sr=100; const hz = mags.map((_,k)=> k*(sr/(N*step))); ctx.save(); ctx.globalAlpha=0.25; ctx.lineWidth=2; const colors=['#00ffff','#ff00aa','#ffaa00']; [...this.active].forEach((idx,ci)=>{ const [f0,f1]=this.bands[idx].f; ctx.strokeStyle=colors[ci%colors.length]; ctx.beginPath(); for(let x=0;x<W;x++){ const f = x/W*(sr/2); const gain = (f>=f0 && f<=f1) ? 1 : 0; const y = H - (gain*H*0.9 + H*0.05); if(x===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); } ctx.stroke(); }); ctx.restore(); }catch{}
    }
}

document.addEventListener('DOMContentLoaded', ()=> { window.filterBank = new FilterBank(); });
