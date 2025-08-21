/* eslint-disable */
// PredictiveBaseline.js — Simple moving average / AR(1)-like forecast overlay (Feature 15)
// Safe-by-default: OFF unless toggled. Draws a light forecast line on waveformCanvas.
(function(){
  class PredictiveBaseline {
    constructor(){
      this.enabled = false;
      this.mode = 'MA'; // 'MA' or 'AR1'
      this.window = 20;
      this.alpha = 0.6; // for AR(1)
      this._ui();
    }

    _ui(){
      const html = `
      <div id="predictivePanel" style="position:fixed;right:20px;bottom:120px;background:var(--card-bg);border:1px solid var(--card-border);padding:10px 12px;border-radius:8px;z-index:999">
        <div style="font-weight:600;margin-bottom:6px;color:var(--text-color,#fff)">Predictive Baseline</div>
        <label style="display:flex;align-items:center;gap:8px;color:var(--text-color,#fff)"><input type="checkbox" id="togglePredictive"> Enable</label>
        <div style="margin-top:6px;color:var(--text-color,#fff)">Mode:
          <select id="predMode">
            <option value="MA">Moving Avg</option>
            <option value="AR1">AR(1)</option>
          </select>
        </div>
        <div id="predMaRow" style="margin-top:6px;color:var(--text-color,#fff)">Window: <input id="predWindow" type="number" min="3" max="200" value="20" style="width:60px"></div>
        <div id="predArRow" style="margin-top:6px;color:var(--text-color,#fff);display:none">Alpha: <input id="predAlpha" type="number" step="0.05" min="0.05" max="0.95" value="0.6" style="width:60px"></div>
        <div id="predStatus" class="muted" style="margin-top:8px;color:#ccc">OFF</div>
      </div>`;
      document.addEventListener('DOMContentLoaded', ()=>{
        document.body.insertAdjacentHTML('beforeend', html);
        const t = document.getElementById('togglePredictive');
        const sel = document.getElementById('predMode');
        const win = document.getElementById('predWindow');
        const al = document.getElementById('predAlpha');
        t.addEventListener('change', (e)=>{ this.enabled = e.target.checked; this._status(this.enabled? 'ON':'OFF'); this.render(window.currentRawData||[]); });
        sel.addEventListener('change', (e)=>{ this.mode = e.target.value; document.getElementById('predMaRow').style.display = this.mode==='MA'?'block':'none'; document.getElementById('predArRow').style.display = this.mode==='AR1'?'block':'none'; this.render(window.currentRawData||[]); });
        win.addEventListener('change', (e)=>{ const v=Number(e.target.value)||20; this.window=Math.max(3,Math.min(200,v)); this.render(window.currentRawData||[]); });
        al.addEventListener('change', (e)=>{ const v=Number(e.target.value)||0.6; this.alpha=Math.max(0.05,Math.min(0.95,v)); this.render(window.currentRawData||[]); });
      });
    }

    _status(msg){ const el=document.getElementById('predStatus'); if (el) el.textContent = msg; }

    onWaveform(samples){ this.render(samples); }

    render(samples){
      if (!this.enabled) return;
      const data = (samples||[]).slice();
      if (data.length<5) return;
      let forecast;
      if (this.mode==='MA') forecast = this._movingAverage(data, this.window);
      else forecast = this._ar1(data, this.alpha);
      this._drawOverlay(forecast);
    }

    _movingAverage(data, w){
      const out = new Array(data.length);
      let sum = 0;
      for (let i=0;i<data.length;i++){
        sum += data[i];
        if (i>=w) sum -= data[i-w];
        out[i] = i>=w-1 ? sum/Math.min(i+1,w) : data[i];
      }
      return out;
    }

    _ar1(data, alpha){
      const out = new Array(data.length);
      out[0]=data[0];
      for (let i=1;i<data.length;i++) out[i] = alpha*out[i-1] + (1-alpha)*data[i-1];
      return out;
    }

    _drawOverlay(series){
      try{
        const c = document.getElementById('waveformCanvas'); if (!c) return; const ctx = c.getContext('2d');
        const W=c.width, H=c.height; if (!W||!H) return;
        // compute min/max from current data for consistent scaling with app.js renderer
        const data = window.currentRawData||[]; if (!data.length) return;
        const min = Math.min(...data), max = Math.max(...data), range=(max-min)||1;
        ctx.save();
        ctx.lineWidth = 1.5; ctx.strokeStyle = '#ffaa00'; ctx.globalAlpha=0.8; ctx.beginPath();
        for (let x=0;x<W;x++){
          const idx = Math.floor((x/W) * (series.length-1));
          const v = (series[idx]-min)/range; const y = H - v*(H*0.9) - H*0.05;
          if (x===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }
        ctx.stroke(); ctx.restore();
      }catch{}
    }
  }

  document.addEventListener('DOMContentLoaded', ()=>{ window.predictiveBaseline = new PredictiveBaseline(); });
})();
