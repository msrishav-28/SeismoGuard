/* eslint-disable */
// Clustering.js — Lightweight K-Means clustering for detected events (Feature 11)
// Safe-by-default: OFF unless toggled on. No heavy deps. Works on current detectedEvents array.
(function(){
  class EventClustering {
    constructor(){
      this.enabled = false;
      this.k = 3;
      this.palette = ['#00ffff','#ff6b6b','#ffd166','#06d6a0','#a26bff'];
      this._ui();
    }

    _ui(){
      const html = `
      <div id="clusteringPanel" style="position:fixed;right:20px;bottom:200px;background:var(--card-bg);border:1px solid var(--card-border);padding:10px 12px;border-radius:8px;z-index:999">
        <div style="font-weight:600;margin-bottom:6px;color:var(--text-color,#fff)">Clustering</div>
        <label style="display:flex;align-items:center;gap:8px;color:var(--text-color,#fff)">
          <input type="checkbox" id="toggleClustering"> Enable
        </label>
        <div style="margin-top:8px;color:var(--text-color,#fff)">
          k: <input id="clusterK" type="number" min="2" max="5" value="3" style="width:50px"> 
          <button id="reclusterBtn" class="btn btn-secondary" style="margin-left:8px;padding:4px 8px">Re-cluster</button>
        </div>
        <div id="clusterStatus" class="muted" style="margin-top:8px;color:#ccc">OFF</div>
      </div>`;
      document.addEventListener('DOMContentLoaded', ()=>{
        document.body.insertAdjacentHTML('beforeend', html);
        document.getElementById('toggleClustering').addEventListener('change', (e)=>{
          this.enabled = e.target.checked;
          this._status(this.enabled ? 'ON' : 'OFF');
          if (this.enabled) this.onEventsUpdated(window.detectedEvents||[]);
        });
        document.getElementById('clusterK').addEventListener('change', (e)=>{
          const v = Number(e.target.value)||3; this.k = Math.max(2, Math.min(5, v));
        });
        document.getElementById('reclusterBtn').addEventListener('click', ()=>{
          if (this.enabled) this.onEventsUpdated(window.detectedEvents||[]);
        });
      });
    }

    _status(msg){ const el = document.getElementById('clusterStatus'); if (el) el.textContent = msg; }

    onEventsUpdated(events){
      if (!this.enabled) return;
      const evs = (events||[]).filter(e=> typeof e.magnitude==='number' && e.timestamp);
      if (evs.length < this.k){ this._status(`Need ≥${this.k} events (have ${evs.length})`); return; }
      // Feature extraction: [magnitude, time_norm, confidence, type_code]
      const minT = Math.min(...evs.map(e=> e.timestamp));
      const maxT = Math.max(...evs.map(e=> e.timestamp));
      const span = (maxT-minT)||1;
      const typeMap = new Map();
      const feats = evs.map(e=>{
        if (!typeMap.has(e.type)) typeMap.set(e.type, typeMap.size);
        const t = (e.timestamp - minT)/span;
        const conf = typeof e.confidence==='number' ? e.confidence/100 : 0.9;
        const typec = (typeMap.get(e.type)||0)/4; // normalize small integer
        return [e.magnitude||0, t, conf, typec];
      });
      const labels = this._kmeans(feats, this.k, 6);
      // Persist cluster labels back onto detectedEvents by id, then update DOM labels/colors
      evs.forEach((e,i)=>{ e.cluster = labels[i]; });
      this._renderLabels();
      this._status(`Clustered ${evs.length} events into ${this.k} groups`);
      // Optional: notify others (e.g., Timeline) to re-render
      if (window.timeline && typeof window.timeline.render==='function') window.timeline.render();
    }

    _kmeans(X, k, iters=8){
      const n = X.length, d = X[0].length;
      // Init centers as k random distinct points
      const idxs = Array.from({length:n}, (_,i)=>i).sort(()=> Math.random()-0.5).slice(0,k);
      let C = idxs.map(i=> X[i].slice());
      let labels = new Array(n).fill(0);
      const dist2 = (a,b)=>{ let s=0; for(let j=0;j<d;j++){ const t=a[j]-b[j]; s+=t*t; } return s; };
      for (let it=0; it<iters; it++){
        // assign
        for (let i=0;i<n;i++){
          let best=0, bestd=Infinity;
          for (let c=0;c<k;c++){ const dd = dist2(X[i], C[c]); if (dd<bestd){ bestd=dd; best=c; } }
          labels[i]=best;
        }
        // update
        const sums = Array.from({length:k}, ()=> new Array(d).fill(0));
        const counts = new Array(k).fill(0);
        for (let i=0;i<n;i++){ const lab=labels[i]; counts[lab]++; for(let j=0;j<d;j++){ sums[lab][j]+=X[i][j]; } }
        for (let c=0;c<k;c++){
          if (counts[c]===0) continue;
          for (let j=0;j<d;j++){ sums[c][j]/=counts[c]; }
          C[c]=sums[c];
        }
      }
      return labels;
    }

    _renderLabels(){
      try{
        const list = document.getElementById('eventList'); if (!list) return;
        const items = Array.from(list.querySelectorAll('.event-item'));
        items.forEach(item=>{
          const id = item.getAttribute('data-event-id');
          const ev = (window.detectedEvents||[]).find(e=> e.id===id);
          // Remove old badge
          const old = item.querySelector('.cluster-badge'); if (old) old.remove();
          if (!ev || typeof ev.cluster!=='number') return;
          const color = this.palette[ev.cluster % this.palette.length];
          const badge = document.createElement('span');
          badge.className = 'cluster-badge';
          badge.style.cssText = 'margin-left:8px;padding:2px 6px;border-radius:10px;font-size:11px;background:'+color+'20;color:'+color+';border:1px solid '+color+'55;';
          badge.textContent = `C${ev.cluster+1}`;
          // Insert after magnitude
          const mag = item.querySelector('.event-magnitude');
          if (mag && mag.nextSibling) mag.after(badge); else item.appendChild(badge);
        });
      }catch{}
    }
  }

  document.addEventListener('DOMContentLoaded', ()=>{ window.eventClustering = new EventClustering(); });
})();
