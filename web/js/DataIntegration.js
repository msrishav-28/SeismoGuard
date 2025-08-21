/* eslint-disable */
// DataIntegration.js — Frontend panel to fetch priority datasets via backend proxy
(function(){
  class DataIntegrationPanel {
    constructor(){ this._ui(); this.resultsEl=null; }
    _ui(){
      const html = `
      <div id="dataIntegrationPanel" style="position:fixed;left:20px;bottom:20px;background:var(--card-bg);border:1px solid var(--card-border);padding:10px 12px;border-radius:8px;z-index:999;max-width:420px">
        <div style="font-weight:600;margin-bottom:6px;color:var(--text-color,#fff)">Data Integration</div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px">
          <button class="btn" id="btnUSGS" style="padding:4px 8px">USGS: Past Day M4.5+</button>
          <button class="btn" id="btnIRIS" style="padding:4px 8px">IRIS: Event (24h)</button>
          <button class="btn" id="btnEMSC" style="padding:4px 8px">EMSC: Event (24h)</button>
        </div>
        <div id="dataResults" style="max-height:180px;overflow:auto;font-size:12px;color:#ccc"></div>
      </div>`;
      document.addEventListener('DOMContentLoaded', ()=>{
        document.body.insertAdjacentHTML('beforeend', html);
        this.resultsEl = document.getElementById('dataResults');
        const safeSend = (payload)=>{ if (window.wsClient) window.wsClient.send(payload); };
        const pretty = (obj)=> '<pre style="white-space:pre-wrap">'+(typeof obj==='string'?obj:JSON.stringify(obj,null,2))+'</pre>';
        document.getElementById('btnUSGS').addEventListener('click', ()=>{
          this.resultsEl.innerHTML = 'Fetching USGS...';
          safeSend({ type:'data_fetch', provider:'usgs_realtime', endpoint:'4.5_day.geojson' });
        });
        document.getElementById('btnIRIS').addEventListener('click', ()=>{
          this.resultsEl.innerHTML = 'Fetching IRIS...';
          safeSend({ type:'data_fetch', provider:'iris_event', params:{} });
        });
        document.getElementById('btnEMSC').addEventListener('click', ()=>{
          this.resultsEl.innerHTML = 'Fetching EMSC...';
          safeSend({ type:'data_fetch', provider:'emsc_event', params:{} });
        });
        if (window.wsClient){
          window.wsClient.subscribe('data_fetch_result', (res)=>{
            this.resultsEl.innerHTML = pretty(res?.data || res?.error || 'No data');
          });
        }
      });
    }
  }
  document.addEventListener('DOMContentLoaded', ()=>{ window.dataIntegration = new DataIntegrationPanel(); });
})();
