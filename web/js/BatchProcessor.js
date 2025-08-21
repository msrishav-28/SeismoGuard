/* eslint-disable */
// BatchProcessor.js - CSV-only client-side batch + WS processing
class BatchProcessor {
    constructor(){ this.queue=[]; this.running=false; this.init(); }
    init(){ this.injectUI(); this.attach(); }
    injectUI(){ const html=`
        <div id="batchPanel" style="position:fixed;left:20px;bottom:80px;background:var(--card-bg);padding:12px;border-radius:8px;z-index:999">
            <h4 style="margin:0 0 6px;color:var(--text-color,#fff)">Batch Processing</h4>
            <input type="file" id="batchFiles" accept=".csv" multiple>
            <div style="margin-top:8px"><button class="btn" id="batchStartBtn">Start</button>
            <button class="btn btn-secondary" id="batchClearBtn">Clear</button></div>
            <div id="batchStatus" style="margin-top:8px;color:var(--text-color,#fff)">Idle</div>
        </div>`; document.body.insertAdjacentHTML('beforeend', html); }
    attach(){ document.getElementById('batchStartBtn').addEventListener('click',()=> this.start()); document.getElementById('batchClearBtn').addEventListener('click',()=>{this.queue=[]; this.updateStatus('Cleared');}); document.getElementById('batchFiles').addEventListener('change',(e)=> this.enqueueFiles(e.target.files)); }
    enqueueFiles(files){ for(const f of files){ this.queue.push(f); } this.updateStatus(`${this.queue.length} file(s) queued`); }
    updateStatus(t){ const el=document.getElementById('batchStatus'); if (el) el.textContent=t; }
    async start(){ if (this.running) return; this.running=true; this.updateStatus('Running...'); for (const file of this.queue){ try{ const samples = await this.readCsv(file); await this.process(samples); this.updateStatus(`Processed: ${file.name}`);} catch(e){ this.updateStatus(`Failed: ${file.name}`);} } this.running=false; this.updateStatus('Done'); }
    readCsv(file){ return new Promise((resolve,reject)=>{ const reader = new FileReader(); reader.onload = ()=>{ try{ const lines = String(reader.result).trim().split(/\r?\n/); const nums = lines.map(l=> parseFloat(l.split(',')[0])).filter(n=> Number.isFinite(n)); resolve(nums); }catch(e){ reject(e);} }; reader.onerror = reject; reader.readAsText(file); }); }
    async process(samples){ return new Promise((resolve)=>{ try{ if (window.wsClient){ window.wsClient.subscribe('processing_result', ()=> resolve()); window.wsClient.send({ type:'process', data: samples.slice(0,200000) }); } else { setTimeout(resolve, 300); } } catch(e){ resolve(); } }); }
}

document.addEventListener('DOMContentLoaded', ()=> { window.batchProcessor = new BatchProcessor(); });
