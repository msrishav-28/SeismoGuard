/* eslint-disable */
// ExportManager.js - Multi-format export with preview
/**
 * Handles export modal and multi-format data generation with optional compression.
 * @class ExportManager
 */
class ExportManager {
    constructor() {
        this.formats = ['CSV', 'JSON', 'XML', 'HDF5', 'MATLAB', 'Excel'];
        this.exportHistory = [];
        this.init();
    }
    init() { this.createUI(); this.attachListeners(); }
    createUI() {
        const html = `
        <div class="export-modal" id="exportModal" style="display:none;">
            <div class="export-content">
                <h3>📤 Export Data</h3>
                <div class="export-options">
                    <div class="format-selection"><label>Export Format:</label>
                        <select id="exportFormat">${this.formats.map(f=>`<option value="${f}">${f}</option>`).join('')}</select>
                    </div>
                    <div class="data-selection"><label>Data to Export:</label>
                        <div class="checkbox-group">
                            <label><input type="checkbox" name="exportData" value="raw" checked> Raw Data</label>
                            <label><input type="checkbox" name="exportData" value="processed" checked> Processed Data</label>
                            <label><input type="checkbox" name="exportData" value="events" checked> Events</label>
                            <label><input type="checkbox" name="exportData" value="statistics"> Statistics</label>
                            <label><input type="checkbox" name="exportData" value="metadata"> Metadata</label>
                        </div>
                    </div>
                    <div class="compression-options">
                        <label><input type="checkbox" id="compressExport"> Compress Output</label>
                        <label><input type="checkbox" id="includeVisuals"> Include Visualizations</label>
                    </div>
                    <div class="export-preview"><label>Preview:</label>
                        <pre id="exportPreview">Select options to see preview...</pre>
                    </div>
                    <div class="export-buttons">
                        <button class="btn" id="doExportBtn">Export</button>
                        <button class="btn btn-secondary" id="cancelExportBtn">Cancel</button>
                    </div>
                </div>
            </div>
        </div>`;
        const style = document.createElement('style');
        style.textContent = `
            .export-modal { position: fixed; inset: 0; background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center; z-index: 10000; }
            .export-content { background: var(--card-bg, linear-gradient(135deg, #1e1e3f, #2a2a4e)); padding: 30px; border-radius: 15px; width: 500px; max-height: 80vh; overflow-y: auto; }
            .export-options > div { margin: 20px 0; }
            .checkbox-group { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
            .export-preview { background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; max-height: 200px; overflow-y: auto; }
            .export-preview pre { color: var(--text-color,#fff); font-family: 'Courier New', monospace; font-size: 12px; margin: 0; }
        `;
        document.head.appendChild(style);
        document.body.insertAdjacentHTML('beforeend', html);
    }
    attachListeners() {
        document.querySelectorAll('#exportModal input, #exportModal select').forEach(el => el.addEventListener('change', ()=> this.updatePreview()));
        document.addEventListener('click', (e) => {
            if (e.target && e.target.id === 'doExportBtn') this.doExport();
            if (e.target && e.target.id === 'cancelExportBtn') this.closeModal();
        });
    }
    openModal() { document.getElementById('exportModal').style.display = 'flex'; this.updatePreview(); }
    closeModal() { document.getElementById('exportModal').style.display = 'none'; }
    updatePreview() { const f = document.getElementById('exportFormat').value; document.getElementById('exportPreview').textContent = this.generatePreview(f); }
    generatePreview(format) {
        const sample = { timestamp: new Date().toISOString(), events: [{ time: 'T+100s', magnitude: 2.3, type: 'moonquake' }], statistics: { mean: 1.23e-9, std: 4.56e-10 } };
        switch(format){
            case 'JSON': return JSON.stringify(sample,null,2);
            case 'CSV': return 'timestamp,event_time,magnitude,type\n' + `${sample.timestamp},T+100s,2.3,moonquake`;
            case 'XML': return `<?xml version="1.0" encoding="UTF-8"?>\n<seismic_data>\n  <timestamp>${sample.timestamp}</timestamp>\n  <events>\n    <event>\n      <time>T+100s</time>\n      <magnitude>2.3</magnitude>\n      <type>moonquake</type>\n    </event>\n  </events>\n</seismic_data>`;
            case 'MATLAB': return `% SeismoGuard Export\n% Generated: ${sample.timestamp}\nevents = struct('time','T+100s','magnitude',2.3,'type','moonquake');`;
            default: return 'Binary format - preview not available';
        }
    }
    async doExport() {
    const format = document.getElementById('exportFormat').value;
        const compress = document.getElementById('compressExport').checked;
        const includeVisuals = document.getElementById('includeVisuals').checked;
        const selectedData = {}; document.querySelectorAll('input[name="exportData"]:checked').forEach(i=> selectedData[i.value]=true);
        const exportData = await this.prepareExportData(selectedData, includeVisuals);
    let { content, mime } = this.formatData(exportData, format);
    if (compress) ({ content, mime } = await this.compressData(content, mime));
    this.downloadFile(content, `seismic_export_${Date.now()}.${this.getExtension(format, compress)}`, mime);
    const size = (content instanceof Blob) ? content.size : (typeof content === 'string' ? content.length : 0);
    this.exportHistory.push({ timestamp: new Date(), format, compressed: compress, size });
        this.closeModal(); if (window.audioEngine) window.audioEngine.play('success');
    }
    async prepareExportData(selections, includeVisuals){
        const data = {};
        if (selections.raw && window.currentRawData) data.raw = window.currentRawData;
        if (selections.processed && window.currentProcessedData) data.processed = window.currentProcessedData;
        if (selections.events && window.detectedEvents) data.events = window.detectedEvents;
        if (selections.statistics && window.statisticsPanel) data.statistics = window.statisticsPanel.stats;
        if (selections.metadata) data.metadata = { exportTime: new Date().toISOString(), planet: window.currentPlanet || 'moon', samplingRate: 100, version: '1.0.0' };
        if (includeVisuals) data.visualizations = await this.captureVisualizations();
        return data;
    }
    formatData(data, format){
        switch(format){
            case 'JSON': return { content: JSON.stringify(data,null,2), mime: 'application/json' };
            case 'CSV': return { content: this.toCSV(data), mime: 'text/csv' };
            case 'XML': return { content: this.toXML(data), mime: 'application/xml' };
            case 'MATLAB': return { content: this.toMATLAB(data), mime: 'text/plain' };
            case 'Excel': return this.toExcel(data);
            case 'HDF5': return this.toHDF5(data);
            default: return { content: JSON.stringify(data), mime: 'application/json' };
        }
    }
    toCSV(data){ let csv=''; if (data.events && data.events.length){ const headers = Object.keys(data.events[0]); csv += headers.join(',') + '\n'; data.events.forEach(e=> { csv += headers.map(h=> e[h]).join(',') + '\n'; }); } return csv; }
    toXML(data){ const toXMLString = (obj, root='root') => { let xml = `<${root}>`; for (const [k,v] of Object.entries(obj)){ if (typeof v === 'object' && v !== null){ if (Array.isArray(v)) v.forEach(it=> xml += toXMLString(it,k)); else xml += toXMLString(v,k); } else { xml += `<${k}>${v}</${k}>`; } } xml += `</${root}>`; return xml; }; return '<?xml version="1.0" encoding="UTF-8"?>\n' + toXMLString(data,'seismic_data'); }
    toMATLAB(data){ return `% SeismoGuard Export\n% ${new Date().toISOString()}\n% This is a placeholder.`; }
    toExcel(data){
        try {
            if (!window.XLSX) throw new Error('XLSX not available');
            const wb = XLSX.utils.book_new();
            if (data.events) {
                const ws = XLSX.utils.json_to_sheet(data.events);
                XLSX.utils.book_append_sheet(wb, ws, 'Events');
            }
            if (data.statistics) {
                const flat = this.flattenObject(data.statistics);
                const ws = XLSX.utils.json_to_sheet([flat]);
                XLSX.utils.book_append_sheet(wb, ws, 'Statistics');
            }
            const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
            return { content: new Blob([wbout], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }), mime: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' };
        } catch (e) {
            return { content: 'Excel export not available in this environment', mime: 'text/plain' };
        }
    }
    toHDF5(){
        // Not feasible purely in-browser without heavy libs; provide placeholder
        return { content: 'HDF5 export is not supported in the browser build.', mime: 'text/plain' };
    }
    async compressData(content, mime){
        try {
            if (typeof content === 'string'){
                const enc = new TextEncoder().encode(content);
                const gz = window.pako ? window.pako.gzip(enc) : enc; // fallback: no compression
                return { content: new Blob([gz], { type: mime }), mime };
            }
            if (content instanceof Blob){
                const buf = await content.arrayBuffer();
                const gz = window.pako ? window.pako.gzip(new Uint8Array(buf)) : new Uint8Array(buf);
                return { content: new Blob([gz], { type: content.type || mime }), mime: content.type || mime };
            }
            return { content, mime };
        } catch (e) { return { content, mime }; }
    }
    async captureVisualizations(){ const canvases = ['waveformCanvas','spectrogramCanvas','performanceChart'].map(id=> document.getElementById(id)); return canvases.filter(Boolean).map(c=> c.toDataURL()); }
    downloadFile(content, filename, mime){
        const blob = content instanceof Blob ? content : new Blob([content], { type: mime || 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url);
    }
    flattenObject(obj, prefix=''){
        const out = {};
        for (const [k,v] of Object.entries(obj||{})){
            const key = prefix ? `${prefix}.${k}` : k;
            if (v && typeof v === 'object' && !Array.isArray(v)) Object.assign(out, this.flattenObject(v, key));
            else out[key] = v;
        }
        return out;
    }
    getExtension(format, compressed){ const map = { JSON:'json', CSV:'csv', XML:'xml', MATLAB:'m', Excel:'xlsx', HDF5:'h5' }; let ext = map[format] || 'dat'; if (compressed) ext += '.gz'; return ext; }
}

document.addEventListener('DOMContentLoaded', () => { window.exportManager = new ExportManager(); });
