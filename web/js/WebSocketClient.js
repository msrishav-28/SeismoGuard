/* eslint-disable */
// WebSocketClient.js - Robust client with reconnection and buffering
/**
 * Real-time WebSocket client with reconnection, buffering, and pub/sub.
 * @class WebSocketClient
 */
class WebSocketClient {
    constructor() {
    this.ws = null; this.url = 'ws://127.0.0.1:8765/ws';
        this.reconnectInterval = 5000; this.reconnectAttempts = 0; this.maxReconnectAttempts = 10;
        this.messageQueue = []; this.subscribers = new Map(); this.isConnected = false;
        this.dataBuffer = []; this.bufferSize = 1000; this.init();
    }
    init(){ this.connect(); this.createStatusIndicator(); }
    connect(){
        try { this.ws = new WebSocket(this.url);
            this.ws.onopen = () => this.onOpen();
            this.ws.onmessage = (e) => this.onMessage(e);
            this.ws.onerror = (err) => this.onError(err);
            this.ws.onclose = () => this.onClose();
        } catch (err) { console.error('WebSocket connection failed:', err); this.scheduleReconnect(); }
    }
    onOpen(){ console.log('WebSocket connected'); this.isConnected = true; this.reconnectAttempts = 0; this.updateStatusIndicator('connected');
        while (this.messageQueue.length) this.send(this.messageQueue.shift());
        this.send({ type: 'subscribe', channels: ['seismic_data','events','alerts'] }); this.emit('connection', { status: 'connected' }); }
    onMessage(event){ try { const data = JSON.parse(event.data); switch(data.type){
        case 'seismic_data': this.handleSeismicData(data.payload); break;
        case 'event_detected': this.handleEventDetection(data.payload); break;
        case 'alert': this.handleAlert(data.payload); break;
        case 'statistics': this.handleStatistics(data.payload); break;
        case 'heartbeat': this.handleHeartbeat(); break;
        default: this.emit(data.type, data.payload); } } catch(err){ console.error('Error processing message:', err); } }
    onError(error){ console.error('WebSocket error:', error); this.updateStatusIndicator('error'); this.emit('error', error); }
    onClose(){ console.log('WebSocket disconnected'); this.isConnected = false; this.updateStatusIndicator('disconnected'); this.emit('connection', { status: 'disconnected' }); this.scheduleReconnect(); }
    scheduleReconnect(){ if (this.reconnectAttempts < this.maxReconnectAttempts){ this.reconnectAttempts++; setTimeout(()=> this.connect(), this.reconnectInterval); } else { console.error('Max reconnection attempts reached'); this.updateStatusIndicator('failed'); } }
    send(data){ if (this.isConnected && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(data)); else this.messageQueue.push(data); }
    handleSeismicData(payload){ this.dataBuffer.push(...payload.samples); if (this.dataBuffer.length > this.bufferSize) this.dataBuffer = this.dataBuffer.slice(-this.bufferSize);
        if (window.updateWaveform) window.updateWaveform(this.dataBuffer);
        this.emit('seismic_data', this.dataBuffer); }
    handleEventDetection(event){ if (window.audioEngine) window.audioEngine.onEventDetected(event.magnitude); if (window.addEventToList) window.addEventToList(event); this.showNotification('Event Detected', `Magnitude ${event.magnitude} ${event.type}`); this.emit('event_detected', event); }
    handleAlert(alert){ if (window.audioEngine) window.audioEngine.play('warning'); this.showNotification('Alert', alert.message, 'warning'); this.emit('alert', alert); }
    handleStatistics(stats){ if (window.statisticsPanel) window.statisticsPanel.updateFromServer(stats); this.emit('statistics', stats); }
    handleHeartbeat(){ this.send({ type: 'heartbeat_ack' }); }
    subscribe(event, cb){ if (!this.subscribers.has(event)) this.subscribers.set(event, []); this.subscribers.get(event).push(cb); return () => { const arr = this.subscribers.get(event); const i = arr.indexOf(cb); if (i>-1) arr.splice(i,1); }; }
    emit(event, data){ const arr = this.subscribers.get(event); if (arr) arr.forEach(cb => { try { cb(data); } catch(err){ console.error(`Error in event handler for ${event}:`, err); } }); }
    createStatusIndicator(){ const html = `<div class="websocket-status" id="wsStatus"><div class="status-dot" id="wsStatusDot"></div><span class="status-text" id="wsStatusText">Connecting...</span></div>`;
        const style = document.createElement('style'); style.textContent = `
            .websocket-status { position: fixed; bottom: 20px; left: 20px; background: var(--card-bg, rgba(255,255,255,0.1)); padding: 10px 15px; border-radius: 20px; display: flex; align-items: center; gap: 10px; backdrop-filter: blur(10px); z-index: 1000; }
            .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #ffff00; animation: pulse 2s infinite; }
            .status-dot.connected { background: #00ff00; animation: none; }
            .status-dot.disconnected { background: #ff0000; animation: pulse 1s infinite; }
            .status-dot.error { background: #ff6600; animation: pulse 0.5s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
            .status-text { color: var(--text-color,#fff); font-size: 12px; }
        `; document.head.appendChild(style); document.body.insertAdjacentHTML('beforeend', html); }
    updateStatusIndicator(status){ const dot = document.getElementById('wsStatusDot'); const text = document.getElementById('wsStatusText'); if (!dot||!text) return; dot.className = `status-dot ${status}`; const messages = { connected:'Connected', disconnected:'Disconnected', error:'Connection Error', failed:'Connection Failed', connecting:'Connecting...' }; text.textContent = messages[status] || 'Unknown'; }
    showNotification(title, message, type='info'){ const el = document.createElement('div'); el.className = `notification notification-${type}`; el.innerHTML = `<strong>${title}</strong><p>${message}</p>`; document.body.appendChild(el); setTimeout(()=> el.classList.add('show'), 10); setTimeout(()=> { el.classList.remove('show'); setTimeout(()=> el.remove(), 300); }, 5000); }
    startStreaming(channel='default'){ this.send({ type: 'start_stream', channel }); }
    stopStreaming(channel='default'){ this.send({ type: 'stop_stream', channel }); }
    requestHistoricalData(startTime, endTime){ this.send({ type: 'historical_data', start: startTime, end: endTime }); }
    updateSettings(settings){ this.send({ type: 'update_settings', settings }); }
}

document.addEventListener('DOMContentLoaded', () => { window.wsClient = new WebSocketClient(); });
