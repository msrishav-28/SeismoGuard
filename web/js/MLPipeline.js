/* eslint-disable */
// MLPipeline.js - TF.js async-loaded models with graceful fallback
/**
 * Frontend ML utilities backed by TensorFlow.js with async loading and UI.
 * @class MLPipeline
 */
class MLPipeline {
    constructor(){ this.models = {}; this.isReady = false; this.modelUrls = { detector:'/models/detector.json', classifier:'/models/classifier.json', predictor:'/models/predictor.json' }; this.init(); }
    async init(){ await this.loadModels(); this.createUI(); }
    async loadModels(){ try { await this.loadTensorFlow(); for (const [name, url] of Object.entries(this.modelUrls)){
                try { this.models[name] = await tf.loadLayersModel(url); console.log(`Loaded model: ${name}`); }
                catch { console.warn(`Could not load ${name} model, creating new`); this.models[name] = this.createModel(name); }
            } this.isReady = true; } catch (e) { console.error('Failed to initialize ML pipeline:', e); } }
    async loadTensorFlow(){ return new Promise((res, rej)=> { const s = document.createElement('script'); s.src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest'; s.onload=res; s.onerror=rej; document.head.appendChild(s); }); }
    createModel(type){ const model = tf.sequential(); switch(type){
        case 'detector': model.add(tf.layers.dense({ units:64, activation:'relu', inputShape:[100] })); model.add(tf.layers.dropout({ rate:0.3 })); model.add(tf.layers.dense({ units:32, activation:'relu' })); model.add(tf.layers.dropout({ rate:0.3 })); model.add(tf.layers.dense({ units:1, activation:'sigmoid' })); model.compile({ optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy'] }); break;
        case 'classifier': model.add(tf.layers.conv1d({ filters:32, kernelSize:3, activation:'relu', inputShape:[1000,1] })); model.add(tf.layers.maxPooling1d({ poolSize:2 })); model.add(tf.layers.conv1d({ filters:64, kernelSize:3, activation:'relu' })); model.add(tf.layers.globalMaxPooling1d()); model.add(tf.layers.dense({ units:128, activation:'relu' })); model.add(tf.layers.dropout({ rate:0.5 })); model.add(tf.layers.dense({ units:4, activation:'softmax' })); model.compile({ optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy'] }); break;
        case 'predictor': model.add(tf.layers.lstm({ units:50, returnSequences:true, inputShape:[100,1] })); model.add(tf.layers.lstm({ units:50, returnSequences:false })); model.add(tf.layers.dense({ units:25, activation:'relu' })); model.add(tf.layers.dense({ units:1 })); model.compile({ optimizer:'adam', loss:'meanSquaredError' }); break; }
        return model; }
    async detectEvents(data){ if (!this.isReady || !this.models.detector) return null; const input = tf.tensor2d(data, [1, data.length]); const pred = await this.models.detector.predict(input).data(); input.dispose(); return pred[0] > 0.5; }
    async classifyEvent(waveform){ if (!this.isReady || !this.models.classifier) return null; const input = tf.tensor3d(waveform, [1, waveform.length, 1]); const pred = await this.models.classifier.predict(input).data(); input.dispose(); const classes=['moonquake','marsquake','impact','artificial']; const idx = pred.indexOf(Math.max(...pred)); return { type: classes[idx], confidence: pred[idx], probabilities: Object.fromEntries(classes.map((c,i)=> [c, pred[i]])) };
    }
    async predictNextValues(ts, steps=10){ if (!this.isReady || !this.models.predictor) return null; const out=[]; let cur = ts.slice(-100); for (let i=0;i<steps;i++){ const input = tf.tensor3d(cur,[1,100,1]); const pred = await this.models.predictor.predict(input).data(); out.push(pred[0]); cur=[...cur.slice(1), pred[0]]; input.dispose(); } return out; }
    async trainModel(name, trainingData, labels, options={}){ const model = this.models[name]; if (!model) return null; const defaults = { epochs:50, batchSize:32, validationSplit:0.2, callbacks:{ onEpochEnd:(epoch, logs)=> this.updateTrainingProgress(epoch, logs) } }; const opts = { ...defaults, ...options }; const xs = tf.tensor2d(trainingData); const ys = tf.tensor2d(labels); const history = await model.fit(xs, ys, opts); xs.dispose(); ys.dispose(); return history; }
    createUI(){ const html = `
        <div class="ml-panel" id="mlPanel" style="display:none;">
            <div class="ml-header"><h3>🤖 Machine Learning Pipeline</h3><button class="close-btn" onclick="mlPipeline.closePanel()">×</button></div>
            <div class="ml-tabs"><button class="ml-tab active" data-tab="inference">Inference</button><button class="ml-tab" data-tab="training">Training</button><button class="ml-tab" data-tab="analysis">Analysis</button></div>
            <div class="ml-content">
                <div class="ml-tab-content active" id="inference-tab">
                    <div class="model-status"><h4>Model Status</h4><div class="status-grid">
                        <div class="model-item"><span class="model-name">Detector</span><span class="model-status-indicator ${this.models.detector?'ready':'loading'}"></span></div>
                        <div class="model-item"><span class="model-name">Classifier</span><span class="model-status-indicator ${this.models.classifier?'ready':'loading'}"></span></div>
                        <div class="model-item"><span class="model-name">Predictor</span><span class="model-status-indicator ${this.models.predictor?'ready':'loading'}"></span></div>
                    </div></div>
                    <div class="inference-controls"><button class="btn" onclick="mlPipeline.runInference()">Run Inference</button><button class="btn" onclick="mlPipeline.runBatchInference && mlPipeline.runBatchInference()">Batch Process</button></div>
                    <div class="inference-results" id="inferenceResults"></div>
                </div>
                <div class="ml-tab-content" id="training-tab">
                    <div class="training-controls">
                        <label>Select Model:</label><select id="trainingModel"><option value="detector">Event Detector</option><option value="classifier">Event Classifier</option><option value="predictor">Time Series Predictor</option></select>
                        <label>Training Data:</label><input type="file" id="trainingData" accept=".csv,.json">
                        <label>Epochs:</label><input type="number" id="epochs" value="50" min="1" max="1000">
                        <label>Batch Size:</label><input type="number" id="batchSize" value="32" min="1" max="256">
                        <button class="btn" onclick="mlPipeline.startTraining && mlPipeline.startTraining()">Start Training</button>
                    </div>
                    <div class="training-progress" id="trainingProgress" style="display:none;">
                        <h4>Training Progress</h4>
                        <div class="progress-bar"><div class="progress-fill" id="trainingProgressBar"></div></div>
                        <div class="training-metrics"><span>Epoch: <span id="currentEpoch">0</span></span><span>Loss: <span id="trainingLoss">--</span></span><span>Accuracy: <span id="trainingAccuracy">--</span></span></div>
                        <canvas id="trainingChart" width="400" height="200"></canvas>
                    </div>
                </div>
                <div class="ml-tab-content" id="analysis-tab">
                    <div class="analysis-tools"><h4>Model Analysis</h4><button class="btn" onclick="mlPipeline.analyzeFeatures && mlPipeline.analyzeFeatures()">Feature Importance</button><button class="btn" onclick="mlPipeline.generateConfusionMatrix && mlPipeline.generateConfusionMatrix()">Confusion Matrix</button><button class="btn" onclick="mlPipeline.exportModel()">Export Model</button></div>
                    <div class="analysis-results" id="analysisResults"></div>
                </div>
            </div>
        </div>`;
        const style = document.createElement('style'); style.textContent = `
            .ml-panel { position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%); width: 600px; max-height: 80vh; background: var(--card-bg, linear-gradient(135deg,#1e1e3f,#2a2a4e)); border-radius: 15px; padding: 20px; z-index: 10000; overflow-y: auto; }
            .ml-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; }
            .ml-tabs { display:flex; gap:10px; margin-bottom:20px; }
            .ml-tab { flex:1; padding:10px; background: rgba(0,0,0,0.3); border:1px solid var(--accent-color,#00ffff); color: var(--text-color,#fff); cursor:pointer; border-radius:5px; transition: all 0.3s ease; }
            .ml-tab.active { background: var(--accent-color,#00ffff); color:#000; }
            .ml-tab-content { display:none; }
            .ml-tab-content.active { display:block; }
            .model-status-indicator { width:10px; height:10px; border-radius:50%; display:inline-block; }
            .model-status-indicator.ready { background:#00ff00; }
            .model-status-indicator.loading { background:#ffff00; animation:pulse 1s infinite; }
            @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }
            .status-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
            .model-item { display:flex; flex-direction:column; align-items:center; gap:10px; padding:15px; background: rgba(0,0,0,0.3); border-radius:8px; }
            .training-controls { display:flex; flex-direction:column; gap:15px; }
        `; document.head.appendChild(style); document.body.insertAdjacentHTML('beforeend', html); this.attachMLListeners(); }
    attachMLListeners(){ document.querySelectorAll('.ml-tab').forEach(tab=> tab.addEventListener('click', ()=> { document.querySelectorAll('.ml-tab').forEach(t=> t.classList.remove('active')); document.querySelectorAll('.ml-tab-content').forEach(c=> c.classList.remove('active')); tab.classList.add('active'); document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active'); })); }
    openPanel(){ document.getElementById('mlPanel').style.display = 'block'; }
    closePanel(){ document.getElementById('mlPanel').style.display = 'none'; }
    async runInference(){ if (!window.currentProcessedData){ alert('No data available for inference'); return; } const results = document.getElementById('inferenceResults'); results.innerHTML = '<p>Running inference...</p>'; const isEvent = await this.detectEvents(window.currentProcessedData); let html = `<div class="inference-result"><h4>Detection Result</h4><p>Event Detected: ${isEvent ? 'Yes' : 'No'}</p>`; if (isEvent){ const classification = await this.classifyEvent(window.currentProcessedData); html += `<h4>Classification</h4><p>Type: ${classification.type}</p><p>Confidence: ${(classification.confidence*100).toFixed(1)}%</p>`; html += `<h4>Probabilities</h4>`; for (const [type,prob] of Object.entries(classification.probabilities)){ html += `<div class="prob-bar"><span>${type}: ${(prob*100).toFixed(1)}%</span><div class="bar" style="width:${prob*100}%"></div></div>`; } } html += `</div>`; results.innerHTML = html; }
    updateTrainingProgress(epoch, logs){ document.getElementById('currentEpoch').textContent = epoch + 1; document.getElementById('trainingLoss').textContent = logs.loss?.toFixed(4) ?? '--'; if (logs.acc !== undefined) document.getElementById('trainingAccuracy').textContent = (logs.acc*100).toFixed(1)+'%'; const total = parseInt(document.getElementById('epochs').value, 10) || 1; document.getElementById('trainingProgressBar').style.width = (((epoch+1)/total)*100)+'%'; }
    async exportModel(modelName='detector'){ const model = this.models[modelName]; if (!model) return; await model.save(`downloads://${modelName}_model`); alert(`Model ${modelName} exported successfully!`); }
}

document.addEventListener('DOMContentLoaded', () => { window.mlPipeline = new MLPipeline(); });
