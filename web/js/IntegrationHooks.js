/* eslint-disable */
// IntegrationHooks.js - Hook into existing functions without modifying them
(function(){
    // Preserve originals
    const _runDetection = window.runDetection;
    const _exportResults = window.exportResults;
    const _processFiles = window.processFiles;

    if (typeof _runDetection === 'function') {
        window.runDetection = function(...args){
            if (window.audioEngine) window.audioEngine.play('notification');
            try { return _runDetection.apply(this, args); }
            finally {
                setTimeout(()=> {
                    if (window.statisticsPanel){
                        window.statisticsPanel.stats.events.total = (window.statisticsPanel.stats.events.total || 0) + (window.detectedEvents?.length || 0);
                        window.statisticsPanel.updateUI();
                    }
                    if (window.mlPipeline && window.currentProcessedData){ window.mlPipeline.runInference(); }
                    if (window.wsClient){ window.wsClient.send({ type: 'events_detected', count: window.detectedEvents?.length || 0 }); }
                    if (window.audioEngine) window.audioEngine.play('success');
                }, 0);
            }
        }
    }

    if (typeof _exportResults === 'function') {
        window.exportResults = function(...args){
            if (window.exportManager && typeof window.exportManager.openModal === 'function'){
                window.exportManager.openModal();
                return;
            }
            return _exportResults.apply(this, args);
        }
    }

    if (typeof _processFiles === 'function') {
        window.processFiles = function(files, ...rest){
            const t0 = performance.now();
            const res = _processFiles.apply(this, [files, ...rest]);
            const t1 = performance.now();
            if (window.statisticsPanel){
                window.statisticsPanel.stats.performance.processingTime = (t1 - t0);
                window.statisticsPanel.updateUI();
            }
            if (window.audioEngine) window.audioEngine.onProcessingComplete();
            return res;
        }
    }
})();
