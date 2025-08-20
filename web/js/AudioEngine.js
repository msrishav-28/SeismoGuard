/* eslint-disable */
// AudioEngine.js - Efficient sound alerts using Web Audio API
/**
 * Sound alert manager using Web Audio API with synthetic tones.
 * @class AudioEngine
 */
class AudioEngine {
    constructor() {
        this.audioContext = null;
        this.sounds = {};
        this.enabled = this.loadPreference() !== false;
        this.volume = parseFloat(localStorage.getItem('seismoguard-volume')) || 0.5;
        this.init();
    }
    init() {
        document.addEventListener('click', () => {
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                this.generateSounds();
            }
        }, { once: true });
        this.createUI();
    }
    generateSounds() {
        this.sounds = {
            eventDetected: this.createBeep(440, 0.2, 'sine'),
            highMagnitude: this.createBeep(880, 0.3, 'square'),
            warning: this.createWarning(),
            success: this.createSuccess(),
            notification: this.createNotification()
        };
    }
    createBeep(frequency, duration, type = 'sine') {
        return () => {
            if (!this.enabled || !this.audioContext) return;
            const oscillator = this.audioContext.createOscillator();
            const gainNode = this.audioContext.createGain();
            oscillator.connect(gainNode); gainNode.connect(this.audioContext.destination);
            oscillator.frequency.value = frequency; oscillator.type = type;
            gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
            gainNode.gain.linearRampToValueAtTime(this.volume, this.audioContext.currentTime + 0.01);
            gainNode.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + duration);
            oscillator.start(); oscillator.stop(this.audioContext.currentTime + duration);
        };
    }
    createWarning() { return () => { if (!this.enabled || !this.audioContext) return; this.createBeep(600, 0.15, 'square')(); setTimeout(() => this.createBeep(400, 0.15, 'square')(), 150); }; }
    createSuccess() { return () => { if (!this.enabled || !this.audioContext) return; [261.63, 329.63, 392.00, 523.25].forEach((f,i)=> setTimeout(()=> this.createBeep(f,0.1,'sine')(), i*50)); }; }
    createNotification() { return () => { if (!this.enabled || !this.audioContext) return; this.createBeep(659.25, 0.1, 'sine')(); setTimeout(()=> this.createBeep(783.99, 0.15, 'sine')(), 100); }; }
    createUI() {
        const html = `
        <div class="audio-controls" id="audioControls">
            <button class="audio-toggle" id="audioToggle" title="Toggle Sound"><span class="audio-icon">${this.enabled ? '🔊' : '🔇'}</span></button>
            <input type="range" class="volume-slider" id="volumeSlider" min="0" max="100" value="${this.volume*100}" title="Volume" style="display: none;">
        </div>`;
        const style = document.createElement('style');
        style.textContent = `
            .audio-controls { position: fixed; top: 20px; right: 200px; z-index: 9999; display: flex; align-items: center; gap: 10px; }
            .audio-toggle { width: 40px; height: 40px; border: none; background: var(--card-bg, rgba(255,255,255,0.1)); border-radius: 50%; cursor: pointer; font-size: 20px; transition: all 0.3s ease; backdrop-filter: blur(10px); }
            .audio-toggle:hover { transform: scale(1.1); background: var(--accent-color, #00ffff); }
            .volume-slider { width: 100px; height: 4px; appearance: none; background: var(--card-bg, rgba(255,255,255,0.2)); border-radius: 2px; outline: none; }
            .volume-slider::-webkit-slider-thumb { appearance: none; width: 12px; height: 12px; background: var(--accent-color, #00ffff); border-radius: 50%; cursor: pointer; }
        `;
        document.head.appendChild(style);
        document.body.insertAdjacentHTML('beforeend', html);
        this.attachListeners();
    }
    attachListeners() {
        const toggle = document.getElementById('audioToggle');
        const slider = document.getElementById('volumeSlider');
        toggle.addEventListener('click', () => {
            this.enabled = !this.enabled;
            toggle.querySelector('.audio-icon').textContent = this.enabled ? '🔊' : '🔇';
            slider.style.display = this.enabled ? 'block' : 'none';
            localStorage.setItem('seismoguard-audio', this.enabled);
            if (this.enabled) this.play('notification');
        });
        toggle.addEventListener('mouseenter', () => { if (this.enabled) slider.style.display = 'block'; });
        document.getElementById('audioControls').addEventListener('mouseleave', () => { setTimeout(()=> slider.style.display='none', 1000); });
        slider.addEventListener('input', (e) => { this.volume = e.target.value / 100; localStorage.setItem('seismoguard-volume', this.volume); });
    }
    play(soundName) { if (this.sounds[soundName]) this.sounds[soundName](); }
    loadPreference() { const saved = localStorage.getItem('seismoguard-audio'); return saved === null ? true : saved === 'true'; }
    onEventDetected(magnitude) { if (magnitude > 3.0) this.play('highMagnitude'); else this.play('eventDetected'); }
    onProcessingComplete() { this.play('success'); }
    onError() { this.play('warning'); }
}

document.addEventListener('DOMContentLoaded', () => { window.audioEngine = new AudioEngine(); });
