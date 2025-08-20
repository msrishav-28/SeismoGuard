/* eslint-disable */
// ThemeManager.js - Non-breaking theme system with CSS variables
/**
 * Manages UI themes via CSS variables and minimal DOM updates.
 * @class ThemeManager
 */
class ThemeManager {
    constructor() {
        this.themes = {
            dark: {
                primary: '#0f0c29',
                secondary: '#302b63',
                accent: '#00ffff',
                text: '#ffffff',
                card: 'rgba(255, 255, 255, 0.08)',
                gradients: {
                    background: 'linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)',
                    button: 'linear-gradient(45deg, #00ff88, #00ffff)'
                }
            },
            light: {
                primary: '#ffffff',
                secondary: '#f0f4f8',
                accent: '#0066cc',
                text: '#1a202c',
                card: 'rgba(0, 0, 0, 0.05)',
                gradients: {
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    button: 'linear-gradient(45deg, #667eea, #764ba2)'
                }
            },
            midnight: {
                primary: '#000814',
                secondary: '#001d3d',
                accent: '#ffd60a',
                text: '#ffffff',
                card: 'rgba(255, 214, 10, 0.1)',
                gradients: {
                    background: 'linear-gradient(135deg, #000814 0%, #001d3d 50%, #003566 100%)',
                    button: 'linear-gradient(45deg, #ffd60a, #ffc300)'
                }
            }
        };
        this.currentTheme = this.loadTheme() || 'dark';
        this.init();
    }
    init() {
        this.createToggleUI();
        this.applyTheme(this.currentTheme);
        this.attachEventListeners();
    }
    createToggleUI() {
        const toggleHTML = `
            <div class="theme-toggle" id="themeToggle">
                <div class="theme-selector">
                    <button class="theme-btn" data-theme="dark" title="Dark Mode">🌙</button>
                    <button class="theme-btn" data-theme="light" title="Light Mode">☀️</button>
                    <button class="theme-btn" data-theme="midnight" title="Midnight Mode">🌌</button>
                </div>
            </div>
        `;
        const style = document.createElement('style');
        style.textContent = `
            .theme-toggle { position: fixed; top: 20px; right: 100px; z-index: 9999; background: var(--card-bg); border-radius: 25px; padding: 5px; backdrop-filter: blur(10px); transition: all 0.3s ease; }
            .theme-selector { display: flex; gap: 5px; }
            .theme-btn { width: 40px; height: 40px; border: none; background: transparent; cursor: pointer; font-size: 20px; transition: all 0.3s ease; border-radius: 20px; }
            .theme-btn:hover { background: var(--accent-color); transform: scale(1.1); }
            .theme-btn.active { background: var(--accent-color); box-shadow: 0 0 20px var(--accent-color); }
        `;
        document.head.appendChild(style);
        document.body.insertAdjacentHTML('beforeend', toggleHTML);
    }
    applyTheme(themeName) {
        const theme = this.themes[themeName];
        const root = document.documentElement;
        root.style.transition = 'all 0.5s ease';
        root.style.setProperty('--primary-color', theme.primary);
        root.style.setProperty('--secondary-color', theme.secondary);
        root.style.setProperty('--accent-color', theme.accent);
        root.style.setProperty('--text-color', theme.text);
        root.style.setProperty('--card-bg', theme.card);
        document.body.style.background = theme.gradients.background;
        this.updateExistingElements(theme);
        localStorage.setItem('seismoguard-theme', themeName);
        this.currentTheme = themeName;
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === themeName);
        });
        window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme: themeName } }));
    }
    updateExistingElements(theme) {
        document.querySelectorAll('.card').forEach(card => {
            card.style.background = theme.card;
            card.style.color = theme.text;
        });
        document.querySelectorAll('.btn').forEach(btn => {
            if (!btn.classList.contains('btn-secondary')) {
                btn.style.background = theme.gradients.button;
            }
        });
        if (window.chartInstances) {
            window.chartInstances.forEach(chart => {
                chart.options.scales.x.ticks.color = theme.text;
                chart.options.scales.y.ticks.color = theme.text;
                chart.update();
            });
        }
    }
    attachEventListeners() {
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.addEventListener('click', () => this.applyTheme(btn.dataset.theme));
        });
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'T') this.cycleTheme();
        });
    }
    cycleTheme() {
        const themes = Object.keys(this.themes);
        const currentIndex = themes.indexOf(this.currentTheme);
        this.applyTheme(themes[(currentIndex + 1) % themes.length]);
    }
    loadTheme() { return localStorage.getItem('seismoguard-theme'); }
}

document.addEventListener('DOMContentLoaded', () => {
    window.themeManager = new ThemeManager();
});
