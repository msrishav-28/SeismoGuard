/* eslint-disable */
// Chatbot.js - Minimal UI that sends chat via WebSocket to backend providers (groq/gemini)
class ChatbotUI {
    constructor(){ this.init(); }
    init(){ this.injectStyles(); this.injectUI(); this.attach(); }
    injectStyles(){ const s=document.createElement('style'); s.textContent=`
        .chatbot{ position:fixed; right:20px; bottom:20px; width:360px; background:var(--card-bg,rgba(255,255,255,.08)); border-radius:12px; padding:10px; z-index:10001; backdrop-filter:blur(10px)}
        .chatbot header{ display:flex; align-items:center; justify-content:space-between; color:var(--text-color,#fff)}
        .chatlog{ height:180px; overflow:auto; background:rgba(0,0,0,.2); border-radius:8px; padding:8px; margin:8px 0; color:var(--text-color,#fff); font-size:12px}
        .msg{ margin:4px 0 }
        .msg.me{ color:#00ffff }
        .msg.bot{ color:#fff }
        .row{ display:flex; gap:6px }
        select, input, button, textarea{ background:rgba(0,0,0,.2); border:1px solid var(--accent-color,#00ffff); color:var(--text-color,#fff); border-radius:8px; padding:6px }
        textarea{ flex:1; resize:vertical; min-height:60px }
    `; document.head.appendChild(s); }
    injectUI(){ const html=`<div class='chatbot' id='chatbot'>
        <header><strong>AI Assistant</strong>
            <select id='chatProvider'>
                <option value='auto' selected>Auto</option>
                <option value='groq'>Groq</option>
                <option value='gemini'>Gemini 2.0 Flash</option>
            </select>
        </header>
        <div style='display:flex; gap:6px; margin:6px 0;'>
            <select id='chatFAQ' style='flex:1'>
                <option value=''>FAQ prompts…</option>
            </select>
            <label title='Include recent stats and last events as context' style='display:flex;align-items:center;gap:6px;color:var(--text-color,#fff)'><input type='checkbox' id='chatWithContext'> Context</label>
        </div>
        <div id='chatlog' class='chatlog' aria-live='polite'></div>
        <div class='row'><textarea id='chatInput' placeholder='Ask a question...'></textarea></div>
        <div class='row'><input id='chatSystem' placeholder='(optional) system prompt' style='flex:1'><button class='btn' id='chatSend'>Send</button></div>
        <div id='chatHint' style='font-size:11px;color:var(--text-color,#ccc);margin-top:6px'>Requires server env vars GROQ_API_KEY and/or GOOGLE_API_KEY</div>
    </div>`; document.body.insertAdjacentHTML('beforeend', html); }
    attach(){
        document.getElementById('chatSend').addEventListener('click', ()=> this.send());
        document.getElementById('chatInput').addEventListener('keydown', (e)=>{ if(e.key==='Enter' && (e.ctrlKey||e.metaKey)) this.send(); });
        const faqSel = document.getElementById('chatFAQ');
        try { (window.chatbotFAQ||[]).forEach((f,i)=>{ const opt=document.createElement('option'); opt.value=String(i); opt.textContent=f.label; faqSel.appendChild(opt); }); } catch {}
        faqSel.addEventListener('change', (e)=>{
            const idx = Number(e.target.value);
            if (Number.isFinite(idx) && window.chatbotFAQ && window.chatbotFAQ[idx]){
                const ta = document.getElementById('chatInput');
                ta.value = window.chatbotFAQ[idx].prompt;
            }
            e.target.value='';
        });
    }
    log(role, text){ const area=document.getElementById('chatlog'); const div=document.createElement('div'); div.className = `msg ${role==='user'?'me':'bot'}`; div.textContent = text; area.appendChild(div); area.scrollTop = area.scrollHeight; }
    send(){
        const input=document.getElementById('chatInput');
        const provider=document.getElementById('chatProvider').value;
        const systemBase=document.getElementById('chatSystem').value||'';
        const withContext=document.getElementById('chatWithContext').checked;
        const msg = input.value.trim(); if(!msg) return; this.log('user', msg); input.value='';
        const sys = withContext ? this.composeContext(systemBase) : (systemBase||undefined);
        try{
            if(window.wsClient){
                const unsub = window.wsClient.subscribe('chat_response', (res)=>{ if(!res) return; const { text, error, provider:prov } = res; this.log('assistant', error?`[${prov||provider}] Error: ${error}`: text||'(empty)'); if (typeof unsub==='function') unsub(); });
                window.wsClient.send({ type:'chat', provider, system: sys, message: msg });
            } else {
                this.log('assistant', '(WebSocket not connected)');
            }
        } catch(e){ this.log('assistant', 'Failed to send'); }
    }
    composeContext(systemBase){
        try{
            const stats = window.statisticsPanel?.stats || {};
            const lastEvents = (window.detectedEvents||[]).slice(0,5);
            const compact = {
                performance: stats.performance,
                events: lastEvents.map(e=> ({ t:e.time||e.timestamp, type:e.type, mag:e.magnitude, conf:e.confidence }))
            };
            return `${systemBase}\nContext:\n${JSON.stringify(compact).slice(0,1500)}`.trim();
        }catch{ return systemBase; }
    }
}

document.addEventListener('DOMContentLoaded', ()=> { window.chatbot = new ChatbotUI(); });
