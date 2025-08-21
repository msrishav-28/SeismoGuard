import os
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any

import httpx


@dataclass
class ChatResult:
    provider: str
    text: str
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    if v is not None and isinstance(v, str):
        v = v.strip()
    return v


async def call_groq(prompt: str, system: Optional[str] = None) -> ChatResult:
    api_key = _get_env("GROQ_API_KEY")
    model = _get_env("GROQ_MODEL", "llama-3.1-70b-versatile")
    if not api_key:
        return ChatResult(provider="groq", text="", error="Missing GROQ_API_KEY in environment")
    url = _get_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                return ChatResult(provider="groq", text="", error=f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return ChatResult(provider="groq", text=text, meta={"model": model})
    except Exception as e:
        return ChatResult(provider="groq", text="", error=str(e))


async def call_gemini(prompt: str, system: Optional[str] = None) -> ChatResult:
    api_key = _get_env("GOOGLE_API_KEY")
    # Default to gemini-2.0-flash per request; allow override
    model = _get_env("GEMINI_MODEL", "gemini-2.0-flash")
    if not api_key:
        return ChatResult(provider="gemini", text="", error="Missing GOOGLE_API_KEY in environment")
    base = _get_env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
    url = f"{base}/v1beta/models/{model}:generateContent?key={api_key}"
    contents = [{
        "role": "user",
        "parts": ([{"text": system}] if system else []) + [{"text": prompt}]
    }]
    payload = {"contents": contents}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                return ChatResult(provider="gemini", text="", error=f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            candidates = data.get("candidates", [])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and isinstance(parts, list):
                    # Concatenate any text parts
                    text = "".join(p.get("text", "") for p in parts)
            return ChatResult(provider="gemini", text=text, meta={"model": model})
    except Exception as e:
        return ChatResult(provider="gemini", text="", error=str(e))


async def route_chat(provider: str, message: str, system: Optional[str] = None) -> ChatResult:
    p = (provider or "auto").lower()
    if p == "groq":
        return await call_groq(message, system)
    if p == "gemini":
        return await call_gemini(message, system)
    # auto: try groq then gemini
    first = await call_groq(message, system)
    if not first.error and first.text:
        return first
    second = await call_gemini(message, system)
    return second
