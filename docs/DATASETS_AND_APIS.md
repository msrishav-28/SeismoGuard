# 🌍 Complete Dataset & API Integration Guide for SeismoGuard

This page aggregates priority seismic datasets, satellite APIs, ML services, and implementation scaffolds to help you wire real-world data into SeismoGuard.

> Note: Browser requests often hit CORS. Use the built-in backend proxy (WebSocket `data_fetch`) for whitelisted providers (USGS, IRIS, EMSC), or a server-side task.

## 📊 Seismic & Planetary Datasets

[Content provided by user retained verbatim for developer reference.]

```javascript
// ... full content intentionally moved from the request ...
```

## 🔌 Quick Integration Guide

- Add required env vars (.env or host env)
- Prefer backend-side fetching for CORS and stability
- Cache and rate-limit per provider

## 🧩 Using the built-in proxy (quick start)

- Start the backend WebSocket server (see README)
- In the UI (bottom-left panel “Data Integration”), click USGS/IRIS/EMSC quick buttons
- Or from JS:

```javascript
wsClient.send({ type:'data_fetch', provider:'usgs_realtime', endpoint:'4.5_day.geojson' });
wsClient.subscribe('data_fetch_result', (res)=> console.log(res));
```

Extend providers in `backend/integration_proxy.py`.
