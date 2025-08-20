import asyncio
import json
import logging
from datetime import datetime

import numpy as np
import websockets


class SeismicWebSocketServer:
    def __init__(self, detector, host='localhost', port=8080):
        self.detector = detector
        self.host = host
        self.port = port
        self.clients = set()

    async def register(self, websocket):
        self.clients.add(websocket)
        await websocket.send(json.dumps({
            'type': 'connection',
            'status': 'connected',
            'timestamp': datetime.now().isoformat()
        }))

    async def unregister(self, websocket):
        self.clients.discard(websocket)

    async def send_to_all(self, message: str):
        if self.clients:
            await asyncio.gather(*[client.send(message) for client in self.clients], return_exceptions=True)

    async def handle_client(self, websocket, path):
        # Accept / and /ws paths for client compatibility
        try:
            logging.info("Client connected from %s path=%s", getattr(websocket, 'remote_address', None), path)
        except Exception:
            pass
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                except Exception:
                    await websocket.send(json.dumps({'type': 'error', 'message': 'Invalid JSON'}))
                    continue
                await self.process_message(data, websocket)
        finally:
            await self.unregister(websocket)

    async def process_message(self, data, websocket):
        msg_type = data.get('type')
        if msg_type == 'subscribe':
            channels = data.get('channels', [])
            await websocket.send(json.dumps({'type': 'subscribed', 'channels': channels}))
        elif msg_type == 'start_stream':
            asyncio.create_task(self.stream_data(websocket))
        elif msg_type == 'process':
            payload = data.get('data')
            if not isinstance(payload, (list, tuple)):
                await websocket.send(json.dumps({'type': 'error', 'message': 'data must be a list of numbers'}))
                return
            # Input size limit to protect event loop
            max_len = 200_000  # ~2s at 100kHz or ample for our 100Hz use
            if len(payload) > max_len:
                await websocket.send(json.dumps({'type': 'error', 'message': f'data too large (>{max_len})'}))
                return
            try:
                arr = np.asarray(payload, dtype=float)
            except Exception:
                await websocket.send(json.dumps({'type': 'error', 'message': 'data must be numeric'}))
                return

            # Offload CPU work to a worker thread to avoid blocking the WS loop
            start_ts = datetime.now()
            events = await asyncio.to_thread(self.detector.process_data, arr, start_ts)
            result = {'type': 'processing_result', 'events': [self.event_to_dict(e) for e in events]}
            try:
                await websocket.send(json.dumps(result))
            except Exception as exc:
                logging.exception("Failed to send processing_result: %s", exc)

    async def stream_data(self, websocket):
        while True:
            try:
                samples = (np.random.randn(100) * 1e-9).tolist()
                await websocket.send(json.dumps({'type': 'seismic_data', 'payload': {'samples': samples, 'timestamp': datetime.now().isoformat()}}))
                await asyncio.sleep(1)
            except Exception:
                break

    def event_to_dict(self, event):
        return {
            'timestamp': event.timestamp.isoformat(),
            'magnitude': event.magnitude,
            'type': event.event_type,
            'confidence': event.confidence,
            'duration': event.duration,
        }

    async def _run(self):
        logging.basicConfig(level=logging.INFO)
        async with websockets.serve(self.handle_client, self.host, self.port):
            # Run forever until cancelled
            async def stats_task():
                while True:
                    try:
                        payload = {
                            'events_detected': self.detector.metrics.get('events_detected', 0),
                            'compression_ratio': self.detector.metrics.get('compression_ratio', 0),
                            'total_data_points': self.detector.metrics.get('total_data_points', 0),
                            'transmitted_data_points': self.detector.metrics.get('transmitted_data_points', 0),
                            'raw_bytes': self.detector.metrics.get('raw_bytes', 0),
                            'compressed_bytes': self.detector.metrics.get('compressed_bytes', 0),
                        }
                        await self.send_to_all(json.dumps({'type': 'statistics', 'payload': payload}))
                        await asyncio.sleep(5)
                    except Exception:
                        await asyncio.sleep(5)
            await asyncio.gather(asyncio.create_task(stats_task()), asyncio.Future())

    def start(self):
        asyncio.run(self._run())
