import asyncio
import json
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

    async def handle_client(self, websocket):
        await self.register(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
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
            arr = np.array(data['data'])
            events = self.detector.process_data(arr, datetime.now())
            await websocket.send(json.dumps({'type': 'processing_result', 'events': [self.event_to_dict(e) for e in events]}))

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
        async with websockets.serve(self.handle_client, self.host, self.port):
            # Run forever until cancelled
            await asyncio.Future()

    def start(self):
        asyncio.run(self._run())
