import os
import sys
from datetime import datetime

# Add src to path for imports
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from seismic_detector import PlanetarySeismicDetector  # noqa: E402
from websocket_server import SeismicWebSocketServer  # noqa: E402


def main():
    detector = PlanetarySeismicDetector(planet='moon', sampling_rate=100.0)
    server = SeismicWebSocketServer(detector, host='127.0.0.1', port=8765)
    print(f"[{datetime.now().isoformat()}] Starting WebSocket server on ws://127.0.0.1:8765/ws")
    server.start()


if __name__ == '__main__':
    main()
