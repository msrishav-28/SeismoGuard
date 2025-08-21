import os
import sys
from datetime import datetime
import argparse

# Add src to path for imports
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from seismic_detector import PlanetarySeismicDetector  # noqa: E402
try:
    from ml_models import load_or_train_models  # noqa: E402
except Exception:
    load_or_train_models = None  # optional
from websocket_server import SeismicWebSocketServer  # noqa: E402

# Load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="SeismoGuard WebSocket Server")
    parser.add_argument("--planet", default="moon", choices=["moon","mars","earth"], help="Target planet")
    parser.add_argument("--sr", type=float, default=100.0, help="Sampling rate")
    # Fallback training is enabled by default; provide a switch to disable
    parser.add_argument("--no-fallback-train", action="store_true", help="Disable auto-training when model is missing")
    parser.add_argument("--train-only", action="store_true", help="Only train/save model and exit")
    args = parser.parse_args()

    detector = PlanetarySeismicDetector(planet=args.planet, sampling_rate=args.sr)
    models_dir = os.path.join(os.path.dirname(CURRENT_DIR), 'models')

    # Load or optionally train
    if load_or_train_models:
        try:
            model_path = load_or_train_models(
                detector,
                planet=args.planet,
                models_dir=models_dir,
                fallback_train=(not args.no_fallback_train),
            )
            if args.train_only:
                if model_path:
                    print(f"Trained model saved at: {model_path}")
                else:
                    print("Training failed or skipped.")
                return
        except Exception as exc:
            if args.train_only:
                print(f"Training failed: {exc}")
                return
            # proceed without ML
            pass

    server = SeismicWebSocketServer(detector, host='127.0.0.1', port=8765)
    print(f"[{datetime.now().isoformat()}] Starting WebSocket server on ws://127.0.0.1:8765/ws")
    server.start()


if __name__ == '__main__':
    main()
