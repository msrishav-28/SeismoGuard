import numpy as np

from src.data_compression import compress_window, decompress_window, compression_ratio
from src.seismic_detector import PlanetarySeismicDetector


def test_compression_roundtrip():
    arr = (np.sin(np.linspace(0, 10*np.pi, 1000)) * 1e-7).astype(np.float32)
    payload = compress_window(arr)
    restored = decompress_window(payload)
    assert restored.shape == arr.shape
    # Allow tiny numeric diff due to float32 ops
    assert np.allclose(restored, arr, atol=1e-6)


def test_compression_ratio_reasonable():
    arr = (np.sin(np.linspace(0, 8*np.pi, 2000)) * 1e-8).astype(np.float32)
    ratio = compression_ratio(arr)
    assert ratio > 1.1  # should compress a bit


def test_stft_detector_runs():
    det = PlanetarySeismicDetector(planet='moon', sampling_rate=100.0)
    # synth with a short burst
    t = np.linspace(0, 30, int(30*det.sampling_rate), endpoint=False)
    x = np.random.randn(t.size) * 1e-9
    x[1000:1500] += 5e-8 * np.sin(2*np.pi*3*t[:500])
    feats = det.extract_features(x)
    dets = det.detect_events(x, feats)
    assert isinstance(dets, list)
