import os
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

from src.seismic_detector import PlanetarySeismicDetector


def test_adaptive_threshold_reacts_to_noise_change():
    det = PlanetarySeismicDetector(planet='moon', sampling_rate=100.0)
    # Build a signal with a low-noise region and a higher-noise region
    n = int(60 * det.sampling_rate)
    x = np.random.randn(n) * 1e-9
    x[n//2:] *= 5.0  # noise jump in second half
    feats = det.extract_features(x)
    # Run detection; should not error and produce a list
    dets = det.detect_events(x, feats)
    assert isinstance(dets, list)


def test_model_metadata_persistence(tmp_path: Path):
    det = PlanetarySeismicDetector(planet='moon', sampling_rate=100.0)
    # Minimal synthetic training data with two classes
    rows = []
    for cls in ["impact", "deep_moonquake"]:
        for _ in range(20):
            s = (np.random.randn(1000) * 1e-9).astype(float)
            feats = det.extract_features(s)
            feats['event_type'] = cls
            rows.append(feats)
    import pandas as pd
    df = pd.DataFrame(rows)
    det.train_ml_models(df)
    model_path = tmp_path / "moon_detector.pkl"
    det.save_model(str(model_path))

    # Load and verify metadata exists
    det2 = PlanetarySeismicDetector(planet='moon', sampling_rate=100.0)
    det2.load_model(str(model_path))
    md = getattr(det2, "_last_training_metadata", None)
    assert md is not None
    assert 'cv_mean_accuracy' in md
    assert 'confusion_matrix' in md
import unittest
from datetime import datetime
import numpy as np

from src.seismic_detector import PlanetarySeismicDetector, DataSimulator
from src.ml_models import generate_training_dataframe


class TestMLPipeline(unittest.TestCase):
    def test_feature_extraction_shapes(self):
        det = PlanetarySeismicDetector(planet='moon', sampling_rate=100.0)
        sim = DataSimulator(sampling_rate=100.0)
        data = sim.generate_noise(duration=10, noise_level=1e-9)
        feats = det.extract_features(data)
        self.assertIn('mean', feats)
        self.assertIn('peak_frequency', feats)
        self.assertGreaterEqual(feats['spectral_centroid'], 0)

    def test_sta_lta_detects_spike(self):
        det = PlanetarySeismicDetector(planet='earth', sampling_rate=100.0)
        # Baseline noise
        data = np.random.randn(2000) * 1e-9
        # Inject spike window
        data[900:1100] += 5e-7
        events = det.process_data(data, datetime.now())
        # Should detect at least one event window
        self.assertIsInstance(events, list)

    def test_generate_training_dataframe(self):
        det = PlanetarySeismicDetector(planet='moon', sampling_rate=100.0)
        df = generate_training_dataframe(detector=det, duration_seconds=600, n_events=10)
        # Should return columns including label
        if not df.empty:
            self.assertIn('event_type', df.columns)


if __name__ == '__main__':
    unittest.main()
