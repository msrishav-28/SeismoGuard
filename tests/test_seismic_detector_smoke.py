import unittest
from datetime import datetime
import numpy as np

from src.seismic_detector import PlanetarySeismicDetector, DataSimulator


class TestSeismicDetectorSmoke(unittest.TestCase):
    def test_process_simulated_data(self):
        detector = PlanetarySeismicDetector(planet='moon', sampling_rate=100.0)
        sim = DataSimulator(sampling_rate=100.0)
        # Keep it small to be fast
        noise = sim.generate_noise(duration=30, noise_level=1e-9)
        quake = sim.generate_moonquake(duration=10, magnitude=2.0)
        # Insert event into middle
        data = noise.copy()
        start = 1000
        end = start + len(quake)
        data[start:end] += quake[: len(data) - start]

        events = detector.process_data(data, datetime.now())
        # Should run without error and return a list
        self.assertIsInstance(events, list)


if __name__ == "__main__":
    unittest.main()
