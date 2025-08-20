"""
ML model management for SeismoGuard

Provides helpers to:
- Generate a compact training dataset from the built-in simulator
- Train classical ML models (RandomForest + IsolationForest via detector API)
- Save/load models, and a load-or-train convenience for bootstrapping

This module intentionally keeps dependencies minimal and reuses
PlanetarySeismicDetector's existing training methods and feature extractor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd

# Reuse simulator and feature extraction from detector
from .seismic_detector import PlanetarySeismicDetector, DataSimulator  # noqa: E402


@dataclass
class TrainingConfig:
    planet: str = "moon"
    sampling_rate: float = 100.0
    duration_seconds: int = 1800  # 30 min
    n_events: int = 60
    output_model_path: Optional[Path] = None


def generate_training_dataframe(
    detector: PlanetarySeismicDetector,
    duration_seconds: int = 1800,
    n_events: int = 60,
) -> pd.DataFrame:
    """
    Generate a labeled feature dataframe using the built-in simulator.

    Each simulated event window is converted to a row of features
    using the detector.extract_features, with label in 'event_type'.
    """
    sim = DataSimulator(sampling_rate=detector.sampling_rate)
    # Generate a longer noise background then insert random events via sim.generate_dataset
    data, events_df = sim.generate_dataset(n_events=n_events)

    rows: List[Dict] = []
    for _, row in events_df.iterrows():
        start_idx = int(row["start_time"] * detector.sampling_rate)
        end_idx = int(row["end_time"] * detector.sampling_rate)
        if end_idx <= start_idx or start_idx < 0 or end_idx > len(data):
            continue
        segment = data[start_idx:end_idx]
        feats = detector.extract_features(segment)
        feats["event_type"] = row["event_type"]
        rows.append(feats)

    if not rows:
        return pd.DataFrame(columns=[
            "mean","std","max","rms","zero_crossings","peak_frequency",
            "spectral_centroid","spectral_bandwidth","kurtosis","skewness",
            "energy","snr","event_type"
        ])

    return pd.DataFrame(rows)


def train_and_save_models(
    detector: PlanetarySeismicDetector,
    output_path: Path,
    duration_seconds: int = 1200,
    n_events: int = 40,
) -> Path:
    """Train ML models using simulated data and save them to output_path."""
    df = generate_training_dataframe(detector, duration_seconds, n_events)
    if df.empty:
        raise RuntimeError("Training dataset is empty; simulation did not produce valid events.")

    detector.train_ml_models(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save_model(str(output_path))
    return output_path


def load_or_train_models(
    detector: PlanetarySeismicDetector,
    planet: str,
    models_dir: Path | str = "models",
    fallback_train: bool = True,
) -> Optional[Path]:
    """
    Try to load a pre-trained model for the selected planet; if missing and
    fallback_train=True, train a compact model from simulation and save it.
    Returns the path of the loaded/saved model, or None if neither succeeded.
    """
    models_dir = Path(models_dir)
    candidate = models_dir / f"{planet}_detector.pkl"

    if candidate.exists():
        detector.load_model(str(candidate))
        return candidate

    if not fallback_train:
        return None

    # Train a fast, compact model as a fallback
    try:
        return train_and_save_models(detector, candidate, duration_seconds=900, n_events=30)
    except Exception:
        # Leave detector in rule-based mode if training fails
        return None
