"""
SeismoGuard - Intelligent Planetary Seismic Detection System
NASA Space Apps Hackathon Project
Backend Implementation for Seismic Event Detection and Classification
"""

import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

import json
import joblib
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, spectrogram

# Lightweight compression integration for metrics
try:
	from .data_compression import compress_window
except Exception:
	def compress_window(arr: np.ndarray) -> bytes:  # type: ignore
		return np.asarray(arr).tobytes()

# Optional plotting
import matplotlib.pyplot as plt

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Deep Learning components (TensorFlow/Keras)
try:
	import tensorflow as tf  # noqa: F401  # type: ignore[import-not-found]
	from tensorflow import keras  # noqa: F401  # type: ignore[import-not-found]
	from tensorflow.keras import layers, models  # type: ignore[import-not-found]
	DEEP_LEARNING_AVAILABLE = True
except ImportError:
	DEEP_LEARNING_AVAILABLE = False
	print("TensorFlow not available. Using traditional ML methods only.")


@dataclass
class SeismicEvent:
	"""Data class for storing seismic event information"""
	timestamp: datetime
	duration: float
	magnitude: float
	event_type: str  # 'moonquake', 'marsquake', 'impact', 'artificial'
	confidence: float
	p_wave_arrival: Optional[float] = None
	s_wave_arrival: Optional[float] = None
	peak_frequency: float = 0.0
	snr: float = 0.0
	location_estimate: Optional[Dict] = None
	raw_data_indices: Optional[Tuple[int, int]] = None


class PlanetarySeismicDetector:
	"""
	Main class for detecting and classifying seismic events in planetary data.
	Optimized for minimal computational resources suitable for space missions.
	"""

	def __init__(self, planet: str = 'moon', sampling_rate: float = 100.0):
		"""
		Initialize the detector with planet-specific parameters.

		Args:
			planet: Target planet ('moon', 'mars', 'earth')
			sampling_rate: Data sampling rate in Hz
		"""
		self.planet = planet
		self.sampling_rate = sampling_rate
		self.events: List[SeismicEvent] = []
		self.scaler = StandardScaler()

		# Planet-specific parameters
		self.params = self._get_planet_params(planet)

		# Initialize detection algorithms
		self.sta_lta_detector = STALTADetector(sampling_rate)
		self.wavelet_detector = WaveletDetector(sampling_rate)
		self.template_matcher = TemplateMatcher()

		# Initialize ML models
		self.ml_classifier = None
		self.anomaly_detector = None
		self.cnn_model = None

		# Performance metrics
		self.metrics = {
			'total_data_points': 0,
			'transmitted_data_points': 0,
			'events_detected': 0,
			'false_positives': 0,
			'compression_ratio': 0,
			'raw_bytes': 0,
			'compressed_bytes': 0
		}

	def _get_planet_params(self, planet: str) -> Dict:
		"""Get planet-specific detection parameters"""
		params = {
			'moon': {
				'noise_level': 1e-9,  # Very low noise environment
				'typical_magnitude_range': (0.5, 5.0),
				'frequency_range': (0.1, 10.0),  # Hz
				'p_wave_velocity': 6.0,  # km/s
				's_wave_velocity': 3.5,  # km/s
				'sta_window': 5.0,  # seconds
				'lta_window': 30.0,  # seconds
				'trigger_ratio': 3.0
			},
			'mars': {
				'noise_level': 1e-8,  # Higher noise due to atmosphere
				'typical_magnitude_range': (1.0, 4.0),
				'frequency_range': (0.1, 5.0),  # Hz
				'p_wave_velocity': 7.0,  # km/s
				's_wave_velocity': 4.0,  # km/s
				'sta_window': 10.0,
				'lta_window': 60.0,
				'trigger_ratio': 2.5
			},
			'earth': {
				'noise_level': 1e-7,  # Reference
				'typical_magnitude_range': (1.0, 9.0),
				'frequency_range': (0.1, 20.0),  # Hz
				'p_wave_velocity': 8.0,  # km/s
				's_wave_velocity': 4.5,  # km/s
				'sta_window': 1.0,
				'lta_window': 10.0,
				'trigger_ratio': 2.0
			}
		}
		return params.get(planet, params['earth'])

	def process_data(self, data: np.ndarray, timestamp: datetime) -> List[SeismicEvent]:
		"""
		Main processing pipeline for seismic data.

		Args:
			data: Raw seismic data array
			timestamp: Starting timestamp of the data

		Returns:
			List of detected seismic events
		"""
		self.metrics['total_data_points'] += len(data)

		# Step 1: Preprocessing
		filtered_data = self.preprocess_data(data)

		# Step 2: Feature extraction
		features = self.extract_features(filtered_data)

		# Step 3: Multi-algorithm detection
		detections = self.detect_events(filtered_data, features)

		# Step 4: Event classification
		if detections:
			events = self.classify_events(filtered_data, detections, features, timestamp)
			self.events.extend(events)

			# Update metrics
			for event in events:
				if event.raw_data_indices:
					start, end = event.raw_data_indices
					length = end - start
					self.metrics['transmitted_data_points'] += length
					# Track compression metrics using codec
					segment = filtered_data[start:end]
					raw_bytes = int(segment.size * segment.dtype.itemsize)
					try:
						comp = compress_window(segment.astype(np.float32))
						comp_bytes = len(comp)
					except Exception:
						comp_bytes = raw_bytes
					self.metrics['raw_bytes'] += raw_bytes
					self.metrics['compressed_bytes'] += comp_bytes

			self.metrics['events_detected'] += len(events)
			self.update_compression_ratio()

			return events

		return []

	def preprocess_data(self, data: np.ndarray) -> np.ndarray:
		"""Apply preprocessing filters to raw data"""
		# Remove mean and trend
		data = signal.detrend(data)

		# Apply bandpass filter based on planet parameters
		low_freq = self.params['frequency_range'][0]
		high_freq = self.params['frequency_range'][1]

		nyquist = self.sampling_rate / 2
		low = low_freq / nyquist
		high = min(high_freq / nyquist, 0.99)

		b, a = butter(4, [low, high], btype='band')
		filtered = filtfilt(b, a, data)

		return filtered

	def extract_features(self, data: np.ndarray) -> Dict:
		"""Extract relevant features from the data"""
		features = {}

		# Time domain features
		features['mean'] = np.mean(data)
		features['std'] = np.std(data)
		features['max'] = np.max(np.abs(data))
		features['rms'] = np.sqrt(np.mean(data**2))
		features['zero_crossings'] = np.sum(np.diff(np.sign(data)) != 0)

		# Frequency domain features
		fft_vals = np.abs(fft(data))
		freqs = fftfreq(len(data), 1/self.sampling_rate)

		# Only use positive frequencies
		fft_vals = fft_vals[:len(fft_vals)//2]
		freqs = freqs[:len(freqs)//2]

		features['peak_frequency'] = freqs[np.argmax(fft_vals)]
		features['spectral_centroid'] = np.sum(freqs * fft_vals) / np.sum(fft_vals)
		features['spectral_bandwidth'] = np.sqrt(np.sum((freqs - features['spectral_centroid'])**2 * fft_vals) / np.sum(fft_vals))

		# Statistical features
		features['kurtosis'] = self._kurtosis(data)
		features['skewness'] = self._skewness(data)

		# Energy features
		features['energy'] = np.sum(data**2)
		features['snr'] = self.calculate_snr(data)

		return features

	def detect_events(self, data: np.ndarray, features: Dict) -> List[Tuple[int, int]]:
		"""
		Multi-algorithm event detection.
		Returns list of (start_index, end_index) tuples.
		"""
		# If input is long, process in overlapping windows to bound CPU/memory
		win_seconds = max(10.0, float(self.params.get('lta_window', 30.0)))
		window = int(self.sampling_rate * win_seconds)
		step = max(window // 2, 1)

		if len(data) > window * 2:
			return self._detect_events_windowed(data, window, step)

		detections = []

		# 1. STA/LTA Detection
		sta_lta_triggers = self.sta_lta_detector.detect(data, self.params)
		detections.extend(sta_lta_triggers)

		# 2. STFT-based spectral energy Detection (replaces heavy CWT)
		wavelet_triggers = self.wavelet_detector.detect(data)
		detections.extend(wavelet_triggers)

		# 3. Template Matching (if templates available)
		if self.template_matcher.has_templates():
			template_triggers = self.template_matcher.detect(data)
			detections.extend(template_triggers)

		# 4. ML-based Anomaly Detection
		if self.anomaly_detector:
			ml_triggers = self.detect_with_ml(data, features)
			detections.extend(ml_triggers)

		# Merge overlapping detections
		merged_detections = self.merge_detections(detections)

		return merged_detections

	def _detect_events_windowed(self, data: np.ndarray, window: int, step: int) -> List[Tuple[int, int]]:
		"""Run detectors on overlapping windows and merge with offset indices."""
		all_dets: List[Tuple[int, int]] = []
		for start in range(0, max(1, len(data) - window + 1), step):
			end = min(len(data), start + window)
			slice_data = data[start:end]
			# Local detections
			local = []
			local.extend(self.sta_lta_detector.detect(slice_data, self.params))
			local.extend(self.wavelet_detector.detect(slice_data))
			if self.template_matcher.has_templates():
				local.extend(self.template_matcher.detect(slice_data))
			if self.anomaly_detector:
				local.extend(self.detect_with_ml(slice_data, {}))
			# Offset to global indices and append
			for s, e in local:
				all_dets.append((s + start, e + start))
		# Merge overlaps from all windows
		return self.merge_detections(all_dets)

	def classify_events(self, data: np.ndarray, detections: List[Tuple[int, int]], 
						features: Dict, timestamp: datetime) -> List[SeismicEvent]:
		"""Classify detected events into different categories"""
		events = []

		for start_idx, end_idx in detections:
			event_data = data[start_idx:end_idx]

			# Extract event-specific features
			event_features = self.extract_features(event_data)

			# Determine event type
			if self.ml_classifier:
				event_type = self.classify_with_ml(event_features)
			else:
				event_type = self.rule_based_classification(event_features)

			# Estimate magnitude
			magnitude = self.estimate_magnitude(event_data, event_features)

			# Calculate timing
			event_time = timestamp + timedelta(seconds=start_idx/self.sampling_rate)
			duration = (end_idx - start_idx) / self.sampling_rate

			# Find P and S wave arrivals
			p_arrival, s_arrival = self.find_wave_arrivals(event_data)

			# Calculate confidence score
			confidence = self.calculate_confidence(event_features, event_type)

			event = SeismicEvent(
				timestamp=event_time,
				duration=duration,
				magnitude=magnitude,
				event_type=event_type,
				confidence=confidence,
				p_wave_arrival=p_arrival,
				s_wave_arrival=s_arrival,
				peak_frequency=event_features['peak_frequency'],
				snr=event_features['snr'],
				raw_data_indices=(start_idx, end_idx)
			)

			events.append(event)

		return events

	def train_ml_models(self, training_data: pd.DataFrame):
		"""Train machine learning models for detection and classification"""
		# Prepare features and labels
		feature_cols = ['mean', 'std', 'max', 'rms', 'peak_frequency', 
					   'spectral_centroid', 'energy', 'snr']
		X = training_data[feature_cols].values
		y = training_data['event_type'].values

		# Split data with stratification
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=42, stratify=y
		)

		# Scale features
		X_train_scaled = self.scaler.fit_transform(X_train)
		X_test_scaled = self.scaler.transform(X_test)

		# Train Random Forest with calibration and CV if available
		rf = RandomForestClassifier(
			n_estimators=200,
			max_depth=None,
			random_state=42,
			n_jobs=-1
		)
		try:
			from sklearn.calibration import CalibratedClassifierCV
			self.ml_classifier = CalibratedClassifierCV(rf, cv=3, method='sigmoid')
			self.ml_classifier.fit(X_train_scaled, y_train)
		except Exception:
			self.ml_classifier = rf
			self.ml_classifier.fit(X_train_scaled, y_train)

		# Train Isolation Forest for anomaly detection
		self.anomaly_detector = IsolationForest(
			contamination=0.1,
			random_state=42
		)
		self.anomaly_detector.fit(X_train_scaled)

		# Train CNN if TensorFlow available (encode labels to integers)
		if DEEP_LEARNING_AVAILABLE:
			try:
				from sklearn.preprocessing import LabelEncoder
				le = LabelEncoder()
				y_train_enc = le.fit_transform(y_train)
				# Keep encoder for potential downstream usage
				self._label_encoder = le
				self.cnn_model = self.build_cnn_model(X_train.shape[1])
				self.cnn_model.fit(
					X_train_scaled, y_train_enc,
					epochs=20,
					batch_size=32,
					validation_split=0.2,
					verbose=0
				)
			except Exception as _cnn_exc:
				print("CNN training skipped:", _cnn_exc)
				self.cnn_model = None

		# Evaluate with confusion matrix and CV
		from sklearn.metrics import confusion_matrix
		from sklearn.model_selection import StratifiedKFold, cross_val_score
		y_pred = self.ml_classifier.predict(X_test_scaled)
		report = classification_report(y_test, y_pred, output_dict=True)
		cm = confusion_matrix(y_test, y_pred).tolist()
		cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
		cv_scores = cross_val_score(self.ml_classifier, X_train_scaled, y_train, cv=cv, scoring='accuracy')
		self._last_training_metadata = {
			'timestamp': datetime.now().isoformat(),
			'planet': self.planet,
			'sampling_rate': self.sampling_rate,
			'feature_cols': feature_cols,
			'cv_mean_accuracy': float(np.mean(cv_scores)),
			'cv_std_accuracy': float(np.std(cv_scores)),
			'confusion_matrix': cm,
			'classification_report': report,
		}
		print("Classification Report:")
		from pprint import pprint as _pprint
		_pprint(report)
		return self.ml_classifier

	def build_cnn_model(self, input_shape: int):
		"""Build a 1D CNN model for event classification"""
		model = models.Sequential([
			layers.Dense(64, activation='relu', input_shape=(input_shape,)),
			layers.Dropout(0.3),
			layers.Dense(32, activation='relu'),
			layers.Dropout(0.3),
			layers.Dense(16, activation='relu'),
			layers.Dense(4, activation='softmax')  # 4 event types
		])

		model.compile(
			optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
		)

		return model

	def calculate_snr(self, data: np.ndarray) -> float:
		"""Calculate Signal-to-Noise Ratio"""
		signal_power = np.mean(data**2)
		noise_power = self.params['noise_level']**2

		if noise_power > 0:
			snr_db = 10 * np.log10(signal_power / noise_power)
			return max(snr_db, 0)
		return 0

	def estimate_magnitude(self, event_data: np.ndarray, features: Dict) -> float:
		"""Estimate event magnitude based on amplitude and duration"""
		max_amplitude = np.max(np.abs(event_data))
		duration = len(event_data) / self.sampling_rate

		# Simplified magnitude estimation (would need calibration with real data)
		magnitude = np.log10(max_amplitude * 1e9) + 0.5 * np.log10(duration)

		# Constrain to typical range
		min_mag, max_mag = self.params['typical_magnitude_range']
		return np.clip(magnitude, min_mag, max_mag)

	def find_wave_arrivals(self, event_data: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
		"""Detect P and S wave arrival times within an event"""
		# Use AIC (Akaike Information Criterion) picker
		aic = self.calculate_aic(event_data)

		# Find minimum AIC for P-wave
		p_idx = np.argmin(aic[:len(aic)//2])
		p_arrival = p_idx / self.sampling_rate if p_idx > 0 else None

		# Find minimum AIC for S-wave (after P-wave)
		if p_idx > 0:
			s_aic = aic[p_idx:]
			s_idx = np.argmin(s_aic) + p_idx
			s_arrival = s_idx / self.sampling_rate if s_idx > p_idx else None
		else:
			s_arrival = None

		return p_arrival, s_arrival

	def calculate_aic(self, data: np.ndarray) -> np.ndarray:
		"""Calculate Akaike Information Criterion for phase picking"""
		n = len(data)
		aic = np.zeros(n)

		for k in range(1, n-1):
			var1 = np.var(data[:k]) if k > 1 else 1e-10
			var2 = np.var(data[k:]) if k < n-1 else 1e-10

			aic[k] = k * np.log(var1) + (n - k - 1) * np.log(var2)

		return aic

	def rule_based_classification(self, features: Dict) -> str:
		"""Simple rule-based event classification"""
		if features['peak_frequency'] < 0.5:
			return 'deep_moonquake'
		elif features['peak_frequency'] > 5.0:
			return 'impact'
		elif features['energy'] > 1e6:
			return 'shallow_moonquake'
		else:
			return 'thermal'

	def classify_with_ml(self, features: Dict) -> str:
		"""Classify event using trained ML model"""
		feature_vector = np.array([
			features.get('mean', 0),
			features.get('std', 0),
			features.get('max', 0),
			features.get('rms', 0),
			features.get('peak_frequency', 0),
			features.get('spectral_centroid', 0),
			features.get('energy', 0),
			features.get('snr', 0)
		]).reshape(1, -1)

		feature_scaled = self.scaler.transform(feature_vector)
		prediction = self.ml_classifier.predict(feature_scaled)[0]

		return prediction

	def detect_with_ml(self, data: np.ndarray, features: Dict) -> List[Tuple[int, int]]:
		"""Use ML anomaly detection to find events"""
		# Slide window through data
		window_size = int(self.sampling_rate * 10)  # 10 second windows
		step_size = window_size // 2

		detections = []

		for i in range(0, len(data) - window_size, step_size):
			window_data = data[i:i+window_size]
			window_features = self.extract_features(window_data)

			feature_vector = np.array([
				window_features.get('mean', 0),
				window_features.get('std', 0),
				window_features.get('max', 0),
				window_features.get('rms', 0),
				window_features.get('peak_frequency', 0),
				window_features.get('spectral_centroid', 0),
				window_features.get('energy', 0),
				window_features.get('snr', 0)
			]).reshape(1, -1)

			feature_scaled = self.scaler.transform(feature_vector)

			# Check if anomaly (potential event)
			if self.anomaly_detector.predict(feature_scaled)[0] == -1:
				detections.append((i, i + window_size))

		return detections

	def calculate_confidence(self, features: Dict, event_type: str) -> float:
		"""Calculate confidence score for an event classification"""
		confidence = 50.0  # Base confidence

		# Adjust based on SNR
		if features['snr'] > 20:
			confidence += 30
		elif features['snr'] > 10:
			confidence += 20
		elif features['snr'] > 5:
			confidence += 10

		# Adjust based on event characteristics matching expected patterns
		if event_type == 'impact' and features['peak_frequency'] > 5.0:
			confidence += 10
		elif event_type == 'deep_moonquake' and features['peak_frequency'] < 1.0:
			confidence += 10

		# Ensure confidence is within 0-100 range
		return min(max(confidence, 0), 100)

	def merge_detections(self, detections: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
		"""Merge overlapping detection windows"""
		if not detections:
			return []

		# Sort by start index
		sorted_detections = sorted(detections, key=lambda x: x[0])
		merged = [sorted_detections[0]]

		for current in sorted_detections[1:]:
			last = merged[-1]

			# Check for overlap
			if current[0] <= last[1]:
				# Merge
				merged[-1] = (last[0], max(last[1], current[1]))
			else:
				merged.append(current)

		return merged

	def update_compression_ratio(self):
		"""Calculate data compression ratio using bytes if available."""
		if self.metrics.get('compressed_bytes', 0) > 0:
			ratio = self.metrics['raw_bytes'] / max(1, self.metrics['compressed_bytes'])
			self.metrics['compression_ratio'] = float(ratio)
			return
		# Fallback if no byte metrics yet
		if self.metrics['transmitted_data_points'] > 0:
			ratio = self.metrics['total_data_points'] / self.metrics['transmitted_data_points']
			self.metrics['compression_ratio'] = ratio
		else:
			self.metrics['compression_ratio'] = float('inf')

	def _kurtosis(self, data: np.ndarray) -> float:
		"""Calculate kurtosis of the data"""
		n = len(data)
		mean = np.mean(data)
		std = np.std(data)

		if std == 0:
			return 0

		return np.sum(((data - mean) / std) ** 4) / n - 3

	def _skewness(self, data: np.ndarray) -> float:
		"""Calculate skewness of the data"""
		n = len(data)
		mean = np.mean(data)
		std = np.std(data)

		if std == 0:
			return 0

		return np.sum(((data - mean) / std) ** 3) / n

	def export_events(self, filename: str):
		"""Export detected events to JSON file"""
		events_data = []

		for event in self.events:
			events_data.append({
				'timestamp': event.timestamp.isoformat(),
				'duration': event.duration,
				'magnitude': event.magnitude,
				'type': event.event_type,
				'confidence': event.confidence,
				'peak_frequency': event.peak_frequency,
				'snr': event.snr,
				'p_wave_arrival': event.p_wave_arrival,
				's_wave_arrival': event.s_wave_arrival
			})

		export_data = {
			'mission': self.planet.upper(),
			'sampling_rate': self.sampling_rate,
			'events': events_data,
			'metrics': self.metrics,
			'model_metadata': getattr(self, '_last_training_metadata', None),
			'compression_ratio': f"{self.metrics['compression_ratio']:.1f}:1"
		}

		with open(filename, 'w') as f:
			json.dump(export_data, f, indent=2)

		print(f"Exported {len(self.events)} events to {filename}")
		print(f"Compression ratio: {self.metrics['compression_ratio']:.1f}:1")

	def save_model(self, filename: str):
		"""Save trained ML models"""
		model_data = {
			'scaler': self.scaler,
			'ml_classifier': self.ml_classifier,
			'anomaly_detector': self.anomaly_detector,
			'planet_params': self.params,
			'metadata': getattr(self, '_last_training_metadata', {
				'timestamp': datetime.now().isoformat(),
				'planet': self.planet,
				'sampling_rate': self.sampling_rate,
				'feature_cols': ['mean','std','max','rms','peak_frequency','spectral_centroid','energy','snr']
			})
		}

		joblib.dump(model_data, filename)
		print(f"Model saved to {filename}")

	def load_model(self, filename: str):
		"""Load pre-trained ML models"""
		model_data = joblib.load(filename)

		self.scaler = model_data['scaler']
		self.ml_classifier = model_data['ml_classifier']
		self.anomaly_detector = model_data['anomaly_detector']
		self._last_training_metadata = model_data.get('metadata', None)

		print(f"Model loaded from {filename}")


class STALTADetector:
	"""Short-Term Average / Long-Term Average detector"""

	def __init__(self, sampling_rate: float):
		self.sampling_rate = sampling_rate
		self.enable_adaptive = True

	def detect(self, data: np.ndarray, params: Dict) -> List[Tuple[int, int]]:
		"""Detect events using STA/LTA algorithm"""
		sta_window = int(params['sta_window'] * self.sampling_rate)
		lta_window = int(params['lta_window'] * self.sampling_rate)
		trigger_ratio = params['trigger_ratio']

		# Calculate STA/LTA ratio
		sta = self._moving_average(data**2, sta_window)
		lta = self._moving_average(data**2, lta_window)

		# Avoid division by zero
		lta[lta < 1e-10] = 1e-10

		ratio = sta / lta

		# Adaptive threshold using robust stats on ratio
		if self.enable_adaptive:
			med = float(np.median(ratio))
			mad = float(np.median(np.abs(ratio - med)) + 1e-9)
			thr_up = max(trigger_ratio, med + 3.5 * mad)
			thr_down = max(trigger_ratio * 0.5, med + 1.75 * mad)
		else:
			thr_up = trigger_ratio
			thr_down = trigger_ratio * 0.5

		# Find triggers
		triggers = []
		in_event = False
		start_idx = 0

		for i in range(len(ratio)):
			if not in_event and ratio[i] > thr_up:
				in_event = True
				start_idx = i
			elif in_event and ratio[i] < thr_down:
				in_event = False
				triggers.append((start_idx, i))

		return triggers

	def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
		"""Calculate moving average"""
		padded = np.pad(data, (window//2, window//2), mode='edge')
		return np.convolve(padded, np.ones(window)/window, mode='valid')


class WaveletDetector:
	"""STFT-based energy burst detector (replaces heavy CWT for realtime)."""

	def __init__(self, sampling_rate: float):
		self.sampling_rate = sampling_rate

	def detect(self, data: np.ndarray) -> List[Tuple[int, int]]:
		if len(data) < 64:
			return []
		nper = min(256, max(64, len(data)//8))
		over = int(nper * 0.5)
		f, t, Sxx = spectrogram(data, fs=self.sampling_rate, nperseg=nper, noverlap=over, scaling='spectrum', mode='psd')
		# Focus on <=10 Hz by default
		band = f <= 10.0
		if not np.any(band):
			band = slice(None)
		energy_t = np.sum(Sxx[band, :], axis=0)
		if np.max(energy_t) > 0:
			energy = energy_t / np.max(energy_t)
		else:
			energy = energy_t
		med = float(np.median(energy))
		mad = float(np.median(np.abs(energy - med)) + 1e-9)
		thr_up = max(0.2, med + 3.0 * mad)
		thr_down = max(0.1, med + 1.5 * mad)
		frame_len = nper - over
		triggers: List[Tuple[int, int]] = []
		in_event = False
		start_idx_samp = 0
		for i in range(len(energy)):
			if not in_event and energy[i] > thr_up:
				in_event = True
				start_idx_samp = i * frame_len
			elif in_event and energy[i] < thr_down:
				in_event = False
				end_idx_samp = (i + 1) * frame_len
				if end_idx_samp - start_idx_samp > max(10, frame_len // 2):
					triggers.append((max(0, start_idx_samp), min(len(data), end_idx_samp)))
		return triggers


class TemplateMatcher:
	"""Template matching for known event patterns"""

	def __init__(self):
		self.templates = []

	def add_template(self, template: np.ndarray, event_type: str):
		"""Add a template for matching"""
		self.templates.append({
			'data': template / np.max(np.abs(template)),  # Normalize
			'type': event_type
		})

	def has_templates(self) -> bool:
		"""Check if templates are available"""
		return len(self.templates) > 0

	def detect(self, data: np.ndarray, threshold: float = 0.8) -> List[Tuple[int, int]]:
		"""Detect events by matching templates"""
		if not self.templates:
			return []

		triggers = []

		for template in self.templates:
			# Cross-correlation
			correlation = np.correlate(data, template['data'], mode='same')
			correlation = correlation / np.max(np.abs(correlation))

			# Find peaks above threshold
			peaks = np.where(correlation > threshold)[0]

			# Group consecutive peaks into events
			if len(peaks) > 0:
				groups = np.split(peaks, np.where(np.diff(peaks) > 100)[0] + 1)

				for group in groups:
					if len(group) > 0:
						start = max(0, group[0] - len(template['data'])//2)
						end = min(len(data), group[-1] + len(template['data'])//2)
						triggers.append((start, end))

		return triggers


class DataSimulator:
	"""Simulate seismic data for testing"""

	def __init__(self, sampling_rate: float = 100.0):
		self.sampling_rate = sampling_rate

	def generate_noise(self, duration: float, noise_level: float = 1e-9) -> np.ndarray:
		"""Generate background noise"""
		n_samples = int(duration * self.sampling_rate)

		# Combine different noise sources
		white_noise = np.random.randn(n_samples) * noise_level

		# Add colored noise (1/f)
		freqs = np.fft.fftfreq(n_samples, 1/self.sampling_rate)
		f_noise = np.zeros(n_samples, dtype=complex)
		f_noise[1:] = np.random.randn(n_samples-1) / np.abs(freqs[1:])
		colored_noise = np.real(np.fft.ifft(f_noise)) * noise_level

		return white_noise + colored_noise

	def generate_moonquake(self, duration: float = 60.0, magnitude: float = 2.0) -> np.ndarray:
		"""Generate simulated moonquake signal"""
		n_samples = int(duration * self.sampling_rate)
		t = np.linspace(0, duration, n_samples)

		# Envelope: gradual rise and exponential decay
		rise_time = duration * 0.1
		peak_time = duration * 0.2

		envelope = np.zeros(n_samples)
		rise_idx = int(rise_time * self.sampling_rate)
		peak_idx = int(peak_time * self.sampling_rate)

		# Rise phase
		envelope[:rise_idx] = np.linspace(0, 1, rise_idx)

		# Peak phase
		envelope[rise_idx:peak_idx] = 1

		# Decay phase
		decay_rate = 0.1
		envelope[peak_idx:] = np.exp(-decay_rate * (t[peak_idx:] - t[peak_idx]))

		# Generate signal with multiple frequency components
		signal_arr = np.zeros(n_samples)

		# P-wave (higher frequency)
		p_freq = np.random.uniform(2, 5)
		signal_arr += np.sin(2 * np.pi * p_freq * t) * envelope * 0.7

		# S-wave (lower frequency, arrives later)
		s_delay = duration * 0.15
		s_idx = int(s_delay * self.sampling_rate)
		s_freq = np.random.uniform(0.5, 2)
		s_signal = np.zeros(n_samples)
		s_signal[s_idx:] = np.sin(2 * np.pi * s_freq * t[s_idx:])
		signal_arr += s_signal * envelope

		# Scale by magnitude
		amplitude = 10 ** (magnitude - 9)

		return signal_arr * amplitude

	def generate_impact(self, duration: float = 10.0, magnitude: float = 1.5) -> np.ndarray:
		"""Generate simulated impact signal"""
		n_samples = int(duration * self.sampling_rate)
		t = np.linspace(0, duration, n_samples)

		# Sharp spike followed by rapid decay
		impact_time = duration * 0.1
		impact_idx = int(impact_time * self.sampling_rate)

		signal_arr = np.zeros(n_samples)

		# Impact spike
		signal_arr[impact_idx] = 1

		# Ringing with rapid decay
		decay_rate = 2.0
		for i in range(impact_idx + 1, n_samples):
			dt = (i - impact_idx) / self.sampling_rate
			signal_arr[i] = np.sin(2 * np.pi * 10 * dt) * np.exp(-decay_rate * dt)

		# Scale by magnitude
		amplitude = 10 ** (magnitude - 9)

		return signal_arr * amplitude

	def generate_dataset(self, n_events: int = 100) -> Tuple[np.ndarray, pd.DataFrame]:
		"""Generate a complete dataset with multiple events"""
		total_duration = 3600  # 1 hour
		data = self.generate_noise(total_duration)

		events = []

		for _ in range(n_events):
			# Random event type
			event_type = np.random.choice(['moonquake', 'impact'], p=[0.7, 0.3])

			# Random timing
			start_time = np.random.uniform(60, total_duration - 120)
			start_idx = int(start_time * self.sampling_rate)

			if event_type == 'moonquake':
				event_duration = np.random.uniform(30, 90)
				magnitude = np.random.uniform(1.0, 3.5)
				event_signal = self.generate_moonquake(event_duration, magnitude)
			else:
				event_duration = np.random.uniform(5, 20)
				magnitude = np.random.uniform(0.5, 2.0)
				event_signal = self.generate_impact(event_duration, magnitude)

			# Add event to data
			end_idx = min(start_idx + len(event_signal), len(data))
			data[start_idx:end_idx] += event_signal[:end_idx-start_idx]

			# Record event
			events.append({
				'start_time': start_time,
				'end_time': start_time + event_duration,
				'event_type': event_type,
				'magnitude': magnitude
			})

		events_df = pd.DataFrame(events)

		return data, events_df


def visualize_results(data: np.ndarray, events: List[SeismicEvent], sampling_rate: float):
	"""Create visualization of detection results"""
	fig, axes = plt.subplots(3, 1, figsize=(12, 8))

	# Time axis
	time = np.arange(len(data)) / sampling_rate

	# Plot 1: Raw seismic data
	axes[0].plot(time, data, 'b-', linewidth=0.5, alpha=0.7)
	axes[0].set_ylabel('Amplitude')
	axes[0].set_title('Raw Seismic Data with Detected Events')
	axes[0].grid(True, alpha=0.3)

	# Mark detected events
	for event in events:
		if event.raw_data_indices:
			start_idx, end_idx = event.raw_data_indices
			event_time = time[start_idx:end_idx]
			event_data = data[start_idx:end_idx]
			axes[0].plot(event_time, event_data, 'r-', linewidth=1.5, alpha=0.8)
			axes[0].axvspan(time[start_idx], time[end_idx], alpha=0.2, color='red')

	# Plot 2: Spectrogram
	f, t_spec, Sxx = spectrogram(data, sampling_rate, nperseg=256)
	axes[1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
	axes[1].set_ylabel('Frequency (Hz)')
	axes[1].set_title('Spectrogram')
	axes[1].set_ylim([0, 10])

	# Plot 3: Event timeline
	axes[2].set_xlim([0, time[-1]])
	axes[2].set_ylim([0, 4])
	axes[2].set_xlabel('Time (seconds)')
	axes[2].set_ylabel('Magnitude')
	axes[2].set_title('Event Timeline')
	axes[2].grid(True, alpha=0.3)

	for i, event in enumerate(events):
		if event.raw_data_indices:
			start_idx, _ = event.raw_data_indices
			event_time = time[start_idx]

			# Plot event marker
			color = 'red' if 'quake' in event.event_type else 'blue'
			axes[2].scatter(event_time, event.magnitude, s=100, c=color, alpha=0.7, zorder=5)
			axes[2].text(event_time, event.magnitude + 0.2, f"E{i+1}", ha='center', fontsize=8)

	axes[2].legend(['Moonquake', 'Impact'], loc='upper right')

	plt.tight_layout()
	plt.savefig('seismic_detection_results.png', dpi=150, bbox_inches='tight')
	plt.show()


def main():
	"""Main execution function for demonstration"""
	print("=" * 60)
	print("SeismoGuard - Intelligent Planetary Seismic Detection")
	print("NASA Space Apps Hackathon Project")
	print("=" * 60)

	# Initialize detector for Moon
	detector = PlanetarySeismicDetector(planet='moon', sampling_rate=100.0)

	# Generate simulated data
	print("\nGenerating simulated lunar seismic data...")
	simulator = DataSimulator(sampling_rate=100.0)
	data, true_events = simulator.generate_dataset(n_events=10)

	print(f"Generated {len(data)/100:.1f} seconds of data with {len(true_events)} events")

	# Process data
	print("\nProcessing data for event detection...")
	timestamp = datetime.now()
	detected_events = detector.process_data(data, timestamp)

	print(f"\nDetected {len(detected_events)} seismic events")

	# Display results
	print("\nDetected Events:")
	print("-" * 50)
	for event in detected_events:
		print(f"Time: {event.timestamp.strftime('%H:%M:%S')}")
		print(f"  Type: {event.event_type}")
		print(f"  Magnitude: {event.magnitude:.2f}")
		print(f"  Duration: {event.duration:.1f}s")
		print(f"  Confidence: {event.confidence:.1f}%")
		print(f"  SNR: {event.snr:.1f} dB")
		print(f"  Peak Frequency: {event.peak_frequency:.2f} Hz")
		print("-" * 50)

	# Calculate and display metrics
	print("\nPerformance Metrics:")
	print(f"  Total data points: {detector.metrics['total_data_points']:,}")
	print(f"  Transmitted points: {detector.metrics['transmitted_data_points']:,}")
	print(f"  Compression ratio: {detector.metrics['compression_ratio']:.1f}:1")
	if detector.metrics['compression_ratio'] != float('inf'):
		reduction = (1 - 1/detector.metrics['compression_ratio']) * 100
		print(f"  Data reduction: {reduction:.1f}%")

	# Export results
	output_file = f"seismic_events_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
	detector.export_events(output_file)

	# Visualization
	print("\nGenerating visualization...")
	visualize_results(data, detected_events, detector.sampling_rate)

	print("\nProcessing complete!")
	print("=" * 60)


if __name__ == "__main__":
	main()

