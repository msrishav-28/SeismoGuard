"""
Lightweight compression utilities for seismic event windows.

Implements a simple differential (delta) encoding on float32 samples
followed by zlib compression. This is not optimal but is fast and
portable, giving realistic compression metrics for small windows.
"""
from __future__ import annotations

import io
import struct
import zlib
from typing import Tuple

import numpy as np


def _delta_encode(arr: np.ndarray) -> np.ndarray:
	if arr.size == 0:
		return arr
	diffs = np.empty_like(arr)
	diffs[0] = arr[0]
	diffs[1:] = arr[1:] - arr[:-1]
	return diffs


def _delta_decode(diffs: np.ndarray) -> np.ndarray:
	if diffs.size == 0:
		return diffs
	out = np.empty_like(diffs)
	out[0] = diffs[0]
	np.cumsum(diffs, out=out)
	return out


def compress_window(arr: np.ndarray) -> bytes:
	"""Compress a 1D float array using delta+zlib.

	The header stores: version (1 byte), dtype code (1 byte), length (uint32),
	and first sample raw float32 for improved stability.
	"""
	if arr.ndim != 1:
		arr = arr.ravel()
	arr_f32 = np.asarray(arr, dtype=np.float32)
	diffs = _delta_encode(arr_f32)
	# Pack header: version=1, dtype=1 (float32), length
	header = struct.pack("<BBI", 1, 1, diffs.size)
	payload = diffs.tobytes()
	comp = zlib.compress(payload, level=6)
	return header + comp


def decompress_window(data: bytes) -> np.ndarray:
	"""Decompress bytes produced by compress_window back to float32 array."""
	if len(data) < 6:
		raise ValueError("compressed payload too small")
	version, dtype_code, length = struct.unpack("<BBI", data[:6])
	if version != 1 or dtype_code != 1:
		raise ValueError("unsupported codec version or dtype")
	raw = zlib.decompress(data[6:])
	diffs = np.frombuffer(raw, dtype=np.float32, count=length)
	return _delta_decode(diffs)


def compression_ratio(arr: np.ndarray) -> float:
	raw_bytes = int(np.asarray(arr).size * np.asarray(arr).dtype.itemsize)
	comp = compress_window(arr)
	return raw_bytes / max(1, len(comp))
