import sys
import time
import threading
from collections import deque
from typing import Dict, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover - running on systems without audio
    sd = None  # type: ignore

try:
    import pyopensmile
except Exception:
    pyopensmile = None  # type: ignore


class RealTimeOpenSmile:
    """Capture audio from the first available microphone and compute scores.

    The analyzer reads samples continuously from the selected device and keeps
    only the most recent ``buffer_seconds`` of audio in a deque.  Every
    ``1 / frame_hz`` seconds the current buffer is passed to openSMILE to obtain
    features and prints scores such as loudness, pitch and voice quality
    (jitter, shimmer, HNR).  The scoring rate is similar to processing video
    frames.

    Parameters
    ----------
    device_name:
        Optional substring to select an audio device.  If ``None`` the first
        available input device is used.
    samplerate:
        Sampling rate of the microphone.
    buffer_seconds:
        Number of seconds to keep in the sliding window.
    frame_hz:
        How many times per second scores are produced.
    """

    def __init__(
        self,
        device_name: Optional[str] = None,
        samplerate: int = 16000,
        buffer_seconds: int = 3,
        frame_hz: int = 10,
    ) -> None:
        if sd is None:
            raise RuntimeError("sounddevice module is required but not installed")
        if pyopensmile is None:
            raise RuntimeError("pyopensmile module is required but not installed")

        self.device = self._find_device(device_name)
        self.samplerate = samplerate
        self.buffer_samples = samplerate * buffer_seconds
        self.buffer: deque[float] = deque(maxlen=self.buffer_samples)
        self.lock = threading.Lock()
        self.frame_interval = 1.0 / frame_hz

        # Configure openSMILE: GeMAPS functionals are lightweight and suitable
        self.smile = pyopensmile.Smile(
            feature_set=pyopensmile.FeatureSet.GeMAPSv01b,
            feature_level=pyopensmile.FeatureLevel.Functionals,
        )

        # Map user friendly names to actual feature keys and keep only
        # available ones
        feature_map = {
            "loudness": "loudness_sma3_amean",
            "pitch": "F0semitoneFrom27.5Hz_sma3nz_amean",
            "jitter": "jitterLocal_sma3nz_amean",
            "shimmer": "shimmerLocaldB_sma3nz_amean",
            "hnr": "HNRdBACF_sma3nz_amean",
        }
        available = set(self.smile.feature_names)
        self.features: Dict[str, str] = {
            name: key for name, key in feature_map.items() if key in available
        }

    # ------------------------------------------------------------------
    def _find_device(self, device_name: Optional[str]) -> Optional[int]:
        """Return index of a microphone matching ``device_name``.

        If ``device_name`` is ``None``, the first input-capable device is
        selected.  ``None`` is returned if no suitable device is found, letting
        :mod:`sounddevice` choose the default.
        """

        devices = sd.query_devices()
        search = (device_name or "").lower()
        for idx, info in enumerate(devices):
            if info["max_input_channels"] <= 0:
                continue
            name = info["name"].lower()
            if search and search in name:
                return idx
            if not search:
                return idx  # first available input device
        return None

    # ------------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:  # pragma: no cover - only relevant on real hardware
            print(status, file=sys.stderr)
        # ``indata`` is (frames, channels); keep mono samples
        with self.lock:
            self.buffer.extend(indata[:, 0])

    # ------------------------------------------------------------------
    def _process_buffer(self) -> None:
        # Avoid processing until buffer is full enough
        with self.lock:
            if len(self.buffer) < self.buffer_samples:
                return
            audio = np.array(self.buffer, dtype=np.float32)
        audio = audio.reshape(-1, 1)
        feats = self.smile.process_signal(audio, self.samplerate)
        series = feats.iloc[0]
        values = {
            name: float(series.get(key, 0.0)) for name, key in self.features.items()
        }
        if values:
            print(" | ".join(f"{n}: {v:.2f}" for n, v in values.items()))

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the streaming analyzer.  Press Ctrl+C to stop."""
        with sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            device=self.device,
            callback=self._audio_callback,
        ):
            next_time = time.monotonic()
            try:
                while True:
                    now = time.monotonic()
                    if now >= next_time:
                        self._process_buffer()
                        next_time += self.frame_interval
                    time.sleep(0.001)  # small sleep to yield CPU
            except KeyboardInterrupt:  # pragma: no cover - interactive only
                print("\nStopping stream...")


if __name__ == "__main__":
    analyzer = RealTimeOpenSmile()
    analyzer.run()
