import numpy as np
import pytest

from src.config import DEFAULT_CONFIG
from src.preprocessing.signal_cleaner import (
    butterworth_bandpass,
    clean_vep_signal,
    notch_filter,
    resample_to_length,
    zscore_normalize,
)


def _power_at(signal: np.ndarray, fs: float, freq: float) -> float:
    freqs = np.fft.rfftfreq(len(signal), 1 / fs)
    spectrum = np.abs(np.fft.rfft(signal))
    idx = np.argmin(np.abs(freqs - freq))
    return float(spectrum[idx])


class TestButterworthBandpass:
    def test_output_shape_preserved(self, synthetic_vep_array):
        signal, fs = synthetic_vep_array
        out = butterworth_bandpass(signal, fs)
        assert out.shape == signal.shape

    def test_attenuates_low_frequency_drift(self, rng):
        # A 10s window gives 0.1Hz FFT bin resolution, so the drift tone is
        # resolved distinctly from DC instead of aliasing into the same bin.
        fs = 250.0
        t = np.arange(0, 10.0, 1 / fs)
        drift = 0.5 * np.sin(2 * np.pi * 0.1 * t)  # well below 1Hz lowcut
        content = np.sin(2 * np.pi * 20 * t)
        signal = drift + content
        out = butterworth_bandpass(signal, fs, lowcut=1.0, highcut=100.0)
        assert _power_at(out, fs, 0.1) < 0.3 * _power_at(signal, fs, 0.1)

    def test_preserves_inband_content(self, rng):
        fs = 250.0
        t = np.arange(0, 2.0, 1 / fs)
        signal = np.sin(2 * np.pi * 20 * t)
        out = butterworth_bandpass(signal, fs)
        assert _power_at(out, fs, 20) > 0.8 * _power_at(signal, fs, 20)

    def test_rejects_highcut_above_nyquist(self, synthetic_vep_array):
        signal, fs = synthetic_vep_array
        with pytest.raises(ValueError, match="Nyquist"):
            butterworth_bandpass(signal, fs, lowcut=1.0, highcut=200.0)

    def test_rejects_lowcut_above_highcut(self, synthetic_vep_array):
        signal, fs = synthetic_vep_array
        with pytest.raises(ValueError, match="lowcut must be < highcut"):
            butterworth_bandpass(signal, fs, lowcut=100.0, highcut=10.0)

    def test_rejects_too_short_signal(self):
        with pytest.raises(ValueError, match="too short"):
            butterworth_bandpass(np.ones(5), fs=250.0)

    def test_rejects_non_1d_input(self):
        with pytest.raises(ValueError, match="1D"):
            butterworth_bandpass(np.ones((10, 2)) * 1.0, fs=250.0)

    def test_rejects_nan_input(self):
        sig = np.ones(500)
        sig[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            butterworth_bandpass(sig, fs=250.0)


class TestNotchFilter:
    def test_attenuates_mains_hum(self, synthetic_vep_array):
        signal, fs = synthetic_vep_array
        out = notch_filter(signal, fs, freq=50.0)
        before = _power_at(signal, fs, 50.0)
        after = _power_at(out, fs, 50.0)
        assert after < 0.5 * before

    def test_rejects_freq_above_nyquist(self, synthetic_vep_array):
        signal, fs = synthetic_vep_array
        with pytest.raises(ValueError, match="Nyquist"):
            notch_filter(signal, fs, freq=200.0)


class TestZscoreNormalize:
    def test_mean_zero_std_one(self, synthetic_vep_array):
        signal, _ = synthetic_vep_array
        out = zscore_normalize(signal)
        assert out.mean() == pytest.approx(0.0, abs=1e-6)
        assert out.std() == pytest.approx(1.0, abs=1e-6)

    def test_constant_signal_does_not_explode(self):
        out = zscore_normalize(np.full(100, 5.0))
        assert np.all(np.isfinite(out))


class TestResampleToLength:
    def test_upsample_length(self):
        sig = np.linspace(0, 1, 100)
        out = resample_to_length(sig, 250)
        assert out.shape == (250,)

    def test_downsample_length(self):
        sig = np.linspace(0, 1, 500)
        out = resample_to_length(sig, 100)
        assert out.shape == (100,)

    def test_identity_when_same_length(self):
        sig = np.linspace(0, 1, 500)
        out = resample_to_length(sig, 500)
        np.testing.assert_allclose(out, sig)


class TestCleanVepSignal:
    def test_output_shape_dtype(self, synthetic_vep_array):
        signal, fs = synthetic_vep_array
        out = clean_vep_signal(signal, fs)
        assert out.shape == (DEFAULT_CONFIG.vep.length,)
        assert out.dtype == np.float32

    def test_output_is_finite_and_normalized(self, synthetic_vep_array):
        signal, fs = synthetic_vep_array
        out = clean_vep_signal(signal, fs)
        assert np.all(np.isfinite(out))
        assert out.mean() == pytest.approx(0.0, abs=1e-3)

    def test_does_not_shift_p100_like_peak_latency(self, rng):
        # A sharp Gaussian bump at 100ms should still peak near 100ms after
        # zero-phase filtering (group-delay-free filters must not shift it).
        fs = 250.0
        duration = 2.0
        t = np.arange(0, duration, 1 / fs)
        latency = 0.4  # seconds, well within the 1-100Hz passband's dynamics
        bump = 5.0 * np.exp(-((t - latency) ** 2) / (2 * 0.01**2))
        noise = 0.05 * rng.standard_normal(len(t))
        signal = bump + noise + 2.0

        cleaned = clean_vep_signal(signal, fs)
        cleaned_t = np.linspace(0, duration, len(cleaned))
        peak_time = cleaned_t[np.argmax(cleaned)]
        assert peak_time == pytest.approx(latency, abs=0.02)
