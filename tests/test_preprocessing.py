import numpy as np
import pytest
from src.preprocessing.filters import apply_bandpass_filter, apply_notch_filter, remove_baseline_wander, normalize_signal
from src.preprocessing.segmentation import detect_r_peaks_pan_tompkins, segment_ecg_beats, segment_eeg_epochs
from src.preprocessing.feature_extractor import extract_ecg_hrv_features, extract_eeg_spectral_features


def test_filters():
    fs = 250.0
    t = np.linspace(0, 2, int(2 * fs))
    # 10Hz signal + 50Hz noise + 0.1Hz baseline wander
    sig = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t) + 2.0 * np.sin(2 * np.pi * 0.1 * t)
    
    filtered_bp = apply_bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=fs)
    assert filtered_bp.shape == sig.shape
    
    filtered_notch = apply_notch_filter(sig, notch_freq=50.0, fs=fs)
    assert filtered_notch.shape == sig.shape
    
    no_base = remove_baseline_wander(sig, fs=fs)
    assert no_base.shape == sig.shape
    
    norm = normalize_signal(sig, method='zscore')
    assert np.isclose(np.mean(norm), 0.0, atol=1e-2)
    assert np.isclose(np.std(norm), 1.0, atol=1e-2)


def test_segmentation():
    fs = 250.0
    from data.generator import generate_synthetic_ecg, generate_synthetic_eeg
    
    ecg = generate_synthetic_ecg(condition="Normal Sinus Rhythm (NSR)", fs=fs, duration_sec=4.0)
    r_peaks = detect_r_peaks_pan_tompkins(ecg, fs=fs)
    assert len(r_peaks) >= 2
    
    beats, valid_peaks = segment_ecg_beats(ecg, r_peaks, fs=fs)
    assert len(beats) > 0
    assert beats.ndim == 3 # (n_beats, channels, samples)
    
    eeg = generate_synthetic_eeg(condition="Normal EEG (Baseline Alpha/Beta)", channels=8, fs=fs, duration_sec=4.0)
    epochs = segment_eeg_epochs(eeg, epoch_sec=2.0, overlap=0.5, fs=fs)
    assert epochs.shape[0] >= 2
    assert epochs.shape[1] == 8


def test_feature_extractor():
    r_peaks = np.array([250, 500, 750, 1000, 1250])
    hrv = extract_ecg_hrv_features(r_peaks, fs=250.0)
    assert "mean_rr_ms" in hrv
    assert "mean_hr_bpm" in hrv
    assert np.isclose(hrv["mean_hr_bpm"], 60.0, atol=1.0)
    
    from data.generator import generate_synthetic_eeg
    eeg = generate_synthetic_eeg("Normal EEG (Baseline Alpha/Beta)", channels=4, fs=250.0, duration_sec=2.0)
    spectral = extract_eeg_spectral_features(eeg, fs=250.0)
    assert "alpha_rel" in spectral
    assert "spectral_entropy" in spectral
