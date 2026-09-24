import numpy as np
from scipy.signal import welch
from scipy.stats import entropy


def extract_ecg_hrv_features(r_peaks: np.ndarray, fs: float = 250.0) -> dict:
    """
    Computes Time-Domain and Frequency-Domain Heart Rate Variability (HRV) metrics.
    """
    if len(r_peaks) < 2:
        return {
            "mean_rr_ms": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50_pct": 0.0,
            "mean_hr_bpm": 0.0,
            "lf_hf_ratio": 0.0
        }
        
    # RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) / fs * 1000.0
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1) if len(rr_intervals) > 1 else 0.0
    
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0
    nn50 = np.sum(np.abs(rr_diff) > 50) if len(rr_diff) > 0 else 0
    pnn50 = (nn50 / len(rr_diff)) * 100.0 if len(rr_diff) > 0 else 0.0
    
    mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
    
    # Frequency domain LF/HF estimate via Welch
    if len(rr_intervals) >= 4:
        freqs, psd = welch(rr_intervals, fs=4.0, nperseg=min(len(rr_intervals), 16))
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        lf_power = np.sum(psd[lf_mask])
        hf_power = np.sum(psd[hf_mask])
        lf_hf_ratio = float(lf_power / hf_power) if hf_power > 0 else 1.0
    else:
        lf_hf_ratio = 1.0
        
    return {
        "mean_rr_ms": round(float(mean_rr), 2),
        "sdnn_ms": round(float(sdnn), 2),
        "rmssd_ms": round(float(rmssd), 2),
        "pnn50_pct": round(float(pnn50), 2),
        "mean_hr_bpm": round(float(mean_hr), 1),
        "lf_hf_ratio": round(float(lf_hf_ratio), 2)
    }


def extract_eeg_spectral_features(eeg_signal: np.ndarray, fs: float = 250.0) -> dict:
    """
    Computes EEG Power Spectral Density (PSD) across standard frequency bands:
    - Delta (0.5 - 4 Hz)
    - Theta (4 - 8 Hz)
    - Alpha (8 - 13 Hz)
    - Beta (13 - 30 Hz)
    - Gamma (30 - 50 Hz)
    Along with Spectral Entropy and Theta/Beta ratio.
    """
    if eeg_signal.ndim == 2:
        # Average across channels or process mean signal
        mean_sig = np.mean(eeg_signal, axis=0)
    else:
        mean_sig = eeg_signal

    freqs, psd = welch(mean_sig, fs=fs, nperseg=min(len(mean_sig), int(fs*2)))
    
    total_power = np.sum(psd) + 1e-8
    
    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.sum(psd[mask])

    delta = band_power(0.5, 4.0)
    theta = band_power(4.0, 8.0)
    alpha = band_power(8.0, 13.0)
    beta = band_power(13.0, 30.0)
    gamma = band_power(30.0, 50.0)
    
    # Normalized spectral entropy
    psd_norm = psd / total_power
    psd_norm = psd_norm[psd_norm > 0]
    spec_entropy = entropy(psd_norm) / np.log(len(psd_norm)) if len(psd_norm) > 1 else 0.0
    
    theta_beta_ratio = float(theta / beta) if beta > 0 else 1.0

    return {
        "delta_abs": float(delta),
        "theta_abs": float(theta),
        "alpha_abs": float(alpha),
        "beta_abs": float(beta),
        "gamma_abs": float(gamma),
        "delta_rel": round(float(delta / total_power), 4),
        "theta_rel": round(float(theta / total_power), 4),
        "alpha_rel": round(float(alpha / total_power), 4),
        "beta_rel": round(float(beta / total_power), 4),
        "gamma_rel": round(float(gamma / total_power), 4),
        "spectral_entropy": round(float(spec_entropy), 4),
        "theta_beta_ratio": round(float(theta_beta_ratio), 3)
    }
