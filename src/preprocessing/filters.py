import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt


def apply_bandpass_filter(data: np.ndarray, lowcut: float = 0.5, highcut: float = 45.0, fs: float = 250.0, order: int = 4) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter to ECG/EEG signal data.
    
    Parameters:
        data: 1D or 2D array (channels x samples) of signal values.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.
    """
    nyquist = 0.5 * fs
    low = max(0.01, lowcut / nyquist)
    high = min(0.99, highcut / nyquist)
    b, a = butter(order, [low, high], btype='band')
    
    if data.ndim == 1:
        return filtfilt(b, a, data)
    elif data.ndim == 2:
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
        return filtered
    else:
        raise ValueError("Signal data must be 1D or 2D array.")


def apply_notch_filter(data: np.ndarray, notch_freq: float = 50.0, fs: float = 250.0, Q: float = 30.0) -> np.ndarray:
    """
    Applies an IIR notch filter to eliminate powerline interference (50Hz or 60Hz).
    """
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, Q)
    
    if data.ndim == 1:
        return filtfilt(b, a, data)
    elif data.ndim == 2:
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
        return filtered
    else:
        raise ValueError("Signal data must be 1D or 2D array.")


def remove_baseline_wander(data: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Removes low-frequency baseline wander using a high-pass Butterworth filter 
    and cascading median filters.
    """
    if data.ndim == 1:
        # 200ms median filter followed by 600ms median filter for baseline estimate
        kernel1 = int(fs * 0.2) | 1  # ensure odd size
        kernel2 = int(fs * 0.6) | 1
        baseline = medfilt(data, kernel1)
        baseline = medfilt(baseline, kernel2)
        return data - baseline
    elif data.ndim == 2:
        cleaned = np.zeros_like(data)
        kernel1 = int(fs * 0.2) | 1
        kernel2 = int(fs * 0.6) | 1
        for i in range(data.shape[0]):
            base = medfilt(data[i], kernel1)
            base = medfilt(base, kernel2)
            cleaned[i] = data[i] - base
        return cleaned
    return data


def normalize_signal(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalizes signal amplitude using Z-Score or Min-Max scaling.
    """
    if method == 'zscore':
        std = np.std(data, axis=-1, keepdims=True)
        std = np.where(std == 0, 1e-8, std)
        mean = np.mean(data, axis=-1, keepdims=True)
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.min(data, axis=-1, keepdims=True)
        max_val = np.max(data, axis=-1, keepdims=True)
        diff = max_val - min_val
        diff = np.where(diff == 0, 1e-8, diff)
        return (data - min_val) / diff
    else:
        raise ValueError(f"Unknown normalization method: {method}")
