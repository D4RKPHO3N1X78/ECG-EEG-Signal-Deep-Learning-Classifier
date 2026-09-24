import numpy as np
from scipy.signal import find_peaks, butter, filtfilt


def detect_r_peaks_pan_tompkins(ecg_signal: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Implements the Pan-Tompkins algorithm for R-peak detection in ECG signals:
    1. Bandpass filter (5-15 Hz)
    2. Derivative operator
    3. Squaring function
    4. Moving-window integration
    5. Adaptive peak thresholding
    
    Returns array of R-peak indices.
    """
    if ecg_signal.ndim > 1:
        ecg_signal = ecg_signal[0]
        
    nyquist = 0.5 * fs
    low = 5.0 / nyquist
    high = 15.0 / nyquist
    b, a = butter(1, [low, high], btype='band')
    filtered = filtfilt(b, a, ecg_signal)
    
    # 5-point Derivative operator
    differentiated = np.gradient(filtered)
    
    # Squaring function
    squared = differentiated ** 2
    
    # Moving-window integration (150ms window)
    window_len = int(0.15 * fs)
    kernel = np.ones(window_len) / window_len
    integrated = np.convolve(squared, kernel, mode='same')
    
    # Find local peaks in integrated signal
    min_dist = int(0.4 * fs) # minimum 240ms refractory period between beats (250 bpm max)
    thresh = 0.35 * np.max(integrated)
    peaks, _ = find_peaks(integrated, height=thresh, distance=min_dist)
    
    # Refine R-peak location to exact local maximum in original/filtered ECG
    r_peaks = []
    search_win = int(0.08 * fs) # search +/- 80ms around integrated peak
    for p in peaks:
        start = max(0, p - search_win)
        end = min(len(ecg_signal), p + search_win)
        exact_r = start + np.argmax(ecg_signal[start:end])
        r_peaks.append(exact_r)
        
    return np.array(r_peaks, dtype=int)


def segment_ecg_beats(ecg_signal: np.ndarray, r_peaks: np.ndarray = None, fs: float = 250.0, 
                      before_sec: float = 0.3, after_sec: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Segments single-lead or multi-lead ECG signal into individual heartbeats centered around R-peaks.
    Returns:
        beats: array of shape (n_beats, channels, window_length)
        r_peaks: array of R-peak indices corresponding to each beat
    """
    if r_peaks is None or len(r_peaks) == 0:
        signal_for_peaks = ecg_signal[0] if ecg_signal.ndim == 2 else ecg_signal
        r_peaks = detect_r_peaks_pan_tompkins(signal_for_peaks, fs=fs)
        
    before_samples = int(before_sec * fs)
    after_samples = int(after_sec * fs)
    win_len = before_samples + after_samples
    
    beats = []
    valid_r_peaks = []
    
    if ecg_signal.ndim == 1:
        signal_data = ecg_signal[np.newaxis, :]
    else:
        signal_data = ecg_signal
        
    n_ch, total_len = signal_data.shape
    
    for r in r_peaks:
        start = r - before_samples
        end = r + after_samples
        if start >= 0 and end <= total_len:
            beat = signal_data[:, start:end]
            beats.append(beat)
            valid_r_peaks.append(r)
            
    if len(beats) == 0:
        # Fallback uniform windowing if no peaks found
        num_windows = total_len // win_len
        for i in range(num_windows):
            beats.append(signal_data[:, i*win_len:(i+1)*win_len])
            valid_r_peaks.append(i*win_len + before_samples)
            
    return np.array(beats), np.array(valid_r_peaks)


def segment_eeg_epochs(eeg_signal: np.ndarray, epoch_sec: float = 2.0, overlap: float = 0.5, fs: float = 250.0) -> np.ndarray:
    """
    Segments multi-channel EEG signal into overlapping time epochs.
    eeg_signal: shape (channels, samples)
    Returns:
        epochs: shape (n_epochs, channels, epoch_samples)
    """
    if eeg_signal.ndim == 1:
        eeg_signal = eeg_signal[np.newaxis, :]
        
    n_ch, n_samples = eeg_signal.shape
    win_samples = int(epoch_sec * fs)
    step = int(win_samples * (1.0 - overlap))
    
    epochs = []
    start = 0
    while start + win_samples <= n_samples:
        epoch = eeg_signal[:, start:start + win_samples]
        epochs.append(epoch)
        start += step
        
    return np.array(epochs)
