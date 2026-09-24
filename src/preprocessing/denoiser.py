import numpy as np
from sklearn.decomposition import FastICA


def wavelet_denoise_signal(data: np.ndarray, threshold_mult: float = 0.5) -> np.ndarray:
    """
    Applies a discrete signal smoothing/thresholding algorithm to attenuate high-frequency noise 
    and motion artifacts while preserving physiological transient morphology (e.g. QRS complex or EEG spikes).
    """
    if data.ndim == 1:
        # Stationary moving window thresholding filter
        win_size = 5
        kernel = np.ones(win_size) / win_size
        smooth = np.convolve(data, kernel, mode='same')
        residual = data - smooth
        sigma = np.median(np.abs(residual)) / 0.6745
        thresh = sigma * threshold_mult * np.sqrt(2 * np.log(len(data)))
        denoised_res = np.where(np.abs(residual) > thresh, residual, 0.5 * residual)
        return smooth + denoised_res
    elif data.ndim == 2:
        output = np.zeros_like(data)
        for ch in range(data.shape[0]):
            output[ch] = wavelet_denoise_signal(data[ch], threshold_mult)
        return output
    return data


def perform_ica_clean(eeg_channels: np.ndarray, n_components: int = None) -> np.ndarray:
    """
    Uses FastICA on multi-channel EEG data to separate ocular/muscle artifacts from brain activity.
    eeg_channels: shape (n_channels, n_samples)
    """
    if eeg_channels.ndim != 2 or eeg_channels.shape[0] < 2:
        return eeg_channels
    
    n_ch, n_samples = eeg_channels.shape
    n_comp = n_components if n_components else min(n_ch, 4)
    
    try:
        ica = FastICA(n_components=n_comp, random_state=42, max_iter=200)
        # FastICA expects (n_samples, n_channels)
        sources = ica.fit_transform(eeg_channels.T)
        
        # Identify high-variance artifact components (e.g., blink artifacts with extreme kurtosis)
        kurtosis = np.mean((sources - np.mean(sources, axis=0))**4, axis=0) / (np.var(sources, axis=0)**2 + 1e-8)
        # Suppress components with abnormally high kurtosis (> 5.0)
        artifact_indices = np.where(kurtosis > 5.0)[0]
        sources[:, artifact_indices] *= 0.1
        
        reconstructed = ica.inverse_transform(sources).T
        return reconstructed
    except Exception:
        # Fallback if ICA fails to converge
        return eeg_channels
