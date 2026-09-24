import numpy as np


def generate_synthetic_ecg(
    condition: str = "Normal Sinus Rhythm (NSR)",
    fs: float = 250.0,
    duration_sec: float = 5.0,
    noise_level: float = 0.05
) -> np.ndarray:
    """
    Synthesizes realistic 1-lead ECG signals matching specified cardiac condition.
    """
    n_samples = int(fs * duration_sec)
    t = np.linspace(0, duration_sec, n_samples)
    ecg = np.zeros(n_samples)

    if "Atrial Fibrillation" in condition:
        # Irregular heart rate (90 - 140 BPM)
        bpm = 110
        rr_std = 0.18 # high RR variability
        has_p_wave = False
        st_elev = 0.0
        pvc_burst = False
    elif "Premature Ventricular" in condition:
        bpm = 75
        rr_std = 0.05
        has_p_wave = True
        st_elev = 0.0
        pvc_burst = True
    elif "Supraventricular Tachycardia" in condition:
        bpm = 175
        rr_std = 0.02
        has_p_wave = False
        st_elev = 0.0
        pvc_burst = False
    elif "ST-Elevation" in condition:
        bpm = 80
        rr_std = 0.03
        has_p_wave = True
        st_elev = 0.4 # ST elevation
        pvc_burst = False
    else:
        # Normal Sinus Rhythm
        bpm = 70
        rr_std = 0.03
        has_p_wave = True
        st_elev = 0.0
        pvc_burst = False

    mean_rr = 60.0 / bpm
    current_t = 0.1
    beat_count = 0

    while current_t < duration_sec - 0.2:
        rr = max(0.3, mean_rr + np.random.normal(0, rr_std))
        beat_t = current_t
        is_pvc = pvc_burst and (beat_count % 3 == 2)
        
        # Synthesize single beat
        for i, time_val in enumerate(t):
            dt = time_val - beat_t
            if -0.2 < dt < 0.5:
                if is_pvc:
                    # Wide, aberrant PVC wave
                    pvc_wave = 1.2 * np.exp(-((dt - 0.05) ** 2) / 0.003) - 0.8 * np.exp(-((dt - 0.15) ** 2) / 0.004)
                    ecg[i] += pvc_wave
                else:
                    # Normal P-Q-R-S-T morphology
                    p_wave = 0.15 * np.exp(-((dt + 0.12) ** 2) / 0.0008) if has_p_wave else 0.03 * np.sin(2 * np.pi * 6 * dt)
                    q_wave = -0.15 * np.exp(-((dt + 0.02) ** 2) / 0.0002)
                    r_wave = 1.0 * np.exp(-((dt) ** 2) / 0.0003)
                    s_wave = -0.25 * np.exp(-((dt - 0.03) ** 2) / 0.0003)
                    t_wave = (0.25 + st_elev) * np.exp(-((dt - 0.22) ** 2) / 0.004)
                    
                    # ST segment elevation baseline offset
                    st_offset = st_elev if 0.03 <= dt <= 0.25 else 0.0
                    
                    ecg[i] += p_wave + q_wave + r_wave + s_wave + t_wave + st_offset

        current_t += rr if not is_pvc else rr * 0.7 # premature beat
        beat_count += 1

    # Add Gaussian thermal noise & powerline hum
    noise = np.random.normal(0, noise_level, n_samples)
    powerline = 0.03 * np.sin(2 * np.pi * 50 * t)
    ecg += noise + powerline
    return ecg


def generate_synthetic_eeg(
    condition: str = "Normal EEG (Baseline Alpha/Beta)",
    channels: int = 8,
    fs: float = 250.0,
    duration_sec: float = 4.0
) -> np.ndarray:
    """
    Synthesizes multi-channel EEG signals matching specified neurological condition.
    Returns array of shape (channels, samples).
    """
    n_samples = int(fs * duration_sec)
    t = np.linspace(0, duration_sec, n_samples)
    eeg = np.zeros((channels, n_samples))

    for ch in range(channels):
        ch_phase = np.random.uniform(0, 2 * np.pi)
        
        # Base background rhythms
        delta = 0.3 * np.sin(2 * np.pi * 2.5 * t + ch_phase)
        theta = 0.4 * np.sin(2 * np.pi * 6.0 * t + ch_phase)
        alpha = 0.8 * np.sin(2 * np.pi * 10.5 * t + ch_phase)
        beta = 0.3 * np.sin(2 * np.pi * 20.0 * t + ch_phase)
        gamma = 0.1 * np.sin(2 * np.pi * 40.0 * t + ch_phase)
        
        if "Seizure" in condition:
            # 3Hz Spike-and-Wave discharge
            spike_wave = 2.5 * (np.sin(2 * np.pi * 3.0 * t) + 0.6 * np.sin(2 * np.pi * 18.0 * t))
            eeg[ch] = delta + 0.3 * alpha + spike_wave
        elif "Focal Spike" in condition:
            # Periodic localized sharp waves
            focal_spikes = np.zeros(n_samples)
            spike_times = [0.8, 2.0, 3.2]
            for st in spike_times:
                mask = np.abs(t - st) < 0.05
                focal_spikes[mask] += 3.0 * np.sin(2 * np.pi * 25 * (t[mask] - st))
            eeg[ch] = alpha + beta + focal_spikes
        elif "Slow-Wave Encephalopathy" in condition:
            # Dominant high-amplitude Delta/Theta
            heavy_delta = 2.2 * np.sin(2 * np.pi * 1.5 * t + ch_phase)
            heavy_theta = 1.5 * np.sin(2 * np.pi * 4.5 * t + ch_phase)
            eeg[ch] = heavy_delta + heavy_theta + 0.1 * beta
        elif "Artifact Interference" in condition:
            # 50Hz line noise + muscular bursts
            powerline = 1.8 * np.sin(2 * np.pi * 50.0 * t)
            muscle_burst = np.where((t > 1.5) & (t < 2.5), 1.5 * np.random.randn(n_samples), 0)
            eeg[ch] = alpha + beta + powerline + muscle_burst
        else:
            # Normal EEG (Alpha dominant)
            eeg[ch] = 0.2 * delta + 0.2 * theta + alpha + beta + 0.1 * gamma
            
        eeg[ch] += np.random.normal(0, 0.1, n_samples)

    return eeg
