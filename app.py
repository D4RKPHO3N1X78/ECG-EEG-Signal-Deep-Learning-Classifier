import os
import json
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.preprocessing.filters import apply_bandpass_filter, apply_notch_filter, remove_baseline_wander, normalize_signal
from src.preprocessing.denoiser import wavelet_denoise_signal, perform_ica_clean
from src.preprocessing.segmentation import detect_r_peaks_pan_tompkins, segment_ecg_beats, segment_eeg_epochs
from src.preprocessing.feature_extractor import extract_ecg_hrv_features, extract_eeg_spectral_features
from src.models.ecg_model import ECGResNet1DBiLSTM
from src.models.eeg_model import EEGNetTransformer
from src.models.unified_classifier import UnifiedSignalClassifier
from src.evaluation.metrics import compute_strict_evaluation_metrics
from src.training.trainer import SignalModelTrainer
from src.utils.report_generator import generate_clinical_diagnostic_report
from data.generator import generate_synthetic_ecg, generate_synthetic_eeg

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Initialize Unified Model Classifier Engine
classifier_engine = UnifiedSignalClassifier(ecg_channels=1, eeg_channels=8)
device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "online",
        "system": "ECG & EEG Signal Deep Learning Classifier",
        "version": "1.0.0",
        "device": device_str,
        "ecg_classes": ECGResNet1DBiLSTM.ECG_CLASSES,
        "eeg_classes": EEGNetTransformer.EEG_CLASSES
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    sample_ecg_path = os.path.join("data", "sample_ecg.json")
    sample_eeg_path = os.path.join("data", "sample_eeg.json")
    
    ecg_samples = {}
    eeg_samples = {}
    
    if os.path.exists(sample_ecg_path):
        with open(sample_ecg_path, "r") as f:
            ecg_samples = json.load(f)
            
    if os.path.exists(sample_eeg_path):
        with open(sample_eeg_path, "r") as f:
            eeg_samples = json.load(f)
            
    return jsonify({
        "ecg": ecg_samples,
        "eeg": eeg_samples
    })


@app.route("/api/preprocess", methods=["POST"])
def preprocess_signal():
    data = request.json or {}
    raw_signal = np.array(data.get("signal", []))
    signal_type = data.get("type", "ecg").lower()
    fs = float(data.get("fs", 250.0))
    
    enable_bandpass = data.get("bandpass", True)
    enable_notch = data.get("notch", True)
    enable_baseline = data.get("baseline", True)
    enable_denoise = data.get("denoise", True)
    
    if len(raw_signal) == 0:
        return jsonify({"error": "Empty signal provided"}), 400

    processed = raw_signal.copy()
    
    # 1. Baseline Wander Removal
    if enable_baseline:
        processed = remove_baseline_wander(processed, fs=fs)
        
    # 2. Bandpass Filtering
    if enable_bandpass:
        lowcut = 0.5
        highcut = 45.0 if signal_type == "ecg" else 50.0
        processed = apply_bandpass_filter(processed, lowcut=lowcut, highcut=highcut, fs=fs)

    # 3. Notch Filtering (50 Hz powerline)
    if enable_notch:
        processed = apply_notch_filter(processed, notch_freq=50.0, fs=fs)

    # 4. Wavelet / ICA Denoising
    if enable_denoise:
        if signal_type == "eeg" and processed.ndim == 2:
            processed = perform_ica_clean(processed)
        else:
            processed = wavelet_denoise_signal(processed)

    # Compute FFT Power Spectrum for visualization
    if processed.ndim == 1:
        fft_vals = np.abs(np.fft.rfft(processed))
        fft_freqs = np.fft.rfftfreq(len(processed), 1/fs)
    else:
        fft_vals = np.abs(np.fft.rfft(processed[0]))
        fft_freqs = np.fft.rfftfreq(processed.shape[1], 1/fs)

    return jsonify({
        "raw": raw_signal.tolist(),
        "processed": processed.tolist(),
        "fft_freqs": fft_freqs[:80].tolist(),
        "fft_vals": fft_vals[:80].tolist()
    })


@app.route("/api/predict", methods=["POST"])
def predict_signal():
    data = request.json or {}
    raw_signal = np.array(data.get("signal", []))
    signal_type = data.get("type", "ecg").lower()
    fs = float(data.get("fs", 250.0))
    patient_id = data.get("patient_id", "PT-99482")

    if len(raw_signal) == 0:
        return jsonify({"error": "No signal provided"}), 400

    # Apply standard preprocessing
    processed = remove_baseline_wander(raw_signal, fs=fs)
    lowcut = 0.5
    highcut = 45.0 if signal_type == "ecg" else 50.0
    processed = apply_bandpass_filter(processed, lowcut=lowcut, highcut=highcut, fs=fs)
    processed = apply_notch_filter(processed, notch_freq=50.0, fs=fs)
    processed = wavelet_denoise_signal(processed)
    norm_sig = normalize_signal(processed, method="zscore")

    if signal_type == "ecg":
        sig_tensor = torch.tensor(norm_sig, dtype=torch.float32)
        if sig_tensor.ndim == 1:
            sig_tensor = sig_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, L)
        elif sig_tensor.ndim == 2:
            sig_tensor = sig_tensor.unsqueeze(0) # (1, C, L)

        pred_idx, class_name, prob_dict = classifier_engine.predict_ecg(sig_tensor)
        
        # Extract R-peaks & HRV metrics
        peaks = detect_r_peaks_pan_tompkins(norm_sig if norm_sig.ndim==1 else norm_sig[0], fs=fs)
        features = extract_ecg_hrv_features(peaks, fs=fs)
    else:
        sig_tensor = torch.tensor(norm_sig, dtype=torch.float32)
        if sig_tensor.ndim == 1:
            # Duplicate to 8 channels for single-lead input fallback
            sig_tensor = sig_tensor.repeat(8, 1).unsqueeze(0)
        elif sig_tensor.ndim == 2:
            if sig_tensor.shape[0] < 8:
                pad_ch = 8 - sig_tensor.shape[0]
                padding = torch.zeros(pad_ch, sig_tensor.shape[1])
                sig_tensor = torch.cat([sig_tensor, padding], dim=0)
            sig_tensor = sig_tensor[:8].unsqueeze(0)

        pred_idx, class_name, prob_dict = classifier_engine.predict_eeg(sig_tensor)
        features = extract_eeg_spectral_features(norm_sig, fs=fs)

    confidence = prob_dict[class_name]
    
    report_md = generate_clinical_diagnostic_report(
        patient_id=patient_id,
        signal_type=signal_type,
        diagnosis=class_name,
        confidence=confidence,
        probabilities=prob_dict,
        features=features
    )

    return jsonify({
        "patient_id": patient_id,
        "signal_type": signal_type,
        "predicted_class_index": pred_idx,
        "diagnosis": class_name,
        "confidence": confidence,
        "probabilities": prob_dict,
        "features": features,
        "processed_signal": norm_sig.tolist(),
        "report_markdown": report_md
    })


@app.route("/api/train", methods=["POST"])
def trigger_training():
    data = request.json or {}
    signal_type = data.get("type", "ecg").lower()
    epochs = int(data.get("epochs", 10))

    fs = 250.0
    if signal_type == "ecg":
        class_names = ECGResNet1DBiLSTM.ECG_CLASSES
        num_samples_per_class = 20
        seq_len = 1000
        X_list, y_list = [], []
        
        for idx, cls in enumerate(class_names):
            for _ in range(num_samples_per_class):
                sig = generate_synthetic_ecg(condition=cls, fs=fs, duration_sec=4.0)
                norm_sig = normalize_signal(sig, method='zscore')
                X_list.append(norm_sig[np.newaxis, :])
                y_list.append(idx)
                
        X = np.array(X_list) # (N, 1, 1000)
        y = np.array(y_list)
        
        # Split train/val 80/20
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        split = int(0.8 * len(y))
        train_idx, val_idx = indices[:split], indices[split:]
        
        trainer = SignalModelTrainer(model=classifier_engine.ecg_model, lr=1e-3)
        results = trainer.train_epochs(
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_val=X[val_idx],
            y_val=y[val_idx],
            epochs=epochs,
            batch_size=16,
            class_names=class_names
        )
    else:
        class_names = EEGNetTransformer.EEG_CLASSES
        num_samples_per_class = 20
        seq_len = 500
        X_list, y_list = [], []
        
        for idx, cls in enumerate(class_names):
            for _ in range(num_samples_per_class):
                sig = generate_synthetic_eeg(condition=cls, channels=8, fs=fs, duration_sec=2.0)
                norm_sig = normalize_signal(sig, method='zscore')
                X_list.append(norm_sig)
                y_list.append(idx)
                
        X = np.array(X_list) # (N, 8, 500)
        y = np.array(y_list)
        
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        split = int(0.8 * len(y))
        train_idx, val_idx = indices[:split], indices[split:]
        
        trainer = SignalModelTrainer(model=classifier_engine.eeg_model, lr=1e-3)
        results = trainer.train_epochs(
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_val=X[val_idx],
            y_val=y[val_idx],
            epochs=epochs,
            batch_size=16,
            class_names=class_names
        )

    return jsonify({
        "status": "success",
        "signal_type": signal_type,
        "epochs": epochs,
        "history": results["history"],
        "evaluation": results["final_val_metrics"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)

