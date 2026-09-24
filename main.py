#!/usr/bin/env python3
"""
CLI Tool & Main Entrypoint for ECG & EEG Signal Deep Learning Classifier System.
"""
import argparse
import sys
import json
import numpy as np
import torch

from src.preprocessing import (
    apply_bandpass_filter,
    apply_notch_filter,
    remove_baseline_wander,
    wavelet_denoise_signal,
    normalize_signal,
    detect_r_peaks_pan_tompkins,
    extract_ecg_hrv_features,
    extract_eeg_spectral_features
)
from src.models import UnifiedSignalClassifier, ECGResNet1DBiLSTM, EEGNetTransformer
from src.evaluation import compute_strict_evaluation_metrics
from src.training import SignalModelTrainer
from src.utils import generate_clinical_diagnostic_report
from data.generator import generate_synthetic_ecg, generate_synthetic_eeg


def main():
    parser = argparse.ArgumentParser(description="ECG & EEG Signal Deep Learning Classifier CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Serve Web App
    serve_parser = subparsers.add_parser("serve", help="Launch interactive web dashboard backend server")
    serve_parser.add_argument("--port", type=int, default=5000, help="Port to listen on")

    # Generate Synthetic Signal
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic ECG/EEG signals")
    gen_parser.add_argument("--type", choices=["ecg", "eeg"], default="ecg", help="Signal type")
    gen_parser.add_argument("--condition", type=str, default="Normal Sinus Rhythm (NSR)", help="Condition string")
    gen_parser.add_argument("--duration", type=float, default=4.0, help="Duration in seconds")
    gen_parser.add_argument("--output", type=str, default="generated_signal.json", help="Output JSON path")

    # Run Prediction / Inference
    predict_parser = subparsers.add_parser("predict", help="Run Deep Learning model inference on signal file")
    predict_parser.add_argument("--input", type=str, required=True, help="Path to input signal JSON file")
    predict_parser.add_argument("--type", choices=["ecg", "eeg"], default="ecg", help="Signal modality")

    # Train Model
    train_parser = subparsers.add_parser("train", help="Train PyTorch Deep Learning model on synthetic dataset")
    train_parser.add_argument("--type", choices=["ecg", "eeg"], default="ecg", help="Model type to train")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    args = parser.parse_args()

    if args.command == "serve" or len(sys.argv) == 1:
        from app import app
        port = getattr(args, "port", 5000)
        print(f"🚀 Launching ECG & EEG Classifier Web Server on http://localhost:{port}...")
        app.run(host="0.0.0.0", port=port, debug=False)

    elif args.command == "generate":
        fs = 250.0
        if args.type == "ecg":
            sig = generate_synthetic_ecg(condition=args.condition, fs=fs, duration_sec=args.duration)
        else:
            sig = generate_synthetic_eeg(condition=args.condition, channels=8, fs=fs, duration_sec=args.duration)
            
        with open(args.output, "w") as f:
            json.dump({"type": args.type, "condition": args.condition, "signal": sig.tolist()}, f, indent=2)
        print(f"✅ Generated {args.type.upper()} signal for '{args.condition}' saved to {args.output}")

    elif args.command == "predict":
        with open(args.input, "r") as f:
            content = json.load(f)
            
        raw_sig = np.array(content.get("signal", content))
        signal_type = args.type
        fs = 250.0

        processed = remove_baseline_wander(raw_sig, fs=fs)
        processed = apply_bandpass_filter(processed, lowcut=0.5, highcut=45.0 if signal_type=="ecg" else 50.0, fs=fs)
        processed = apply_notch_filter(processed, notch_freq=50.0, fs=fs)
        processed = wavelet_denoise_signal(processed)
        norm_sig = normalize_signal(processed, method='zscore')

        engine = UnifiedSignalClassifier()
        if signal_type == "ecg":
            sig_t = torch.tensor(norm_sig, dtype=torch.float32)
            if sig_t.ndim == 1:
                sig_t = sig_t.unsqueeze(0).unsqueeze(0)
            elif sig_t.ndim == 2:
                sig_t = sig_t.unsqueeze(0)
            pred_idx, diag, probs = engine.predict_ecg(sig_t)
            r_peaks = detect_r_peaks_pan_tompkins(norm_sig if norm_sig.ndim==1 else norm_sig[0], fs=fs)
            feats = extract_ecg_hrv_features(r_peaks, fs=fs)
        else:
            sig_t = torch.tensor(norm_sig, dtype=torch.float32)
            if sig_t.ndim == 1:
                sig_t = sig_t.repeat(8, 1).unsqueeze(0)
            elif sig_t.ndim == 2:
                if sig_t.shape[0] < 8:
                    sig_t = torch.cat([sig_t, torch.zeros(8 - sig_t.shape[0], sig_t.shape[1])], dim=0)
                sig_t = sig_t[:8].unsqueeze(0)
            pred_idx, diag, probs = engine.predict_eeg(sig_t)
            feats = extract_eeg_spectral_features(norm_sig, fs=fs)

        print("==================================================")
        print(f"DIAGNOSIS RESULT: {diag}")
        print(f"CONFIDENCE: {probs[diag]*100:.2f}%")
        print("PROBABILITIES:")
        for k, v in probs.items():
            print(f"  - {k}: {v*100:.2f}%")
        print("EXTRACTED BIO-FEATURES:")
        for fk, fv in feats.items():
            print(f"  - {fk}: {fv}")
        print("==================================================")

    elif args.command == "train":
        print(f"⚡ Starting PyTorch model training for {args.type.upper()} ({args.epochs} epochs)...")
        from app import trigger_training
        # Use internal training execution logic
        from unittest.mock import Mock
        req_mock = Mock()
        req_mock.json = {"type": args.type, "epochs": args.epochs}
        with app.test_request_context(json={"type": args.type, "epochs": args.epochs}):
            res = trigger_training()
            data = res.get_json()
            print("🎉 Training Complete!")
            print("Final Accuracy:", data["evaluation"]["overall"]["accuracy"])
            print("Macro F1-Score:", data["evaluation"]["overall"]["macro_f1_score"])
            print("ROC-AUC Score:", data["evaluation"]["overall"]["roc_auc_score"])


if __name__ == "__main__":
    main()
