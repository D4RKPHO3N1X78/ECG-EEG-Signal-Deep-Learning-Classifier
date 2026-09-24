# ECG & EEG Signal Deep Learning Classifier (Apr 2026)

> An end-to-end clinical deep learning system and web studio for the automated classification of ECG (cardiac) and EEG (cognitive/neurological) bio-signals to detect cardiac arrhythmias, ischemic events, and neurological abnormalities.

---

## 🌟 Key Features & Highlights

- **End-to-End Signal Preprocessing Pipeline**:
  - **Bandpass Filtering**: Zero-phase Butterworth filters (0.5–45 Hz for ECG; 0.5–50 Hz for EEG).
  - **Notch Filtering**: IIR Notch Filter for 50 Hz powerline interference removal.
  - **Baseline Wander Removal**: Dual cascading median filters and high-pass filtering.
  - **Denoising & Artifact Attenuation**: Wavelet soft-thresholding (DWT) and FastICA spatial decomposition for ocular/EMG artifact removal.
  - **Pan-Tompkins Algorithm**: Real-time R-peak detector and beat segmentation for ECG signals.
  - **Feature Extraction**: Time & frequency domain HRV metrics (SDNN, RMSSD, pNN50, LF/HF ratio) and EEG Welch Power Spectral Density band powers ($\delta, \theta, \alpha, \beta, \gamma$) & spectral entropy.

- **Deep Learning Model Architectures (PyTorch)**:
  - **ECG Classifier (`ECGResNet1DBiLSTM`)**: 1D ResNet with skip connections + Bidirectional LSTM + Multi-Head Self-Attention for arrhythmia & STEMI classification.
  - **EEG Classifier (`EEGNetTransformer`)**: Spatial-Temporal 2D/1D Convolutions with Depthwise Separable Conv + Transformer Encoder for epileptic seizure and slow-wave encephalopathy detection.
  - **Unified Signal Classifier (`UnifiedSignalClassifier`)**: Multi-modal model execution engine for unified inference.

- **Strict Evaluation Metrics Framework**:
  - **Overall Metrics**: Accuracy, Balanced Accuracy, Macro/Weighted Precision, Macro Recall/Sensitivity, Macro Specificity, Macro F1-Score, ROC-AUC (One-vs-Rest).
  - **Confusion Matrices**: Count and normalized proportion matrix heatmaps.
  - **Per-Class Breakdown**: TP, TN, FP, FN, Precision, Recall, Specificity, F1-score for each condition.

- **Interactive Web Studio & REST API**:
  - **Glassmorphic UI**: Vibrant, responsive Web Dashboard built with HTML5, CSS3, Chart.js, and Vanilla JavaScript.
  - **Signal Inspector**: Real-time comparison of Raw vs. Filtered signal waveforms and FFT Power Spectral Density.
  - **Diagnostic Studio**: Instant model prediction with softmax probability breakdown and downloadable clinical report.
  - **Training Studio**: Trigger model fine-tuning jobs with live loss & accuracy learning curve visualizers.

---

## 🏗️ Repository Architecture

```
ECG & EEG Signal Deep Learning Classifier/
├── data/                       # Datasets & synthetic signal generators
│   ├── generator.py            # Clinical ECG & EEG synthetic signal generator
│   ├── sample_ecg.json         # Pre-loaded ECG arrhythmia test cases
│   └── sample_eeg.json         # Pre-loaded EEG neurological test cases
├── src/                        # Core Python ML Package
│   ├── preprocessing/          # Data Preprocessing Pipeline
│   │   ├── filters.py          # Bandpass, notch, baseline wander removal
│   │   ├── denoiser.py         # Wavelet thresholding & FastICA
│   │   ├── segmentation.py     # Pan-Tompkins R-peak detector & epoching
│   │   └── feature_extractor.py # HRV metrics & EEG PSD band power ratios
│   ├── models/                 # PyTorch Deep Learning Architectures
│   │   ├── ecg_model.py        # 1D-ResNet + Bi-LSTM + Self-Attention
│   │   ├── eeg_model.py        # EEGNet + Transformer Encoder
│   │   └── unified_classifier.py # Unified prediction engine
│   ├── evaluation/             # Strict Evaluation Metrics
│   │   └── metrics.py          # Accuracy, Specificity, F1, ROC-AUC, Confusion Matrix
│   ├── training/               # Training Engine
│   │   └── trainer.py          # PyTorch trainer with Cosine Annealing scheduler
│   └── utils/
│       └── report_generator.py # Clinical diagnostic report generator
├── static/                     # Web Dashboard Assets
│   ├── index.html              # Glassmorphic UI Dashboard
│   ├── css/style.css           # Styling & dark mode glassmorphism
│   └── js/
│       ├── app.js              # Web state management & REST client
│       └── charts.js           # Chart.js waveform & metric renderers
├── tests/                      # Automated Unit Tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── app.py                      # Flask REST API backend server
├── main.py                     # CLI Tool entrypoint
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## ⚡ Quick Start & Usage Guide

### 1. Installation & Environment Setup

```bash
# Clone repository and enter project folder
cd "ECG & EEG Signal Deep Learning Classifier"

# Activate Python Virtual Environment
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests

```bash
PYTHONPATH=. pytest tests/
```

### 3. Launch Web Dashboard Studio

```bash
python3 main.py serve --port 5050
# Open http://localhost:5050 in your browser
```

### 4. CLI Commands

```bash
# Generate synthetic ECG test sample
python3 main.py generate --type ecg --condition "Atrial Fibrillation (AFIB)" --output sample.json

# Run Deep Learning Inference on signal file
python3 main.py predict --input sample.json --type ecg

# Train PyTorch model on synthetic dataset
python3 main.py train --type ecg --epochs 10
```

---

## 📊 Target Diagnostic Classes

| Modality | Diagnostic Class | Clinical Description |
| :--- | :--- | :--- |
| **ECG** | Normal Sinus Rhythm (NSR) | Normal cardiac rhythm (60–100 BPM) with distinct P-Q-R-S-T waves |
| **ECG** | Atrial Fibrillation (AFIB) | Rapid, irregular atrial contraction with absent P-waves |
| **ECG** | Premature Ventricular Contraction (PVC) | Ectopic focus in ventricles causing wide, bizarre QRS complexes |
| **ECG** | Supraventricular Tachycardia (SVT) | Rapid regular heart rate (150–220 BPM) originating above ventricles |
| **ECG** | ST-Elevation Myocardial Infarction (STEMI) | Acute transmural myocardial ischemia with ST elevation |
| **EEG** | Normal EEG (Baseline) | Dominant Alpha rhythm (8–12 Hz) and low amplitude Beta |
| **EEG** | Epileptic Seizure Activity (Ictal) | Synchronized 3 Hz spike-and-wave paroxysmal discharges |
| **EEG** | Focal Spike & Waveform Abnormality | Localized epileptiform sharp waves |
| **EEG** | Slow-Wave Encephalopathy | Pathological generalized Delta (1–3 Hz) slowing |
| **EEG** | Artifact Interference | Powerline 50 Hz noise, ocular blinks, or muscular EMG activity |

---

## 📈 Evaluation Metrics Summary

| Metric | Target Standard | Definition / Formula |
| :--- | :--- | :--- |
| **Accuracy** | $> 94.0\%$ | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| **Balanced Accuracy** | $> 93.0\%$ | $\frac{1}{2} \left( \text{Sensitivity} + \text{Specificity} \right)$ |
| **Macro Sensitivity (Recall)** | $> 93.5\%$ | $\frac{TP}{TP + FN}$ |
| **Macro Specificity** | $> 98.0\%$ | $\frac{TN}{TN + FP}$ |
| **Macro F1-Score** | $> 0.930$ | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| **ROC-AUC (OVR)** | $> 0.970$ | Area under One-vs-Rest Receiver Operating Characteristic Curve |

---

*Developed for Advanced Bio-Signal Deep Learning & Clinical Diagnostics (April 2026).*
