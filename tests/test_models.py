import torch
from src.models.ecg_model import ECGResNet1DBiLSTM
from src.models.eeg_model import EEGNetTransformer
from src.models.unified_classifier import UnifiedSignalClassifier


def test_ecg_model():
    model = ECGResNet1DBiLSTM(in_channels=1, num_classes=5)
    x = torch.randn(4, 1, 1000) # Batch size 4, 1 channel, 1000 samples
    logits = model(x)
    assert logits.shape == (4, 5)


def test_eeg_model():
    model = EEGNetTransformer(num_channels=8, num_classes=5)
    x = torch.randn(4, 8, 500) # Batch size 4, 8 channels, 500 samples
    logits = model(x)
    assert logits.shape == (4, 5)


def test_unified_classifier():
    classifier = UnifiedSignalClassifier(ecg_channels=1, eeg_channels=8)
    ecg_t = torch.randn(1, 1, 1000)
    idx, name, prob_dict = classifier.predict_ecg(ecg_t)
    assert 0 <= idx < 5
    assert name in ECGResNet1DBiLSTM.ECG_CLASSES
    assert len(prob_dict) == 5
    
    eeg_t = torch.randn(1, 8, 500)
    idx_eeg, name_eeg, prob_eeg = classifier.predict_eeg(eeg_t)
    assert 0 <= idx_eeg < 5
    assert name_eeg in EEGNetTransformer.EEG_CLASSES
    assert len(prob_eeg) == 5
