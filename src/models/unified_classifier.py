import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .ecg_model import ECGResNet1DBiLSTM
from .eeg_model import EEGNetTransformer


class UnifiedSignalClassifier(nn.Module):
    """
    Unified Deep Learning Classifier for ECG and EEG bio-signals.
    Provides automated model selection, preprocessing integration, 
    and multi-class probability outputs.
    """
    def __init__(self, ecg_channels: int = 1, eeg_channels: int = 8):
        super().__init__()
        self.ecg_model = ECGResNet1DBiLSTM(in_channels=ecg_channels, num_classes=5)
        self.eeg_model = EEGNetTransformer(num_channels=eeg_channels, num_classes=5)

    def predict_ecg(self, signal_tensor: torch.Tensor) -> tuple[int, str, dict]:
        """
        Runs ECG model inference and returns (predicted_class_idx, class_name, probabilities_dict)
        """
        self.ecg_model.eval()
        with torch.no_grad():
            logits = self.ecg_model(signal_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            class_name = ECGResNet1DBiLSTM.ECG_CLASSES[pred_idx]
            
            prob_dict = {
                ECGResNet1DBiLSTM.ECG_CLASSES[i]: round(float(probs[i]), 4)
                for i in range(len(probs))
            }
            return pred_idx, class_name, prob_dict

    def predict_eeg(self, signal_tensor: torch.Tensor) -> tuple[int, str, dict]:
        """
        Runs EEG model inference and returns (predicted_class_idx, class_name, probabilities_dict)
        """
        self.eeg_model.eval()
        with torch.no_grad():
            logits = self.eeg_model(signal_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            class_name = EEGNetTransformer.EEG_CLASSES[pred_idx]
            
            prob_dict = {
                EEGNetTransformer.EEG_CLASSES[i]: round(float(probs[i]), 4)
                for i in range(len(probs))
            }
            return pred_idx, class_name, prob_dict
