import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ..evaluation.metrics import compute_strict_evaluation_metrics


class SignalModelTrainer:
    """
    Modular PyTorch Model Trainer for ECG & EEG signal classification architectures.
    Provides automated model training loops, validation monitoring, 
    learning rate scheduling, and strict evaluation metrics calculation.
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": []
        }

    def train_epochs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 15,
        batch_size: int = 32,
        class_names: list[str] = None
    ) -> dict:
        """
        Executes full training & validation procedure over `epochs`.
        """
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # --- Training Phase ---
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            scheduler.step()
            train_loss = running_loss / max(1, total)
            train_acc = correct / max(1, total)

            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            val_preds_list = []
            val_labels_list = []
            val_probs_list = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    val_preds_list.extend(preds.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())
                    val_probs_list.extend(probs.cpu().numpy())

            val_loss = val_loss / max(1, len(val_dataset))
            val_preds_arr = np.array(val_preds_list)
            val_labels_arr = np.array(val_labels_list)
            val_probs_arr = np.array(val_probs_list)
            
            val_metrics = compute_strict_evaluation_metrics(
                y_true=val_labels_arr,
                y_pred=val_preds_arr,
                y_prob=val_probs_arr,
                class_names=class_names
            )
            val_acc = val_metrics["overall"]["accuracy"]
            val_f1 = val_metrics["overall"]["macro_f1_score"]

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["train_acc"].append(round(train_acc, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_acc"].append(round(val_acc, 4))
            self.history["val_f1"].append(round(val_f1, 4))

        return {
            "history": self.history,
            "final_val_metrics": val_metrics
        }
