import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)


def compute_strict_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    class_names: list[str] = None
) -> dict:
    """
    Computes comprehensive evaluation metrics for multi-class ECG/EEG signal classification:
    - Accuracy & Balanced Accuracy
    - Precision, Recall (Sensitivity), Specificity, F1-Score (Macro/Micro/Weighted)
    - ROC-AUC (One-vs-Rest)
    - Confusion Matrix (Count & Normalized)
    - Per-class breakdowns
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(class_names) if class_names else len(unique_classes)
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # Calculate per-class metrics including Specificity
    per_class_metrics = {}
    total_samples = len(y_true)
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fn + fp)

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
        
        specificities.append(specificity)

        per_class_metrics[class_names[i]] = {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Precision": round(float(precision), 4),
            "Sensitivity_Recall": round(float(sensitivity), 4),
            "Specificity": round(float(specificity), 4),
            "F1_Score": round(float(f1), 4)
        }

    macro_specificity = float(np.mean(specificities))

    # ROC-AUC calculation if probabilities are provided
    roc_auc_macro = 0.0
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
        try:
            roc_auc_macro = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        except Exception:
            roc_auc_macro = 0.0

    return {
        "overall": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall_sensitivity": round(float(rec_macro), 4),
            "macro_specificity": round(macro_specificity, 4),
            "macro_f1_score": round(float(f1_macro), 4),
            "weighted_f1_score": round(float(f1_weighted), 4),
            "roc_auc_score": round(roc_auc_macro, 4)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
        "class_names": class_names,
        "per_class": per_class_metrics
    }
