import numpy as np
from src.evaluation.metrics import compute_strict_evaluation_metrics


def test_strict_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 0])
    
    # 3-class probabilities
    y_prob = np.eye(3)[y_pred]
    
    class_names = ["ClassA", "ClassB", "ClassC"]
    metrics = compute_strict_evaluation_metrics(y_true, y_pred, y_prob, class_names)
    
    assert "overall" in metrics
    assert "accuracy" in metrics["overall"]
    assert metrics["overall"]["accuracy"] == 0.9
    assert "macro_specificity" in metrics["overall"]
    assert "confusion_matrix" in metrics
    assert "per_class" in metrics
    assert "ClassA" in metrics["per_class"]
    assert metrics["per_class"]["ClassA"]["Sensitivity_Recall"] == 1.0
