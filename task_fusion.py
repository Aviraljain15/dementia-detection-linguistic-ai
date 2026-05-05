import numpy as np
from typing import Dict
def late_fusion_predict(task_probs: Dict[str, np.ndarray],
                        task_weights: Dict[str, float],
                        threshold: float = 0.5):
    """
    Perform weighted late fusion of task-level probabilities.

    Args:
        task_probs: dict of {task_name: np.array of probabilities}
        task_weights: dict of {task_name: weight (e.g., AUC)}
        threshold: classification threshold

    Returns:
        preds: final binary predictions
        fused_probs: fused probability scores
    """

    # ---------------- VALIDATION ---------------- #
    if not task_probs:
        raise ValueError("task_probs is empty")

    if set(task_probs.keys()) != set(task_weights.keys()):
        raise ValueError("Mismatch between tasks in probs and weights")

    # ---------------- INIT ---------------- #
    first_key = list(task_probs.keys())[0]
    fused_probs = np.zeros_like(task_probs[first_key], dtype=float)

    total_weight = sum(task_weights.values())

    # ---------------- FUSION ---------------- #
    for task, probs in task_probs.items():
        weight = task_weights[task] / total_weight
        fused_probs += probs * weight

    # ---------------- FINAL PREDICTION ---------------- #
    preds = (fused_probs > threshold).astype(int)

    return preds, fused_probs