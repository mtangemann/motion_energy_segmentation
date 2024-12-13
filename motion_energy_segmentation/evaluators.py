"""Evaluators.

Evaluators compute and aggregate performance metrics for a stream of predictions and
targets.
"""

import math
from typing import Dict, List

import pandas as pd
import torch
from benedict import benedict
from torcheval.metrics.functional import binary_auroc


class BinarySegmentationEvaluator:
    """Evaluator for binary segmentation.
    
    This evaluator computes the metrics listed below. For all metrics except AUC the
    prediction is thresholded at 0.5.

    - AUC
    - Accuracy
    - IoU (intersection over union)
    - Precision
    - Recall
    - F-Score

    Example usage:
    ```python
    evaluator = BinarySegmentationEvaluator()

    for input, target in val_data:
        prediction = model(input)
        evaluator.append(input, target)
    
    print(evaluator.summary())  # dict {"auc": ..., "accuracy": ..., ...}
    ```
    """

    def __init__(self) -> None:
        """Initializes the evaluator."""
        self._results: List[Dict[str, float]] = []

    def append(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        metadata: list[dict] | None = None,
    ) -> None:
        """Appends predictions and targets to the evaluator.

        Parameters:
            prediction: Predicted segmentation maps as float tensor of shape
                `(B, 1, H, W)` with values in [0.0, 1.0]. For metrics that require a
                binary prediction, this map is thresholded at `0.5`.
            target: Target segmentation maps of shape `(B, 1, H, W)` with values in
                `{0.0, 1.0}`.
            metadata:
                Example metadata that will be attached to the results data frame.
        
        Raises:
            ValueError: If a target segmentation map contains other values than `0.0` or
                `1.0`.
        """
        if metadata is None:
            metadata = [dict()] * len(prediction)

        examples = zip(prediction, target, metadata)
        for example_prediction, example_target, example_metadata in examples:
            result = self._evaluate(example_prediction, example_target)
            self._results.append({ **example_metadata, **result })

    @staticmethod
    def _evaluate(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        target_values = set(target.unique().tolist())
        if not target_values.issubset({0.0, 1.0}):
            raise ValueError(
                "Target segmentation must only contain values 0.0 and 1.0, "
                f"but got {target_values}"
            )

        prediction_bool = prediction > 0.5
        target_bool = target > 0.5

        auc = binary_auroc(
            prediction.flatten(), target.flatten(), use_fbgemm=False
        ).item()

        num_total = target.numel()
        num_predicted = torch.sum(prediction_bool).item()
        num_target = torch.sum(target_bool).item()
        num_both = torch.sum(prediction_bool & target_bool).item()
        num_any = torch.sum(prediction_bool | target_bool).item()
        num_equal = torch.sum(prediction_bool == target_bool).item()

        accuracy = num_equal / num_total
        precision = num_both / num_predicted if num_predicted > 0 else 1.0
        recall = num_both / num_target if num_target > 0 else 1.0
        iou = num_both / num_any if num_any > 0 else 1.0

        if math.isclose(precision + recall, 0.0):
            f_score = 0.0
        else:
            f_score = 2 * precision * recall / (precision + recall)

        return {
            "auc": auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "iou": iou,
        }

    def results(self) -> pd.DataFrame:
        """Returns the evaluation results per example.
        
        Returns:
            A pandas data frame with the complete evaluation results. Each row contains
                the results for a single example.
        """
        return pd.DataFrame(self._results)

    def summary(self) -> benedict[str, float]:
        """Returns the evaluation results averaged over examples.
        
        Returns:
            A dictionary with the evaluation results averaged over all examples.
        """
        metrics = ["auc", "accuracy", "precision", "recall", "f_score", "iou"]
        return benedict(
            self.results()[metrics].mean().to_dict()
        )
