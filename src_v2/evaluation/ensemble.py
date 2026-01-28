"""
Ensemble evaluation for classifier cross-validation models.

Provides soft voting (weighted probability averaging) and hard voting
(majority vote) for combining predictions from multiple CV fold models.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src_v2.models import create_classifier


def load_ensemble_models(
    checkpoint_paths: List[str],
    device: torch.device
) -> Tuple[List[nn.Module], List[float]]:
    """
    Load ensemble models from checkpoints with validation weights.

    Extracts validation F1-macro scores from results.json in each fold directory
    to use as ensemble weights. Verifies architecture consistency across models.

    Args:
        checkpoint_paths: List of paths to best_classifier.pt files
        device: Device to load models on

    Returns:
        Tuple of (models list, validation F1-macro weights list)

    Raises:
        ValueError: If architectures don't match across models
        FileNotFoundError: If results.json missing from fold directory
    """
    models = []
    weights = []
    reference_backbone = None

    for i, checkpoint_path in enumerate(checkpoint_paths):
        # Load model using factory
        model = create_classifier(checkpoint=checkpoint_path, device=device)
        model.eval()

        # Verify architecture match
        if reference_backbone is None:
            reference_backbone = model.backbone_name
        elif model.backbone_name != reference_backbone:
            raise ValueError(
                f"Architecture mismatch at model {i}: "
                f"has {model.backbone_name}, expected {reference_backbone}"
            )

        # Extract fold directory from checkpoint path
        fold_dir = Path(checkpoint_path).parent
        results_path = fold_dir / "results.json"

        if not results_path.exists():
            raise FileNotFoundError(
                f"Validation results not found: {results_path}\n"
                f"Expected results.json in fold directory."
            )

        # Read validation F1-macro weight
        with open(results_path) as f:
            results = json.load(f)

        val_f1 = results.get("best_val_f1")
        if val_f1 is None:
            raise ValueError(
                f"Missing 'best_val_f1' field in {results_path}\n"
                f"Cannot extract validation weight for ensemble."
            )

        models.append(model)
        weights.append(val_f1)

    return models, weights


def weighted_soft_voting(
    probabilities: List[torch.Tensor],
    weights: List[float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine probability predictions using weighted averaging.

    Weights are normalized to sum to 1.0 before averaging. This preserves
    the probability distribution property (sums to 1.0 across classes).

    Args:
        probabilities: List of probability tensors [(N, num_classes), ...]
        weights: List of validation F1-macro scores [float, ...]

    Returns:
        Tuple of (predictions, weighted_probabilities)
        - predictions: (N,) tensor of predicted class indices
        - weighted_probabilities: (N, num_classes) weighted average probabilities
    """
    # Stack probabilities: (num_models, N, num_classes)
    probs_stacked = torch.stack(probabilities, dim=0)

    # Normalize weights to sum to 1.0
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_normalized = weights_tensor / weights_tensor.sum()

    # Weighted average using einsum: (N, num_classes)
    # 'm' = models, 'n' = samples, 'i' = classes
    weighted_probs = torch.einsum('mni,m->ni', probs_stacked, weights_normalized)

    # Predict: argmax over classes
    predictions = weighted_probs.argmax(dim=1)

    return predictions, weighted_probs


def hard_voting(
    class_predictions: List[torch.Tensor]
) -> torch.Tensor:
    """
    Majority vote across model predictions.

    Uses Counter.most_common() for voting. Ties are broken by taking the
    lowest class index (deterministic tie-breaking).

    Args:
        class_predictions: List of prediction tensors [(N,), ...] of class indices

    Returns:
        Ensemble predictions (N,) tensor of class indices
    """
    # Stack predictions: (num_models, N)
    preds_stacked = torch.stack(class_predictions, dim=0)

    # Mode along model axis (axis 0)
    ensemble_preds = []
    for i in range(preds_stacked.shape[1]):
        sample_votes = preds_stacked[:, i].tolist()
        # most_common returns [(value, count), ...]
        # If tie, take first (lowest class index due to Counter ordering)
        majority_vote = Counter(sample_votes).most_common(1)[0][0]
        ensemble_preds.append(majority_vote)

    return torch.tensor(ensemble_preds)


@torch.no_grad()
def ensemble_inference(
    models: List[nn.Module],
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[List[torch.Tensor], List[torch.Tensor], np.ndarray]:
    """
    Run inference with all models on entire dataset.

    Collects per-model predictions and probabilities for ensemble voting.
    Uses tqdm progress bar for user feedback.

    Args:
        models: List of models in eval mode
        dataloader: DataLoader for test set
        device: Device for inference

    Returns:
        Tuple of (model_preds, model_probs, labels)
        - model_preds: List of prediction tensors [(N,), ...] per model
        - model_probs: List of probability tensors [(N, num_classes), ...] per model
        - labels: (N,) numpy array of true labels
    """
    all_model_preds = [[] for _ in models]
    all_model_probs = [[] for _ in models]
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Ensemble inference"):
        inputs = inputs.to(device)

        # Run inference with each model
        for model_idx, model in enumerate(models):
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_model_probs[model_idx].append(probs.cpu())
            all_model_preds[model_idx].append(preds.cpu())

        all_labels.extend(labels.numpy())

    # Concatenate batches
    model_preds = [torch.cat(p) for p in all_model_preds]
    model_probs = [torch.cat(p) for p in all_model_probs]

    return model_preds, model_probs, np.array(all_labels)


@torch.no_grad()
def predict_with_tta_classifier(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Model-level TTA: average predictions from original + flipped images.

    Unlike landmark TTA, classifier TTA does NOT need symmetry correction
    because class labels (COVID, Normal, Viral_Pneumonia) are anatomically
    symmetric - a flipped COVID X-ray is still COVID.

    Args:
        model: Classifier model in eval mode
        images: Batch (B, 3, H, W)
        device: Device

    Returns:
        Tuple of (tta_averaged_probs, probs_original, probs_flipped)
        - tta_averaged_probs: (B, num_classes) averaged probabilities
        - probs_original: (B, num_classes) original image probabilities
        - probs_flipped: (B, num_classes) flipped image probabilities
    """
    model.eval()
    images = images.to(device)

    # Original prediction
    logits_orig = model(images)
    probs_orig = torch.softmax(logits_orig, dim=1)

    # Flipped prediction (horizontal flip along width dimension)
    images_flipped = torch.flip(images, dims=[3])
    logits_flip = model(images_flipped)
    probs_flip = torch.softmax(logits_flip, dim=1)

    # Simple average (no symmetry correction needed for class labels)
    tta_probs = (probs_orig + probs_flip) / 2

    return tta_probs, probs_orig, probs_flip


@torch.no_grad()
def ensemble_inference_with_tta(
    models: List[nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    use_tta: bool = True
) -> Tuple[List[torch.Tensor], List[torch.Tensor], np.ndarray, Dict]:
    """
    Run ensemble inference with optional TTA on entire dataset.

    Implements dual-level TTA:
    1. Model-level TTA: Each model averages its orig + flip predictions
    2. Ensemble-level TTA: The 5 model-level TTA predictions are ensemble-averaged

    Args:
        models: List of models in eval mode
        dataloader: DataLoader for test set
        device: Device for inference
        use_tta: Whether to apply TTA (default True)

    Returns:
        Tuple of (model_preds, model_probs, labels, tta_details)
        - model_preds: List of prediction tensors [(N,), ...] per model
        - model_probs: List of probability tensors [(N, num_classes), ...] per model
        - labels: (N,) numpy array of true labels
        - tta_details: Dict with original/flipped predictions per model (None if use_tta=False)
    """
    all_model_preds = [[] for _ in models]
    all_model_probs = [[] for _ in models]
    all_labels = []

    # TTA details for traceability
    tta_details = None
    if use_tta:
        tta_details = {
            "model_probs_original": [[] for _ in models],
            "model_probs_flipped": [[] for _ in models],
        }

    for inputs, labels in tqdm(dataloader, desc="Ensemble inference" + (" with TTA" if use_tta else "")):
        inputs = inputs.to(device)

        # Run inference with each model
        for model_idx, model in enumerate(models):
            if use_tta:
                # TTA: average original and flipped predictions
                tta_probs, orig_probs, flip_probs = predict_with_tta_classifier(
                    model, inputs, device
                )
                probs = tta_probs
                tta_details["model_probs_original"][model_idx].append(orig_probs.cpu())
                tta_details["model_probs_flipped"][model_idx].append(flip_probs.cpu())
            else:
                # No TTA: just run model directly
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)

            all_model_probs[model_idx].append(probs.cpu())
            all_model_preds[model_idx].append(preds.cpu())

        all_labels.extend(labels.numpy())

    # Concatenate batches
    model_preds = [torch.cat(p) for p in all_model_preds]
    model_probs = [torch.cat(p) for p in all_model_probs]

    if use_tta:
        tta_details["model_probs_original"] = [torch.cat(p) for p in tta_details["model_probs_original"]]
        tta_details["model_probs_flipped"] = [torch.cat(p) for p in tta_details["model_probs_flipped"]]

    return model_preds, model_probs, np.array(all_labels), tta_details


def validate_ensemble_setup(
    models: List[nn.Module],
    dataloader: DataLoader,
    expected_samples: int = 1895
) -> None:
    """
    Pre-flight sanity checks before ensemble evaluation.

    Validates:
    1. Architecture match across all models
    2. All models in eval mode
    3. Dataset sample count matches expected
    4. Probability outputs sum to 1.0
    5. Predictions in valid class range

    Args:
        models: List of loaded models
        dataloader: Test set dataloader
        expected_samples: Expected total sample count (default: 1895 from Phase 1)

    Raises:
        AssertionError: If any validation check fails
    """
    # 1. Architecture match
    reference_backbone = models[0].backbone_name
    for i, model in enumerate(models[1:], start=1):
        assert model.backbone_name == reference_backbone, \
            f"Model {i} has backbone {model.backbone_name}, expected {reference_backbone}"

    # 2. Models in eval mode
    for i, model in enumerate(models):
        assert not model.training, \
            f"Model {i} is in training mode, call model.eval() before inference"

    # 3. Sample count
    total_samples = len(dataloader.dataset)
    assert total_samples == expected_samples, \
        f"Dataset has {total_samples} samples, expected {expected_samples}"

    # 4. Test probability output on single batch
    test_batch = next(iter(dataloader))[0][:1]  # Single image
    test_batch = test_batch.to(next(models[0].parameters()).device)

    with torch.no_grad():
        test_logits = models[0](test_batch)
        test_probs = torch.softmax(test_logits, dim=1)

    # Probabilities sum to 1.0
    prob_sum = test_probs.sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones(1, device=prob_sum.device), atol=1e-5), \
        f"Probability sum = {prob_sum.item():.6f}, expected 1.0"

    # 5. Valid prediction range
    test_pred = test_logits.argmax(dim=1)
    num_classes = models[0].num_classes
    assert 0 <= test_pred < num_classes, \
        f"Prediction {test_pred.item()} outside valid range [0, {num_classes})"

    print("✓ All sanity checks passed")


def categorize_tta_impact(
    pred_baseline: np.ndarray,
    pred_tta: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, Any]:
    """
    Categorize TTA impact per sample: helped, hurt, or neutral.

    Categories:
    - helped: Baseline wrong, TTA correct
    - hurt: Baseline correct, TTA wrong
    - neutral: Both correct OR both wrong (same outcome)

    Args:
        pred_baseline: (N,) array of baseline predictions (no TTA)
        pred_tta: (N,) array of TTA predictions
        ground_truth: (N,) array of true labels

    Returns:
        Dict with 'per_sample' list and 'summary' counts
    """
    baseline_correct = (pred_baseline == ground_truth)
    tta_correct = (pred_tta == ground_truth)

    per_sample = []
    summary = {"helped": 0, "hurt": 0, "neutral": 0}

    for i in range(len(ground_truth)):
        if not baseline_correct[i] and tta_correct[i]:
            impact = "helped"
        elif baseline_correct[i] and not tta_correct[i]:
            impact = "hurt"
        else:
            impact = "neutral"

        per_sample.append({
            "sample_idx": i,
            "ground_truth": int(ground_truth[i]),
            "baseline_pred": int(pred_baseline[i]),
            "tta_pred": int(pred_tta[i]),
            "impact": impact
        })
        summary[impact] += 1

    return {
        "per_sample": per_sample,
        "summary": summary
    }


def compute_tta_delta_metrics(
    baseline_metrics: Dict[str, float],
    tta_metrics: Dict[str, float],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Compute delta between TTA and baseline metrics.

    Args:
        baseline_metrics: Dict with 'accuracy', 'f1_macro', 'per_class' from baseline
        tta_metrics: Dict with same keys from TTA evaluation
        class_names: List of class names for per-class delta

    Returns:
        Dict with overall and per-class deltas
    """
    return {
        "accuracy_delta": tta_metrics["accuracy"] - baseline_metrics["accuracy"],
        "f1_macro_delta": tta_metrics["f1_macro"] - baseline_metrics["f1_macro"],
        "per_class_f1_delta": {
            cls: tta_metrics["per_class"][cls]["f1-score"] - baseline_metrics["per_class"][cls]["f1-score"]
            for cls in class_names
        }
    }
