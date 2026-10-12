"""
lstm_model.py

LSTM sequence model for panic-value prediction, plus training and
evaluation functions. Metrics (AUPRC, AUROC, sensitivity, specificity) are
computed identically to baseline_model.py -- same sklearn functions, same
definitions -- so the two models' results are directly comparable.

Requires:
    pip install torch scikit-learn numpy pandas
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,  # Colab gotcha -- see sequence_dataset.py / baseline_model.py.
)
logger = logging.getLogger(__name__)


class PanicValueLSTM(nn.Module):
    """Single-layer LSTM producing a per-timestep critical-value logit.

    Deliberately simple architecture (small hidden size, one layer) given
    the limited feature set (3 inputs) and the goal of demonstrating
    whether sequence modeling helps at all over the baseline -- not of
    maximizing raw performance.
    """

    def __init__(self, n_features: int = 3, hidden_size: int = 32, num_layers: int = 1):
        """
        Args:
            n_features: Number of input features per timestep.
            hidden_size: LSTM hidden state dimensionality.
            num_layers: Number of stacked LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_features)

        Returns:
            Logits of shape (batch_size, seq_len) -- one per timestep.
            Apply sigmoid externally (loss function expects raw logits).
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        logits = self.output_layer(lstm_out).squeeze(-1)  # (batch, seq_len)
        return logits


def compute_pos_weight(train_loader: DataLoader) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight from the training split's class ratio.

    Mirrors class_weight='balanced' in the baseline: weights positive
    examples by (negative_count / positive_count) so the rare critical
    class isn't drowned out during training.

    Args:
        train_loader: DataLoader for the training split.

    Returns:
        Scalar tensor to pass as pos_weight to BCEWithLogitsLoss.

    Raises:
        ValueError: If the training split contains zero positive labels
            (pos_weight would be undefined / infinite).
    """
    n_pos, n_neg = 0.0, 0.0
    for _, labels, mask in train_loader:
        valid_labels = labels[mask.bool()]
        n_pos += valid_labels.sum().item()
        n_neg += (valid_labels.numel() - valid_labels.sum().item())

    if n_pos == 0:
        raise ValueError("Training split has zero positive labels; cannot compute pos_weight.")

    pos_weight = n_neg / n_pos
    logger.info("Computed pos_weight=%.2f (n_pos=%d, n_neg=%d)", pos_weight, n_pos, n_neg)
    return torch.tensor(pos_weight)


def train_lstm(
    model: PanicValueLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    patience: int = 3,
    device: Optional[str] = None,
) -> PanicValueLSTM:
    """Train the LSTM with masked loss and simple early stopping on val AUPRC.

    Args:
        model: Freshly initialized PanicValueLSTM.
        train_loader: DataLoader for the training split (shuffle=True).
        val_loader: DataLoader for the validation split (shuffle=False).
        epochs: Maximum number of training epochs.
        learning_rate: Adam learning rate.
        patience: Stop early if val AUPRC doesn't improve for this many
            consecutive epochs. Set to >= epochs to disable early stopping.
        device: 'cuda' or 'cpu'. Autodetects if not specified.

    Returns:
        The model with weights from its BEST epoch (by val AUPRC) restored
        -- not necessarily the weights from the final epoch.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pos_weight = compute_pos_weight(train_loader).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")  # reduction='none' so we can mask
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_auprc = -1.0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_valid_steps = 0.0, 0

        for features, labels, mask in train_loader:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)

            optimizer.zero_grad()
            logits = model(features)

            # Weight positive timesteps manually since BCEWithLogitsLoss's
            # built-in pos_weight assumes a single scalar broadcast, which
            # works fine here -- but we still need per-element masking,
            # which pos_weight alone doesn't provide.
            raw_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight, reduction="none"
            )
            masked_loss = (raw_loss * mask).sum() / mask.sum()

            masked_loss.backward()
            optimizer.step()

            total_loss += masked_loss.item() * mask.sum().item()
            total_valid_steps += mask.sum().item()

        train_loss = total_loss / total_valid_steps
        val_metrics = evaluate_lstm(model, val_loader, device=device)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_AUPRC=%.4f | val_AUROC=%.4f",
            epoch, epochs, train_loss, val_metrics["auprc"], val_metrics["auroc"],
        )

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Restored best model weights (val_AUPRC=%.4f)", best_val_auprc)

    return model


def _collect_predictions(
    model: PanicValueLSTM, data_loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a DataLoader and collect flattened (y_true, y_proba).

    Only valid (non-padded) timesteps are included, using the batch mask.
    """
    model.eval()
    all_labels, all_probas = [], []

    with torch.no_grad():
        for features, labels, mask in data_loader:
            features = features.to(device)
            logits = model(features).cpu()
            probas = torch.sigmoid(logits)

            mask_bool = mask.bool()
            all_labels.append(labels[mask_bool].numpy())
            all_probas.append(probas[mask_bool].numpy())

    return np.concatenate(all_labels), np.concatenate(all_probas)


def evaluate_lstm(
    model: PanicValueLSTM,
    data_loader: DataLoader,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> dict[str, float]:
    """Evaluate the LSTM on a held-out split, flattened over all valid timesteps.

    Metric definitions match baseline_model.evaluate_model exactly, so
    results are directly comparable between the two models.

    Args:
        model: Trained PanicValueLSTM.
        data_loader: DataLoader for the split to evaluate (val or test).
        threshold: Probability cutoff for sensitivity/specificity.
        device: 'cuda' or 'cpu'. Autodetects if not specified.

    Returns:
        Dict with auprc, auroc, sensitivity, specificity, threshold,
        n_rows, n_positive -- same keys as baseline_model.evaluate_model.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y_true, y_proba = _collect_predictions(model, data_loader, device)
    y_pred = (y_proba >= threshold).astype(int)

    auprc = average_precision_score(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "auprc": auprc,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold,
        "n_rows": len(y_true),
        "n_positive": int(y_true.sum()),
    }


def sweep_thresholds_lstm(
    model: PanicValueLSTM,
    data_loader: DataLoader,
    thresholds: Optional[list[float]] = None,
    device: Optional[str] = None,
) -> "pd.DataFrame":  # noqa: F821 -- pandas imported lazily below
    """Threshold sweep for the LSTM, mirroring baseline_model.sweep_thresholds.

    Args:
        model: Trained PanicValueLSTM.
        data_loader: DataLoader for the split to sweep (use val, not test).
        thresholds: Cutoffs to evaluate. Defaults to 0.05-0.95 in steps of 0.05.
        device: 'cuda' or 'cpu'. Autodetects if not specified.

    Returns:
        DataFrame with threshold, sensitivity, specificity, precision,
        n_flagged -- same shape as baseline_model.sweep_thresholds's output.
    """
    import pandas as pd  # local import to avoid a hard pandas dependency at module load

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

    y_true, y_proba = _collect_predictions(model, data_loader, device)

    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rows.append({
            "threshold": threshold,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "n_flagged": int(y_pred.sum()),
        })

    return pd.DataFrame(rows)
