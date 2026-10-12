"""
sequence_dataset.py

Converts the flat, row-per-draw feature table (from lab_sequence_pipeline)
into per-admission sequences for the LSTM, with dynamic length capping and
padded batching via PyTorch's DataLoader.

Each admission's rows are already a valid sequence of time steps (the
feature query guarantees every row has both a prior and next value, and
sequences do not cross admission boundaries -- see lab_sequence_pipeline's
hadm_id-partitioned window). This module's job is purely reshaping that
flat table into (sequence, label) pairs the LSTM can consume.

Requires:
    pip install torch pandas numpy
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,  # Colab pre-configures the root logger; without this,
                 # basicConfig() silently no-ops and INFO logs vanish.
)
logger = logging.getLogger(__name__)

# Same three features as the baseline, for a fair comparison. The LSTM
# could use more (e.g. rate_of_change_per_hour, prev_value_2), but keeping
# the feature set identical isolates the effect of the sequence model
# itself rather than conflating it with richer inputs.
FEATURE_COLUMNS: list[str] = ["current_value", "delta_from_prev", "hours_since_last_draw"]
LABEL_COLUMN: str = "label_next_critical"


def compute_sequence_length_cap(df: pd.DataFrame, percentile: float = 95.0) -> int:
    """Compute a sequence length cap from the data instead of guessing one.

    Args:
        df: Feature DataFrame with 'subject_id' and 'hadm_id' columns.
        percentile: Which percentile of per-admission sequence lengths to
            use as the cap. Higher = less truncation, more padding waste.

    Returns:
        The length cap (rounded up to the nearest integer).
    """
    lengths = df.groupby(["subject_id", "hadm_id"]).size()
    cap = int(np.ceil(np.percentile(lengths, percentile)))
    pct_truncated = 100.0 * (lengths > cap).sum() / len(lengths)
    logger.info(
        "Sequence length cap at p%.0f = %d (truncates %.2f%% of admissions, "
        "max observed = %d)",
        percentile, cap, pct_truncated, int(lengths.max()),
    )
    return cap


def build_admission_sequences(
    df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    label_column: str = LABEL_COLUMN,
    max_len: Optional[int] = None,
) -> list[dict]:
    """Group the flat feature table into one sequence per admission.

    Args:
        df: Feature DataFrame from lab_sequence_pipeline.load_feature_data
            (optionally pre-filtered to a train/val/test split).
        feature_columns: Columns to use as per-timestep features.
        label_column: Column to use as the per-timestep label.
        max_len: If set, admissions longer than this are truncated to their
            MOST RECENT max_len steps (tail truncation). If None, no
            truncation is applied.

    Returns:
        List of dicts, one per admission, each with:
            'subject_id', 'hadm_id': identifiers (kept for debugging/audit)
            'features': np.ndarray of shape (seq_len, n_features)
            'labels': np.ndarray of shape (seq_len,)

    Raises:
        ValueError: If required columns are missing from df.
    """
    required = ["subject_id", "hadm_id", "charttime"] + feature_columns + [label_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    sequences = []
    n_truncated = 0
    for (subject_id, hadm_id), group in df.groupby(["subject_id", "hadm_id"]):
        group_sorted = group.sort_values("charttime")
        if max_len is not None and len(group_sorted) > max_len:
            group_sorted = group_sorted.tail(max_len)
            n_truncated += 1

        sequences.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "features": group_sorted[feature_columns].to_numpy(dtype=np.float32),
            "labels": group_sorted[label_column].to_numpy(dtype=np.float32),
        })

    logger.info(
        "Built %d admission sequences (%d truncated to max_len=%s)",
        len(sequences), n_truncated, max_len,
    )
    return sequences


class PanicValueSequenceDataset(Dataset):
    """PyTorch Dataset wrapping per-admission (features, labels) sequences."""

    def __init__(self, sequences: list[dict]):
        """
        Args:
            sequences: Output of build_admission_sequences.
        """
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Returns (features, labels, length) for one admission.

        Length is returned alongside the raw (unpadded) tensors so the
        collate function can build an accurate padding mask -- padding
        happens at batch time, not here, since the max length within a
        batch is usually much shorter than the dataset-wide cap.
        """
        item = self.sequences[idx]
        features = torch.from_numpy(item["features"])
        labels = torch.from_numpy(item["labels"])
        return features, labels, len(labels)


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length sequences to the batch's own max length.

    Args:
        batch: List of (features, labels, length) tuples from the Dataset.

    Returns:
        padded_features: (batch_size, max_seq_len, n_features)
        padded_labels: (batch_size, max_seq_len)
        mask: (batch_size, max_seq_len) -- 1.0 for real timesteps, 0.0 for
            padding. Use this to mask the loss and metrics so padded steps
            never influence training or evaluation.
    """
    features_list, labels_list, lengths = zip(*batch)

    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=0.0)

    max_len = padded_features.shape[1]
    mask = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0

    return padded_features, padded_labels, mask


def make_dataloader(
    df: pd.DataFrame,
    max_len: Optional[int] = None,
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader directly from a flat feature DataFrame (one split).

    Args:
        df: Feature DataFrame, e.g. train_df, val_df, or test_df from
            lab_sequence_pipeline.split_by_patient.
        max_len: Sequence length cap (see compute_sequence_length_cap).
            Pass the value computed on the TRAINING split for all three
            splits, so val/test use the same truncation rule as training.
        batch_size: Admissions per batch.
        shuffle: Shuffle admission order each epoch. Use True for training,
            False for val/test (order doesn't matter for evaluation, and
            keeping it False makes debugging easier).

    Returns:
        A DataLoader yielding (padded_features, padded_labels, mask) batches.
    """
    sequences = build_admission_sequences(df, max_len=max_len)
    dataset = PanicValueSequenceDataset(sequences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
