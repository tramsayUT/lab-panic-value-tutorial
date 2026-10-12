import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,  # Colab pre-configures the root logger; without force=True
                 # this basicConfig() call silently no-ops and INFO logs
                 # never appear in notebook output.
)
logger = logging.getLogger(__name__)


# Deliberately minimal feature set. current_value stands in for "last known
# value" at prediction time (the row's own draw, not prev_value_1, which is
# one draw further back). rate_of_change_per_hour is left OUT of the
# baseline intentionally -- it has a small number of nulls (division by
# zero when two draws share a timestamp) and including it here would mean
# applying a different row-drop rule to the baseline than to the sequence
# model, muddying the comparison between them.
FEATURE_COLUMNS: list[str] = ["current_value", "delta_from_prev", "hours_since_last_draw"]
LABEL_COLUMN: str = "label_next_critical"


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    n_before = len(df)
    clean_df = df.dropna(subset=required_columns)
    n_dropped = n_before - len(clean_df)
    if n_dropped:
        logger.info(
            "Dropped %d rows (%.3f%%) with nulls in feature/label columns",
            n_dropped, 100.0 * n_dropped / n_before,
        )

    X = clean_df[FEATURE_COLUMNS].copy()
    y = clean_df[LABEL_COLUMN].copy()
    return X, y


def train_baseline_model(
    train_df: pd.DataFrame, random_state: int = 42
) -> tuple[LogisticRegression, StandardScaler]:
    X_train, y_train = prepare_features(train_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    logger.info("Trained baseline logistic regression on %d rows", len(X_train))
    return model, scaler


def evaluate_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, float]:
    X, y = prepare_features(df)
    X_scaled = scaler.transform(X)

    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auprc = average_precision_score(y, y_proba)
    auroc = roc_auc_score(y, y_proba)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "auprc": auprc,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold,
        "n_rows": len(y),
        "n_positive": int(y.sum()),
    }


def sweep_thresholds(
    model: LogisticRegression,
    scaler: StandardScaler,
    df: pd.DataFrame,
    thresholds: Optional[list[float]] = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

    X, y = prepare_features(df)
    X_scaled = scaler.transform(X)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
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


def print_evaluation_report(metrics: dict[str, float], split_name: str) -> None:
    logger.info(
        "[%s] n=%d (positive=%d) | AUPRC=%.4f | AUROC=%.4f | "
        "sensitivity=%.4f | specificity=%.4f @ threshold=%.2f",
        split_name,
        metrics["n_rows"],
        metrics["n_positive"],
        metrics["auprc"],
        metrics["auroc"],
        metrics["sensitivity"],
        metrics["specificity"],
        metrics["threshold"],
    )
