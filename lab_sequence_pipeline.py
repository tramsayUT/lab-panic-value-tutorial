import logging
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyteConfig:
    """Configuration for a single lab analyte's panic-value pipeline.

    Attributes:
        name: Human-readable analyte name, used in logging and output paths.
        itemid: MIMIC-IV d_labitems itemid for the dominant chemistry draw.
        valid_min: Lower bound for plausibility filtering (drops data errors).
        valid_max: Upper bound for plausibility filtering (drops data errors).
        critical_threshold: Value above which the NEXT draw is labeled critical.
            (For a one-directional panic value, e.g. hyperkalemia.)
    """

    name: str
    itemid: int
    valid_min: float
    valid_max: float
    critical_threshold: float


ANALYTE_CONFIGS: dict[str, AnalyteConfig] = {
    "potassium": AnalyteConfig(
        name="potassium",
        itemid=50971,
        valid_min=1.0,
        valid_max=10.0,
        critical_threshold=6.0,
    ),
    "glucose": AnalyteConfig(
        name="glucose",
        itemid=50931,
        valid_min=10.0,
        valid_max=1000.0,
        critical_threshold=400.0,
    ),
}


FEATURE_QUERY_TEMPLATE = """
WITH filtered_labs AS (
  SELECT
    subject_id,
    hadm_id,
    charttime,
    valuenum
  FROM
    `physionet-data.mimiciv_3_1_hosp.labevents`
  WHERE
    itemid = {itemid}
    AND hadm_id IS NOT NULL
    AND valuenum IS NOT NULL
    AND valuenum BETWEEN {valid_min} AND {valid_max}
),
sequenced AS (
  SELECT
    subject_id,
    hadm_id,
    charttime,
    valuenum,
    LAG(valuenum, 1) OVER w AS prev_value_1,
    LAG(valuenum, 2) OVER w AS prev_value_2,
    LAG(charttime, 1) OVER w AS prev_charttime,
    LEAD(valuenum, 1) OVER w AS next_value
  FROM
    filtered_labs
  WINDOW w AS (PARTITION BY subject_id, hadm_id ORDER BY charttime)
)
SELECT
  subject_id,
  hadm_id,
  charttime,
  valuenum AS current_value,
  prev_value_1,
  prev_value_2,
  TIMESTAMP_DIFF(charttime, prev_charttime, HOUR) AS hours_since_last_draw,
  valuenum - prev_value_1 AS delta_from_prev,
  SAFE_DIVIDE(
    valuenum - prev_value_1,
    TIMESTAMP_DIFF(charttime, prev_charttime, HOUR)
  ) AS rate_of_change_per_hour,
  next_value,
  CASE WHEN next_value > {critical_threshold} THEN 1 ELSE 0 END AS label_next_critical
FROM
  sequenced
WHERE
  prev_value_1 IS NOT NULL
  AND next_value IS NOT NULL
ORDER BY
  subject_id, hadm_id, charttime
"""


def load_feature_data(analyte: str, project_id: Optional[str] = None) -> pd.DataFrame:
    if analyte not in ANALYTE_CONFIGS:
        raise KeyError(
            f"Unknown analyte '{analyte}'. Valid options: {list(ANALYTE_CONFIGS)}"
        )

    resolved_project_id = project_id or os.environ.get("GCP_PROJECT_ID")
    if not resolved_project_id:
        raise ValueError(
            "No GCP project ID provided. Pass project_id= or set the "
            "GCP_PROJECT_ID environment variable."
        )

    config = ANALYTE_CONFIGS[analyte]
    query = FEATURE_QUERY_TEMPLATE.format(
        project_id=resolved_project_id,
        itemid=config.itemid,
        valid_min=config.valid_min,
        valid_max=config.valid_max,
        critical_threshold=config.critical_threshold,
    )

    logger.info("Running feature query for analyte=%s (itemid=%d)", analyte, config.itemid)
    client = bigquery.Client(project=resolved_project_id)
    df = client.query(query).to_dataframe()
    logger.info("Retrieved %d rows for analyte=%s", len(df), analyte)

    return df


def split_by_patient(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "subject_id" not in df.columns:
        raise ValueError("df must contain a 'subject_id' column to split by patient.")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0 to leave room for test.")

    patient_ids = df["subject_id"].unique()

    train_ids, remaining_ids = train_test_split(
        patient_ids, train_size=train_frac, random_state=random_state
    )
    val_relative_frac = val_frac / (1.0 - train_frac)
    val_ids, test_ids = train_test_split(
        remaining_ids, train_size=val_relative_frac, random_state=random_state
    )

    train_df = df[df["subject_id"].isin(train_ids)].copy()
    val_df = df[df["subject_id"].isin(val_ids)].copy()
    test_df = df[df["subject_id"].isin(test_ids)].copy()

    logger.info(
        "Split: %d train patients (%d rows), %d val patients (%d rows), "
        "%d test patients (%d rows)",
        len(train_ids), len(train_df),
        len(val_ids), len(val_df),
        len(test_ids), len(test_df),
    )

    return train_df, val_df, test_df


def validate_split_quality(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_rows = len(split_df)
        n_positive = int(split_df["label_next_critical"].sum())
        pct_positive = 100.0 * n_positive / n_rows if n_rows else 0.0
        n_null_features = split_df[
            ["prev_value_1", "hours_since_last_draw", "rate_of_change_per_hour"]
        ].isnull().sum().sum()

        logger.info(
            "[%s] rows=%d, positive=%d (%.3f%%), null feature values=%d",
            split_name, n_rows, n_positive, pct_positive, n_null_features,
        )


if __name__ == "__main__":
    # Example usage -- requires GCP_PROJECT_ID to be set in the environment.
    for analyte_name in ANALYTE_CONFIGS:
        features_df = load_feature_data(analyte_name)
        train_split, val_split, test_split = split_by_patient(features_df)
        validate_split_quality(train_split, val_split, test_split)
