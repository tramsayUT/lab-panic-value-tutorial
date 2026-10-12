# Predicting Critical Lab Values from Draw History

Predicts whether a patient's next lab draw will be critical, comparing a logistic regression baseline with an LSTM on admission-scoped draw sequences from MIMIC-IV. Potassium (critical > 6.0 mEq/L) is the primary analyte. High-side glucose (> 400 mg/dL) is a generalization check.

## Contents

| File | Purpose |
|---|---|
| `panic_value_tutorial.ipynb` | Main tutorial notebook (run this) |
| `lab_sequence_pipeline.py` | BigQuery feature query and patient-level split |
| `baseline_model.py` | Logistic regression baseline and evaluation |
| `sequence_dataset.py` | Admission sequences, padding and batching |
| `lstm_model.py` | LSTM model, training and evaluation |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.10+ and the packages in `requirements.txt`
- PhysioNet-credentialed access to MIMIC-IV v3.1 on BigQuery
- A Google Cloud project with the BigQuery API enabled


## How to run

1. Open `panic_value_tutorial.ipynb` in Google Colab.
2. Upload the four `.py` files to the Colab session (Files pane).
3. In the setup cell, replace `YOUR_GCP_PROJECT_ID` with your own project ID.
4. Run all cells.
