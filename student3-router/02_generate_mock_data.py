"""
BioSense-AI - Mock Data Generator (02_generate_mock_data.py)
------------------------------------------------------------
Generates realistic mock datasets for Student 1 (ECG) and Student 2 (PPG)
compliant with 01_schema_contract.py for Week 2 deliverables.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple

# Ensure current scratch dir is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location("schema_contract", os.path.join(os.path.dirname(__file__), "01_schema_contract.py"))
schema_contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema_contract)

COMPOSITE_KEY = schema_contract.COMPOSITE_KEY

np.random.seed(42)

def generate_mock_datasets(
    output_dir_ecg: str = "student1_ecg",
    output_dir_ppg: str = "student2_ppg",
    n_subjects: int = 3
) -> Tuple[str, str]:
    """Generates mock parquet files for ECG and PPG."""
    subjects = [f"s{i+1}" for i in range(n_subjects)]
    activities = ["sit", "walk", "run"]
    window_len_s = 8.0
    step_s = 4.0
    total_duration_s = 120.0  # 2 minutes per activity
    
    rows_ecg = []
    rows_ppg = []

    for sub in subjects:
        for act in activities:
            base_hr = 65.0 if act == "sit" else (95.0 if act == "walk" else 135.0)
            time_starts = np.arange(0, total_duration_s - window_len_s, step_s)
            
            for t_start in time_starts:
                t_end = t_start + window_len_s
                hr_ref = float(np.round(base_hr + np.random.normal(0, 2.0), 2))
                
                # --- ECG (Student 1) ---
                noise_ecg = np.random.normal(0, 1.0 if act == "sit" else 3.0)
                hr_ecg = float(np.round(hr_ref + noise_ecg, 2))
                err_ecg = float(np.round(abs(hr_ecg - hr_ref), 2))
                pred_err_ecg = float(np.round(max(0.0, err_ecg + np.random.normal(0, 0.5)), 2))
                
                rows_ecg.append({
                    "subject_id": sub,
                    "activity": act,
                    "window_start_s": float(t_start),
                    "window_end_s": float(t_end),
                    "hr_ref": hr_ref,
                    "hr_ecg": hr_ecg,
                    "err_ecg": err_ecg,
                    "predicted_err_ecg": pred_err_ecg,
                    "ecg_sqi": float(np.clip(np.round(1.0 - (err_ecg / 20.0), 3), 0.0, 1.0)),
                    "ecg_amplitude_mv": float(np.round(np.random.uniform(0.8, 1.5), 3))
                })

                # --- PPG (Student 2) ---
                ppg_noise_std = 1.5 if act == "sit" else (5.0 if act == "walk" else 10.0)
                noise_ppg = np.random.normal(0, ppg_noise_std)
                hr_ppg = float(np.round(hr_ref + noise_ppg, 2))
                err_ppg = float(np.round(abs(hr_ppg - hr_ref), 2))
                pred_err_ppg = float(np.round(max(0.0, err_ppg + np.random.normal(0, 1.0)), 2))
                
                rows_ppg.append({
                    "subject_id": sub,
                    "activity": act,
                    "window_start_s": float(t_start),
                    "window_end_s": float(t_end),
                    "hr_ref": hr_ref,
                    "hr_ppg": hr_ppg,
                    "err_ppg": err_ppg,
                    "predicted_err_ppg": pred_err_ppg,
                    "ppg_sqi": float(np.clip(np.round(1.0 - (err_ppg / 25.0), 3), 0.0, 1.0)),
                    "imu_acc_magnitude": float(np.round(0.1 if act == "sit" else (1.2 if act == "walk" else 2.8), 2)),
                    "pressure_lc1": float(np.round(np.random.uniform(10.0, 50.0), 2))
                })

    df_ecg = pd.DataFrame(rows_ecg)
    df_ppg = pd.DataFrame(rows_ppg)

    # Introduce minor asymmetry (drop 2 windows from PPG) to test merge checker coverage report
    df_ppg = df_ppg.iloc[:-2].copy()

    os.makedirs(output_dir_ecg, exist_ok=True)
    os.makedirs(output_dir_ppg, exist_ok=True)

    path_ecg = os.path.join(output_dir_ecg, "window_table_ecg.parquet")
    path_ppg = os.path.join(output_dir_ppg, "window_table_ppg.parquet")

    df_ecg.to_parquet(path_ecg, index=False)
    df_ppg.to_parquet(path_ppg, index=False)

    print(f"[OK] Mock ECG dataset saved: {path_ecg} ({len(df_ecg)} windows)")
    print(f"[OK] Mock PPG dataset saved: {path_ppg} ({len(df_ppg)} windows)")

    return path_ecg, path_ppg

if __name__ == "__main__":
    generate_mock_datasets()
