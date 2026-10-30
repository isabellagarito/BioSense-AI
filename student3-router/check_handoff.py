"""
BioSense-AI - Handoff Checker and Integration Pipeline (check_handoff.py)
------------------------------------------------------------------------
Validates Student 1 (ECG) and Student 2 (PPG) handoff files, performs
strict composite key merge, outputs 'multimodal_windows.parquet' and
initializes 'student3_router/predictions_all_methods.parquet'.
"""

import os
import sys
import logging
import importlib.util
import pandas as pd
import numpy as np

# Dynamically import 01_schema_contract.py
scratch_dir = os.path.dirname(os.path.abspath(__file__))
contract_path = os.path.join(scratch_dir, "01_schema_contract.py")

spec = importlib.util.spec_from_file_location("schema_contract", contract_path)
schema_contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema_contract)

COMPOSITE_KEY = schema_contract.COMPOSITE_KEY
REQUIRED_COLS_ECG = schema_contract.REQUIRED_COLS_ECG
REQUIRED_COLS_PPG = schema_contract.REQUIRED_COLS_PPG
REQUIRED_COLS_ROUTER_OUTPUT = schema_contract.REQUIRED_COLS_ROUTER_OUTPUT
validate_dataframe_schema = schema_contract.validate_dataframe_schema

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

def run_handoff_check(
    path_ecg: str = "student1_ecg/window_table_ecg.parquet",
    path_ppg: str = "student2_ppg/window_table_ppg.parquet",
    out_multimodal_path: str = "multimodal_windows.parquet",
    out_router_dir: str = "student3_router"
) -> bool:
    logging.info("=== BioSense-AI Handoff Verification Pipeline ===")
    
    if not os.path.exists(path_ecg):
        logging.error(f"ECG file not found at: {path_ecg}")
        return False
    if not os.path.exists(path_ppg):
        logging.error(f"PPG file not found at: {path_ppg}")
        return False

    df_ecg = pd.read_parquet(path_ecg)
    df_ppg = pd.read_parquet(path_ppg)

    # 1. Validate Schemas
    report_ecg = validate_dataframe_schema(df_ecg, "Student 1 (ECG)", REQUIRED_COLS_ECG)
    report_ppg = validate_dataframe_schema(df_ppg, "Student 2 (PPG)", REQUIRED_COLS_PPG)

    logging.info(f"ECG Validation: Valid={report_ecg.is_valid}, Missing={report_ecg.missing_cols}, NullKeys={report_ecg.null_key_count}, DupKeys={report_ecg.duplicate_key_count}")
    logging.info(f"PPG Validation: Valid={report_ppg.is_valid}, Missing={report_ppg.missing_cols}, NullKeys={report_ppg.null_key_count}, DupKeys={report_ppg.duplicate_key_count}")

    if not (report_ecg.is_valid and report_ppg.is_valid):
        logging.error(" Schema validation failed. Halting pipeline.")
        return False

    # 2. Strict Inner Join by Composite Key
    df_merged = pd.merge(df_ecg, df_ppg, on=COMPOSITE_KEY, how="inner", suffixes=("_ecg", "_ppg"))

    # Consolidate hr_ref
    if "hr_ref_ecg" in df_merged.columns:
        df_merged["hr_ref"] = df_merged["hr_ref_ecg"]
        df_merged.drop(columns=["hr_ref_ecg", "hr_ref_ppg"], inplace=True)

    # 3. Print Coverage Report
    n_ecg = len(df_ecg)
    n_ppg = len(df_ppg)
    n_merged = len(df_merged)
    
    print("\n" + "="*60)
    print("         MULTIMODAL DATA INTEGRATION COVERAGE REPORT")
    print("="*60)
    print(f"• Total Student 1 (ECG) windows : {n_ecg}")
    print(f"• Total Student 2 (PPG) windows : {n_ppg}")
    print(f"• Successfully aligned windows  : {n_merged}")
    print(f"• Unmatched ECG windows dropped : {n_ecg - n_merged}")
    print(f"• Unmatched PPG windows dropped : {n_ppg - n_merged}")
    print("="*60 + "\n")

    # 4. Export Multimodal Windows
    os.makedirs(os.path.dirname(out_multimodal_path) or ".", exist_ok=True)
    df_merged.to_parquet(out_multimodal_path, index=False)
    logging.info(f" Multimodal merged table saved to: {out_multimodal_path}")

    # 5. Build Initialized 'student3_router/predictions_all_methods.parquet'
    os.makedirs(out_router_dir, exist_ok=True)
    out_predictions_path = os.path.join(out_router_dir, "predictions_all_methods.parquet")

    df_preds = df_merged[COMPOSITE_KEY + ["hr_ref", "hr_ecg", "hr_ppg", "err_ecg", "err_ppg", "predicted_err_ecg", "predicted_err_ppg"]].copy()
    
    # Fill baseline columns
    df_preds["fold_id"] = 0
    df_preds["hr_fixed_fusion"] = np.round((df_preds["hr_ecg"] + df_preds["hr_ppg"]) / 2.0, 2)
    df_preds["hr_sqi_weighted"] = df_preds["hr_fixed_fusion"]
    df_preds["hr_learned_fusion"] = df_preds["hr_fixed_fusion"]
    
    # Initial Router Logic
    select_ecg = df_preds["predicted_err_ecg"] <= df_preds["predicted_err_ppg"]
    df_preds["action_router"] = np.where(select_ecg, "ECG", "PPG")
    df_preds["hr_router"] = np.where(select_ecg, df_preds["hr_ecg"], df_preds["hr_ppg"])
    df_preds["err_router"] = np.where(select_ecg, df_preds["err_ecg"], df_preds["err_ppg"])
    
    # Abstain flag
    abstain_mask = (df_preds["predicted_err_ecg"] > 15.0) & (df_preds["predicted_err_ppg"] > 15.0)
    df_preds["flag_abstain"] = np.where(abstain_mask, 1, 0)
    df_preds.loc[abstain_mask, "action_router"] = "ABSTAIN"

    df_preds.to_parquet(out_predictions_path, index=False)
    logging.info(f" Initialized router predictions saved to: {out_predictions_path}")

    return True

if __name__ == "__main__":
    run_handoff_check()
