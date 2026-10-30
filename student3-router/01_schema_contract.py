"""
BioSense-AI - Schema Contract Definition (01_schema_contract.py)
----------------------------------------------------------------
Defines the official data integration contract between Student 1 (ECG),
Student 2 (PPG), Student 3 (Reliability Router), and Student 4 (Evaluation).
"""

from dataclasses import dataclass, field
from typing import List
import pandas as pd

# 1. Composite Key (Mandatory for ALL tables)
COMPOSITE_KEY: List[str] = [
    "subject_id",
    "activity",
    "window_start_s",
    "window_end_s"
]

# 2. Student 1 (ECG) Required Minimum Columns
REQUIRED_COLS_ECG: List[str] = [
    "subject_id",
    "activity",
    "window_start_s",
    "window_end_s",
    "hr_ref",
    "hr_ecg",
    "err_ecg",
    "predicted_err_ecg"
]

# 3. Student 2 (PPG) Required Minimum Columns
REQUIRED_COLS_PPG: List[str] = [
    "subject_id",
    "activity",
    "window_start_s",
    "window_end_s",
    "hr_ref",
    "hr_ppg",
    "err_ppg",
    "predicted_err_ppg"
]

# 4. Student 3 (Router Handoff Output) Required Columns
REQUIRED_COLS_ROUTER_OUTPUT: List[str] = [
    "subject_id",
    "activity",
    "window_start_s",
    "window_end_s",
    "fold_id",
    "hr_ref",
    "hr_ecg",
    "hr_ppg",
    "hr_fixed_fusion",
    "hr_sqi_weighted",
    "hr_learned_fusion",
    "hr_router",
    "err_ecg",
    "err_ppg",
    "err_router",
    "predicted_err_ecg",
    "predicted_err_ppg",
    "action_router",
    "flag_abstain"
]

@dataclass
class ValidationReport:
    is_valid: bool
    source_name: str
    missing_cols: List[str] = field(default_factory=list)
    null_key_count: int = 0
    duplicate_key_count: int = 0
    total_rows: int = 0

def validate_dataframe_schema(df: pd.DataFrame, source_name: str, required_cols: List[str]) -> ValidationReport:
    """
    Validates a DataFrame against the schema contract:
    - Checks required columns presence.
    - Checks null values in composite key.
    - Checks duplicate rows on composite key.
    """
    total_rows = len(df)
    missing = [col for col in required_cols if col not in df.columns]
    
    null_count = 0
    for col in COMPOSITE_KEY:
        if col in df.columns:
            null_count += int(df[col].isnull().sum())
            
    dup_count = 0
    if all(col in df.columns for col in COMPOSITE_KEY):
        dup_count = int(df.duplicated(subset=COMPOSITE_KEY).sum())
        
    is_valid = (len(missing) == 0) and (null_count == 0) and (dup_count == 0)
    
    return ValidationReport(
        is_valid=is_valid,
        source_name=source_name,
        missing_cols=missing,
        null_key_count=null_count,
        duplicate_key_count=dup_count,
        total_rows=total_rows
    )
