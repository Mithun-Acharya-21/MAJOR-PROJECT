#!/usr/bin/env python3
"""
DAIC-WOZ Participant-Level Split Generator & Leakage Auditor
NeuroScreen Research Pipeline - Phase 3 Splitting

Generates:
    - data/splits/daic_woz/train.csv
    - data/splits/daic_woz/validation.csv
    - data/splits/daic_woz/test.csv
    - data/splits/daic_woz/split_report.json
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

RANDOM_SEED = 42


def create_splits(manifest_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "split_report.json"

    if not manifest_path.exists():
        print(f"[-] Participant manifest not found at: {manifest_path}")
        return False

    df = pd.read_csv(manifest_path)
    if df.empty:
        print("[!] Participant manifest is currently empty (Awaiting official DAIC-WOZ data).")
        empty_report = {
            "status": "BLOCKED_AWAITING_DATA",
            "message": "Zero records found in participant manifest. Run inspect_dataset.py after unzipping official DAIC-WOZ data.",
            "total_participants": 0,
            "train_count": 0,
            "validation_count": 0,
            "test_count": 0,
            "leakage_detected": False
        }
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(empty_report, f, indent=2)
        # Create empty placeholder files with schema
        for name in ["train.csv", "validation.csv", "test.csv"]:
            pd.DataFrame(columns=df.columns).to_csv(output_dir / name, index=False)
        return False

    # Check for official AVEC partition files in raw_dir
    raw_dir = Path("data/raw/daic_woz")
    train_avec = list(raw_dir.glob("train_split*.csv"))
    dev_avec = list(raw_dir.glob("dev_split*.csv"))
    test_avec = list(raw_dir.glob("test_split*.csv"))

    if train_avec and dev_avec and test_avec:
        print("[*] Official AVEC 2016/2017 partition files detected. Preserving official benchmark splits.")
        df_train_part = pd.read_csv(train_avec[0])
        df_dev_part = pd.read_csv(dev_avec[0])
        df_test_part = pd.read_csv(test_avec[0])

        id_col = "Participant_ID" if "Participant_ID" in df_train_part.columns else "participant_id"
        train_pids = set(df_train_part[id_col].astype(int))
        val_pids = set(df_dev_part[id_col].astype(int))
        test_pids = set(df_test_part[id_col].astype(int))

        df_train = df[df["participant_id"].astype(int).isin(train_pids)].copy()
        df_val = df[df["participant_id"].astype(int).isin(val_pids)].copy()
        df_test = df[df["participant_id"].astype(int).isin(test_pids)].copy()
        strategy = "Official AVEC 2016/2017 Challenge Partitions"
    else:
        print("[*] Performing 70/15/15 stratified participant-level split with random seed 42.")
        strategy = "70% Train / 15% Validation / 15% Test Stratified by PHQ-8 Binary Label"

        # Separate labeled from unlabeled
        labeled_df = df[df["phq8_binary"].notna()].copy()
        unlabeled_df = df[df["phq8_binary"].isna()].copy()

        if len(labeled_df) < 10:
            print("[-] Insufficient labeled samples for stratified splitting.")
            return False

        # First split: 70% train vs 30% temp (val+test)
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
        y = labeled_df["phq8_binary"].astype(int)
        train_idx, temp_idx = next(sss1.split(labeled_df, y))

        df_train = labeled_df.iloc[train_idx].copy()
        temp_df = labeled_df.iloc[temp_idx].copy()

        # Second split: 50% val vs 50% test of the temp set (= 15% / 15% of total)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
        y_temp = temp_df["phq8_binary"].astype(int)
        val_idx, test_idx = next(sss2.split(temp_df, y_temp))

        df_val = temp_df.iloc[val_idx].copy()
        df_test = temp_df.iloc[test_idx].copy()

        if not unlabeled_df.empty:
            print(f"[!] Warning: {len(unlabeled_df)} unlabeled participants excluded from evaluation splits.")

    # Strict Leakage Verification
    train_ids = set(df_train["participant_id"])
    val_ids = set(df_val["participant_id"])
    test_ids = set(df_test["participant_id"])

    leak_train_val = train_ids.intersection(val_ids)
    leak_train_test = train_ids.intersection(test_ids)
    leak_val_test = val_ids.intersection(test_ids)

    leakage_detected = bool(leak_train_val or leak_train_test or leak_val_test)
    if leakage_detected:
        raise ValueError(f"CRITICAL: Participant leakage detected! Overlaps: {leak_train_val}, {leak_train_test}, {leak_val_test}")

    # Save CSVs
    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "validation.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)

    split_report = {
        "status": "VALIDATED_LEAK_FREE",
        "split_strategy": strategy,
        "random_seed": RANDOM_SEED,
        "total_participants": len(df),
        "train": {
            "count": len(df_train),
            "depressed": int((df_train["phq8_binary"] == 1).sum()),
            "control": int((df_train["phq8_binary"] == 0).sum())
        },
        "validation": {
            "count": len(df_val),
            "depressed": int((df_val["phq8_binary"] == 1).sum()),
            "control": int((df_val["phq8_binary"] == 0).sum())
        },
        "test": {
            "count": len(df_test),
            "depressed": int((df_test["phq8_binary"] == 1).sum()),
            "control": int((df_test["phq8_binary"] == 0).sum())
        },
        "leakage_verification": {
            "train_val_overlap": len(leak_train_val),
            "train_test_overlap": len(leak_train_test),
            "val_test_overlap": len(leak_val_test),
            "zero_leakage_certified": not leakage_detected
        }
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=2)

    print(f"[+] Splits successfully generated and leak-free certified:")
    print(f"    - Train: {len(df_train)} participants")
    print(f"    - Validation: {len(df_val)} participants")
    print(f"    - Test (Locked): {len(df_test)} participants")
    return True


if __name__ == "__main__":
    manifest_csv = Path("data/manifests/daic_woz/participant_manifest.csv")
    splits_dir = Path("data/splits/daic_woz")
    create_splits(manifest_csv, splits_dir)
