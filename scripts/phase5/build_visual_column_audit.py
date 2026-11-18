"""Generate documented column classification report for AVDD-TCMI visual features.

Audits all 538 OpenFace visual columns according to docs/phase_5_research_protocol.md Section 6.4.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_JSON = ROOT / "data" / "reports" / "avdd_tcmi_visual_validation.json"
OUT_JSON = ROOT / "data" / "reports" / "avdd_visual_feature_column_audit.json"

DROPPED_CONSTANT = {"face_id", "success"}
EXCLUDED_METADATA = {"frame", "timestamp", "confidence"}


def classify_column(col_name: str) -> dict:
    if col_name in DROPPED_CONSTANT:
        return {
            "column_name": col_name,
            "data_type": "int64/float64",
            "feature_family": "constant_metadata",
            "retained": False,
            "excluded": True,
            "reason_for_exclusion": "Constant column across the entire corpus (std = 0); carries zero predictive variance and cannot be standardized.",
            "evidence_source": "docs/phase_5_research_protocol.md Section 6.4; data/reports/avdd_tcmi_visual_validation.json",
            "openface_semantics": f"Tracking status flag ({col_name}); identical value across all 1,230 files and 3.1M frames."
        }
    if col_name in {"frame", "timestamp"}:
        return {
            "column_name": col_name,
            "data_type": "int64/float64",
            "feature_family": "temporal_metadata",
            "retained": False,
            "excluded": True,
            "reason_for_exclusion": "Encodes recording length and sample frame rate, not facial behavior. Including it would allow model to exploit session duration as a shortcut.",
            "evidence_source": "docs/phase_5_research_protocol.md Section 6.4",
            "openface_semantics": "Sequential video frame index or absolute video timestamp in seconds."
        }
    if col_name == "confidence":
        return {
            "column_name": col_name,
            "data_type": "float64",
            "feature_family": "quality_control",
            "retained": False,
            "excluded": True,
            "reason_for_exclusion": "Tracking confidence score. Excluded from primary model inputs to prevent shortcut learning on recording condition artifacts; retained separately as QC covariate (confidence_mean).",
            "evidence_source": "docs/phase_5_research_protocol.md Section 6.4",
            "openface_semantics": "OpenFace landmark detection confidence (0.0 to 1.0)."
        }
    if col_name.startswith("gaze_"):
        return {
            "column_name": col_name,
            "data_type": "float64",
            "feature_family": "gaze",
            "retained": True,
            "excluded": False,
            "reason_for_exclusion": None,
            "evidence_source": "docs/phase_5_research_protocol.md Section 6.4; OpenFace 2.0 gaze estimation documentation",
            "openface_semantics": "Eye gaze direction vectors (gaze_0 for left eye, gaze_1 for right eye) and 2D gaze angles in radians."
        }
    if col_name.startswith("pose_"):
        return {
            "column_name": col_name,
            "data_type": "float64",
            "feature_family": "head_pose",
            "retained": True,
            "excluded": False,
            "reason_for_exclusion": None,
            "evidence_source": "docs/phase_5_research_protocol.md Section 6.4; OpenFace 2.0 head pose documentation",
            "openface_semantics": "Head position translation (Tx, Ty, Tz in millimeters) and rotation (Rx, Ry, Rz in radians pitch, yaw, roll)."
        }
    if col_name.startswith("AU"):
        is_reg = col_name.endswith("_r")
        return {
            "column_name": col_name,
            "data_type": "float64",
            "feature_family": "action_units",
            "retained": True,
            "excluded": False,
            "reason_for_exclusion": None,
            "evidence_source": "docs/phase_5_research_protocol.md Section 6.4; FACS (Ekman & Friesen); OpenFace AU recognition",
            "openface_semantics": f"Facial Action Unit {'intensity (0.0-5.0 continuous)' if is_reg else 'presence (0 or 1 classification)'} for {col_name.split('_')[0]}."
        }
    if col_name.startswith("eye_lmk_"):
        is_3d = col_name.startswith("eye_lmk_X_") or col_name.startswith("eye_lmk_Y_") or col_name.startswith("eye_lmk_Z_")
        return {
            "column_name": col_name,
            "data_type": "float64",
            "feature_family": "eye_landmarks",
            "retained": True,
            "excluded": False,
            "reason_for_exclusion": None,
            "flagged_caveat": "Raw landmark coordinate: strongly confounds subject identity, camera distance, and head pose.",
            "evidence_source": "docs/phase_5_research_protocol.md Section 6.4; OpenFace 56-point eye landmark model",
            "openface_semantics": f"{'3D eye landmark coordinate in camera coordinate space (mm)' if is_3d else '2D eye landmark coordinate in pixel image coordinates'}."
        }
    if any(col_name.startswith(f"{dim}_") for dim in ["X", "Y", "Z"]):
        return {
            "column_name": col_name,
            "data_type": "float64",
            "feature_family": "facial_landmarks",
            "retained": True,
            "excluded": False,
            "reason_for_exclusion": None,
            "flagged_caveat": "Raw 3D landmark coordinate: absolute values mix head pose, camera geometry, and facial morphology (identity).",
            "evidence_source": "docs/phase_5_research_protocol.md Section 6.4; OpenFace 68-point 3D landmark detector",
            "openface_semantics": f"3D location of facial landmark {col_name.split('_')[1]} in camera coordinates (mm)."
        }
    return {
        "column_name": col_name,
        "data_type": "float64",
        "feature_family": "other",
        "retained": True,
        "excluded": False,
        "reason_for_exclusion": None,
        "evidence_source": "unknown",
        "openface_semantics": "unclassified column"
    }


def main():
    val_data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    raw_columns = val_data["column_names"]

    columns_audit = [classify_column(c) for c in raw_columns]

    family_counts = {}
    for c in columns_audit:
        f = c["feature_family"]
        family_counts[f] = family_counts.get(f, 0) + 1

    retained_cols = [c["column_name"] for c in columns_audit if c["retained"]]
    dropped_cols = [c["column_name"] for c in columns_audit if c["excluded"]]

    report = {
        "schema_version": "1.0",
        "dataset": "AVDD-TCMI",
        "modality": "visual",
        "total_columns": len(raw_columns),
        "retained_columns_count": len(retained_cols),
        "excluded_columns_count": len(dropped_cols),
        "family_summary": family_counts,
        "policy_reference": "docs/phase_5_research_protocol.md Section 6.4",
        "feature_count_per_sample_after_aggregation": len(retained_cols) * 7,
        "aggregation_functionals": ["mean", "std", "median", "iqr", "p05", "p95", "slope"],
        "columns": columns_audit
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audit written to: {OUT_JSON}")
    print(f"Total: {len(raw_columns)}, Retained: {len(retained_cols)}, Excluded: {len(dropped_cols)}")
    print("Families:", family_counts)


if __name__ == "__main__":
    main()
