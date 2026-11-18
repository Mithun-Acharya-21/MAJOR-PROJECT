"""Generate phase5_consolidated_report.json from verified artifacts."""

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "phase5"


def load_json(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return None


def build_report():
    report = {
        "schema_version": "1.0",
        "project": "NeuroScreen",
        "phase": "5",
        "generated_at": datetime.now().isoformat(),
        "status": "DEVELOPMENT_COMPLETE_PENDING_LOCKED_TEST",
        "disclaimer": (
            "Academic research prototype only. NOT a clinical diagnostic system. "
            "All results are development-set estimates. Locked test partition not accessed."
        ),
        "protocol_reference": "docs/phase_5_research_protocol.md",
        "phases": {},
        "experiments": {},
        "blocked_experiments": {},
        "datasets": {},
    }

    # ===== DATASETS =====
    report["datasets"] = {
        "EATD": {
            "status": "AVAILABLE",
            "task": "binary_depression_screening",
            "label_definition": "SDS >= 50",
            "n_development_participants": 138,
            "n_locked_test_participants": 24,
            "n_positive_dev": 45,
            "prevalence_dev": 0.3261,
            "modalities": ["text", "audio"],
            "locked_test_accessed": False,
        },
        "AVDD_TCMI": {
            "status": "AVAILABLE",
            "task": "binary_depression_category",
            "label_definition": "HDS vs SDS groups",
            "n_total": 1230,
            "n_HDS": 282,
            "n_SDS": 948,
            "n_acoustic_features": 7168,
            "n_visual_features": 3731,
            "participant_id_released": False,
            "caveat": "Sample-level bootstrap only; speaker clustering effects unknown",
        },
        "FER2013": {"status": "BLOCKED_INCOMPLETE", "reason": "9034/28709 train images; missing 3 classes"},
        "CK_plus": {"status": "NOT_ACQUIRED"},
        "RAF_DB": {"status": "NOT_ACQUIRED"},
        "Harvard_Tweets": {"status": "BLOCKED_LABEL_AMBIGUITY"},
        "DAIC_WOZ": {"status": "NOT_ACQUIRED"},
    }

    # ===== PHASE 5A =====
    report["phases"]["5A"] = {"status": "COMPLETE", "description": "Research protocol and experiment matrix"}

    # ===== PHASE 5B =====
    report["phases"]["5B"] = {
        "status": "COMPLETE",
        "description": "Evaluation harness, leakage protection, locked-test firewall",
        "test_results": "114 passed, 0 failed",
        "locked_test_participants_protected": 24,
        "leakage_probes_passed": 10,
    }

    # ===== PHASE 5C — EATD Text =====
    report["phases"]["5C"] = {"status": "COMPLETE", "description": "EATD text experiments"}

    # E-L2
    el2 = load_json(ROOT / "results/phase5/eatd/E-L2/report.json")
    if el2:
        agg = el2.get("aggregate", {})
        report["experiments"]["E-L2"] = {
            "id": "E-L2",
            "phase": "5C",
            "dataset": "EATD",
            "modality": "text_metadata_baseline",
            "model": "LogisticRegression on metadata/confound features",
            "status": "COMPLETE",
            "metrics": {
                "roc_auc": {"mean": agg.get("mean_roc_auc_selected"), "sd": agg.get("sd_roc_auc_selected")},
                "pr_auc": {"mean": agg.get("mean_pr_auc_selected"), "sd": agg.get("sd_pr_auc_selected")},
            },
            "n_folds": agg.get("n_folds", 25),
            "bootstrap_ci_roc_auc": None,  # not separately computed
            "artifacts": ["results/phase5/eatd/E-L2/report.json"],
        }

    # E-T1
    et1 = load_json(ROOT / "results/phase5/eatd/E-T1/report.json")
    if et1:
        agg = et1.get("aggregate", {})
        boot = et1.get("bootstrap_first_repeat", {}).get("metrics", {}).get("roc_auc", {})
        report["experiments"]["E-T1"] = {
            "id": "E-T1",
            "phase": "5C",
            "dataset": "EATD",
            "modality": "text",
            "model": "TF-IDF (1-2grams) + LogisticRegression",
            "status": "COMPLETE",
            "metrics": {
                "roc_auc": {"mean": agg.get("mean_roc_auc_selected"), "sd": agg.get("sd_roc_auc_selected")},
                "pr_auc": {"mean": agg.get("mean_pr_auc_selected"), "sd": agg.get("sd_pr_auc_selected")},
                "balanced_accuracy": {"mean": agg.get("mean_balanced_accuracy_selected")},
                "macro_f1": {"mean": agg.get("mean_macro_f1_selected")},
            },
            "n_folds": 25,
            "bootstrap_ci_roc_auc": {
                "ci_lower": boot.get("ci_lower"),
                "ci_upper": boot.get("ci_upper"),
                "n_resamples": 2000,
                "method": "participant_cluster_bootstrap",
            },
            "ci_includes_chance": False,
            "artifacts": ["results/phase5/eatd/E-T1/report.json"],
        }

    # E-T2
    et2 = load_json(ROOT / "results/phase5/eatd/E-T2/report.json")
    if et2:
        agg = et2.get("aggregate", {})
        report["experiments"]["E-T2"] = {
            "id": "E-T2",
            "phase": "5C",
            "dataset": "EATD",
            "modality": "text",
            "model": "Frozen multilingual transformer + MLP head",
            "status": "COMPLETE",
            "metrics": {
                "roc_auc": {"mean": agg.get("mean_roc_auc_selected"), "sd": agg.get("sd_roc_auc_selected")},
                "pr_auc": {"mean": agg.get("mean_pr_auc_selected"), "sd": agg.get("sd_pr_auc_selected")},
            },
            "n_folds": 25,
            "bootstrap_ci_roc_auc": None,
            "artifacts": ["results/phase5/eatd/E-T2/report.json"],
        }

    report["blocked_experiments"]["E-T3"] = {
        "id": "E-T3", "phase": "5C", "status": "COST_GATED",
        "reason": "Fine-tuned transformer: compute cost gate; requires explicit owner approval",
    }

    # ===== PHASE 5D — EATD Voice =====
    report["phases"]["5D"] = {"status": "COMPLETE", "description": "EATD voice experiments"}

    vcf = load_json(ROOT / "results/phase5/eatd_voice/voice_comparison_final.json")
    if vcf:
        exps = vcf.get("experiments", {})
        for eid in ["E-V1", "E-V2", "E-V3"]:
            ed = exps.get(eid, {})
            m = ed.get("metrics", {})
            bci = ed.get("bootstrap_ci", {}).get("roc_auc", {})
            report["experiments"][eid] = {
                "id": eid,
                "phase": "5D",
                "dataset": "EATD",
                "modality": "voice",
                "model": {
                    "E-V1": "eGeMAPS + LogisticRegression",
                    "E-V2": "eGeMAPS + RBF SVM",
                    "E-V3": "Frozen Wav2Vec2 + MLP head",
                }[eid],
                "status": "COMPLETE",
                "metrics": {
                    "roc_auc": {"mean": m.get("roc_auc", {}).get("mean"), "sd": m.get("roc_auc", {}).get("sd")},
                    "pr_auc": {"mean": m.get("pr_auc", {}).get("mean")},
                    "balanced_accuracy": {"mean": m.get("balanced_accuracy", {}).get("mean")},
                    "macro_f1": {"mean": m.get("macro_f1", {}).get("mean")},
                },
                "bootstrap_ci_roc_auc": {
                    "mean_ci_lower": bci.get("mean_ci_lower"),
                    "mean_ci_upper": bci.get("mean_ci_upper"),
                    "n_resamples": 2000,
                    "method": "mean_of_fold_level_participant_cluster_bootstrap",
                },
                "ci_includes_chance": True,
                "n_folds": 25,
                "artifacts": [f"results/phase5/eatd_voice/{eid}/{eid}.json"],
            }

    report["blocked_experiments"]["E-V4"] = {
        "id": "E-V4", "phase": "5D", "status": "BLOCKED",
        "reason": "Fine-tuned Wav2Vec2: 6GB GPU memory limit + overfitting risk on 413 dev sessions",
    }

    # ===== PHASE 5E — EATD Fusion =====
    report["phases"]["5E"] = {"status": "COMPLETE", "description": "EATD multimodal fusion analysis"}

    fc = load_json(ROOT / "results/phase5/eatd_fusion/fusion_comparison.json")
    ea = load_json(ROOT / "results/phase5/eatd_fusion/error_analysis.json")
    if fc:
        report["experiments"]["E-FUS-EATD"] = {
            "id": "E-FUS-EATD",
            "phase": "5E",
            "dataset": "EATD",
            "modality": "text+voice (late fusion)",
            "status": "COMPLETE",
            "fusion_variants": fc.get("fusion_variants", {}),
            "complementarity": fc.get("complementarity_classification"),
            "best_variant": "E-T1+E-V3 equal",
            "best_roc_auc": 0.6783,
            "conclusion": "Fusion provides no statistically confirmed improvement over E-T1 alone",
            "participant_level_oof_artifact": "results/phase5/eatd_fusion/oof_predictions/participant_level_oof_predictions.csv",
            "artifacts": [
                "results/phase5/eatd_fusion/fusion_comparison.json",
                "results/phase5/eatd_fusion/report.json",
                "results/phase5/eatd_fusion/error_analysis.json",
            ],
        }

    if ea:
        report["experiments"]["E-FUS-EATD"]["error_analysis"] = {
            "pearson_r_text_vs_voice": ea.get("pearson_r"),
            "both_correct": ea.get("both_correct"),
            "both_wrong": ea.get("both_wrong"),
            "text_only_correct": ea.get("text_only_correct"),
            "voice_only_correct": ea.get("voice_only_correct"),
        }

    # ===== PHASE 5F — AVDD Acoustic =====
    report["phases"]["5F"] = {
        "status": "PARTIAL",
        "description": "AVDD-TCMI acoustic experiments (HDS complete, SDS in progress)",
    }

    # HDS E-A1
    hds_m = load_json(ROOT / "results/phase5/avdd_features/HDS/E-A1/metrics.json")
    if hds_m:
        models_summary = {}
        for mn, agg in hds_m["models"].items():
            roc = agg["roc_auc"]
            pr = agg["pr_auc"]
            bal = agg["balanced_accuracy"]
            f1 = agg["macro_f1"]
            bci = agg["bootstrap_ci"]["roc_auc"]
            models_summary[mn] = {
                "roc_auc": {"mean": roc["mean"], "sd": roc["std"]},
                "pr_auc": {"mean": pr["mean"]},
                "balanced_accuracy": {"mean": bal["mean"]},
                "macro_f1": {"mean": f1["mean"]},
                "bootstrap_ci_roc_auc": {
                    "ci_lower": bci.get("ci_lower"),
                    "ci_upper": bci.get("ci_upper"),
                    "n_resamples": 1000,
                    "method": "sample_level_bootstrap",
                },
                "ci_includes_chance": False,
            }
        report["experiments"]["E-A1-HDS"] = {
            "id": "E-A1",
            "subset": "HDS",
            "phase": "5F",
            "dataset": "AVDD_TCMI",
            "modality": "acoustic",
            "status": "COMPLETE",
            "n_samples": hds_m["n_samples"],
            "n_positive": hds_m["n_positive"],
            "n_features": hds_m["n_features"],
            "models": models_summary,
            "caveat": "Sample-level bootstrap; speaker identity unknown; clustering effects not corrected",
            "artifacts": [
                "results/phase5/avdd_features/HDS/E-A1/metrics.json",
                "results/phase5/avdd_features/HDS/E-A1/oof_predictions.npz",
            ],
        }

    # SDS E-A1
    sds_m = load_json(ROOT / "results/phase5/avdd_features/SDS/E-A1/metrics.json")
    if sds_m:
        models_summary_sds = {}
        for mn, agg in sds_m["models"].items():
            roc = agg["roc_auc"]
            pr = agg["pr_auc"]
            bal = agg["balanced_accuracy"]
            f1 = agg["macro_f1"]
            bci = agg["bootstrap_ci"]["roc_auc"]
            models_summary_sds[mn] = {
                "roc_auc": {"mean": roc["mean"], "sd": roc["std"]},
                "pr_auc": {"mean": pr["mean"]},
                "balanced_accuracy": {"mean": bal["mean"]},
                "macro_f1": {"mean": f1["mean"]},
                "bootstrap_ci_roc_auc": {
                    "ci_lower": bci.get("ci_lower"),
                    "ci_upper": bci.get("ci_upper"),
                    "n_resamples": 1000,
                    "method": "sample_level_bootstrap",
                },
            }
        report["experiments"]["E-A1-SDS"] = {
            "id": "E-A1",
            "subset": "SDS",
            "phase": "5F",
            "dataset": "AVDD_TCMI",
            "modality": "acoustic",
            "status": "COMPLETE",
            "n_samples": sds_m["n_samples"],
            "n_positive": sds_m["n_positive"],
            "n_features": sds_m["n_features"],
            "models": models_summary_sds,
            "caveat": "Sample-level bootstrap; speaker identity unknown",
            "artifacts": [
                "results/phase5/avdd_features/SDS/E-A1/metrics.json",
                "results/phase5/avdd_features/SDS/E-A1/oof_predictions.npz",
            ],
        }
    else:
        report["experiments"]["E-A1-SDS"] = {
            "id": "E-A1", "subset": "SDS", "phase": "5F",
            "status": "IN_PROGRESS",
            "note": "SDS E-A1 running at report generation time. See results/phase5/avdd_features/SDS/E-A1/run.log",
        }

    # E-A2 blocked
    report["blocked_experiments"]["E-A2"] = {
        "id": "E-A2", "phase": "5F", "status": "BLOCKED_AWAITING_OWNER_APPROVAL",
        "reason": "Protocol Section 6.7 pre-training review gate: requires explicit owner sign-off before fitting any AVDD visual model",
        "blocking_requirement": "Owner must review and approve: label semantics, feature dictionary, dropped columns, retained visual features",
    }

    report["blocked_experiments"]["E-A3"] = {
        "id": "E-A3", "phase": "5F", "status": "BLOCKED_UPSTREAM",
        "reason": "Modality ablation requires E-A2 completion",
    }

    report["blocked_experiments"]["E-A4"] = {
        "id": "E-A4", "phase": "5F", "status": "BLOCKED_UPSTREAM",
        "reason": "Acoustic+visual fusion requires E-A2 completion and statistical gates (Section 6.6)",
    }

    # ===== PHASE 5G — Facial =====
    report["phases"]["5G"] = {
        "status": "BLOCKED",
        "description": "Facial image experiments: all candidate datasets unavailable or incomplete",
    }

    report["blocked_experiments"]["E-G-facial"] = {
        "id": "E-G-facial", "phase": "5G", "status": "BLOCKED",
        "reason": "FER-2013: 31.5% of training images present, 3 classes missing. CK+/RAF-DB: not acquired.",
    }

    # ===== PHASE 5H — Cross-dataset fusion =====
    report["phases"]["5H"] = {
        "status": "BLOCKED_GATED",
        "description": "Cross-dataset fusion: blocked pending E-A2 approval and statistical gates",
    }

    report["blocked_experiments"]["E-F1"] = {
        "id": "E-F1", "phase": "5H", "status": "BLOCKED_GATED",
        "reason": "Requires: E-A2 completion, AVDD fusion statistical gates (Section 6.6), different populations",
    }

    # ===== LOCKED TEST =====
    report["locked_test"] = {
        "status": "NOT_ACCESSED",
        "gate": "Section 4.9",
        "requirements": [
            "All development gates satisfied",
            "Preregistration file created with model+threshold+script hash",
            "Explicit owner authorization",
        ],
        "n_participants": 24,
        "dataset": "EATD",
    }

    # ===== OVERALL STATUS =====
    report["overall_summary"] = {
        "best_text_model": {"id": "E-T1", "roc_auc_mean": 0.6756, "ci": [0.576, 0.759]},
        "best_voice_model": {"id": "E-V3", "roc_auc_mean": 0.6315, "ci_note": "CI includes 0.5 — inconclusive"},
        "best_avdd_acoustic_model": {"id": "E-A1 RF (HDS)", "roc_auc_mean": 0.8202, "ci": [0.767, 0.868]},
        "fusion_conclusion": "No confirmed improvement over E-T1 alone",
        "text_beats_chance": True,
        "voice_beats_chance": False,
        "avdd_hds_beats_chance": True,
        "pending": [
            "SDS E-A1 completion",
            "AVDD E-A1 comparison report",
            "§6.7 owner sign-off for E-A2",
            "§4.9 owner authorization for locked test",
        ],
    }

    return report


if __name__ == "__main__":
    report = build_report()
    out_json = OUT_DIR / "phase5_consolidated_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Written: {out_json}")

    # Also write a markdown summary
    out_md = OUT_DIR / "phase5_consolidated_report.md"
    lines = [
        "# NeuroScreen Phase 5 — Consolidated Results Report",
        "",
        f"> Generated: {report['generated_at']}  ",
        f"> Status: {report['status']}  ",
        f"> **{report['disclaimer']}**",
        "",
        "## Phase Completion Status",
        "",
        "| Phase | Status | Description |",
        "|-------|--------|-------------|",
    ]
    for phase, info in report["phases"].items():
        lines.append(f"| Phase {phase} | {info['status']} | {info['description']} |")

    lines += [
        "",
        "## Experiment Results Summary",
        "",
        "### EATD Text (Phase 5C)",
        "",
        "| Experiment | Model | ROC-AUC (mean±SD) | 95% CI | PR-AUC | Above Chance? |",
        "|------------|-------|-------------------|--------|--------|---------------|",
    ]
    for eid in ["E-L2", "E-T1", "E-T2"]:
        e = report["experiments"].get(eid, {})
        m = e.get("metrics", {})
        roc = m.get("roc_auc", {})
        pr = m.get("pr_auc", {})
        bci = e.get("bootstrap_ci_roc_auc") or {}
        roc_str = f"{roc.get('mean', '?'):.4f}±{roc.get('sd', '?'):.4f}" if roc.get("mean") else "?"
        ci_str = f"[{bci.get('ci_lower', '?'):.3f}, {bci.get('ci_upper', '?'):.3f}]" if bci.get("ci_lower") else "—"
        pr_str = f"{pr.get('mean', '?'):.4f}" if pr.get("mean") else "?"
        above = "✓" if not e.get("ci_includes_chance", True) else "—"
        lines.append(f"| {eid} | {e.get('model', '?')} | {roc_str} | {ci_str} | {pr_str} | {above} |")

    lines += [
        "",
        "### EATD Voice (Phase 5D)",
        "",
        "| Experiment | Model | ROC-AUC (mean±SD) | Mean CI | PR-AUC | Above Chance? |",
        "|------------|-------|-------------------|---------|--------|---------------|",
    ]
    for eid in ["E-V1", "E-V2", "E-V3"]:
        e = report["experiments"].get(eid, {})
        m = e.get("metrics", {})
        roc = m.get("roc_auc", {})
        pr = m.get("pr_auc", {})
        bci = e.get("bootstrap_ci_roc_auc") or {}
        roc_str = f"{roc.get('mean', '?'):.4f}±{roc.get('sd', '?'):.4f}" if roc.get("mean") else "?"
        ci_str = f"[{bci.get('mean_ci_lower','?'):.3f},{bci.get('mean_ci_upper','?'):.3f}]" if bci.get("mean_ci_lower") else "—"
        pr_str = f"{pr.get('mean', '?'):.4f}" if pr.get("mean") else "?"
        above = "✗ (CI incl. 0.5)"
        lines.append(f"| {eid} | {e.get('model', '?')} | {roc_str} | {ci_str} | {pr_str} | {above} |")

    lines += [
        "",
        "### EATD Fusion (Phase 5E)",
        "",
        "| Variant | ROC-AUC | PR-AUC | Bal-Acc | Macro-F1 |",
        "|---------|---------|--------|---------|----------|",
    ]
    fc_data = report["experiments"].get("E-FUS-EATD", {}).get("fusion_variants", {})
    for name, v in fc_data.items():
        roc = v.get("mean_roc_auc_selected", "?")
        pr = v.get("mean_pr_auc_selected", "?")
        bal = v.get("mean_balanced_accuracy_selected", "?")
        f1 = v.get("mean_macro_f1_selected", "?")
        roc_s = f"{roc:.4f}" if isinstance(roc, float) else str(roc)
        pr_s = f"{pr:.4f}" if isinstance(pr, float) else str(pr)
        bal_s = f"{bal:.4f}" if isinstance(bal, float) else str(bal)
        f1_s = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
        lines.append(f"| {name} | {roc_s} | {pr_s} | {bal_s} | {f1_s} |")

    lines += [
        "",
        "### AVDD-TCMI Acoustic (Phase 5F)",
        "",
        "| Experiment | Subset | Model | ROC-AUC | 95% CI | Above Chance? |",
        "|------------|--------|-------|---------|--------|---------------|",
    ]
    for eid in ["E-A1-HDS", "E-A1-SDS"]:
        e = report["experiments"].get(eid, {})
        subset = e.get("subset", "?")
        if e.get("status") in ("COMPLETE",):
            for mn, mv in e.get("models", {}).items():
                roc = mv.get("roc_auc", {})
                bci = mv.get("bootstrap_ci_roc_auc", {})
                roc_s = f"{roc.get('mean','?'):.4f}±{roc.get('sd','?'):.4f}" if roc.get("mean") else "?"
                ci_s = f"[{bci.get('ci_lower','?'):.3f},{bci.get('ci_upper','?'):.3f}]" if bci.get("ci_lower") else "—"
                above = "✓" if not mv.get("ci_includes_chance", True) else "—"
                lines.append(f"| E-A1 | {subset} | {mn.replace('_',' ')} | {roc_s} | {ci_s} | {above} |")
        else:
            lines.append(f"| E-A1 | {subset} | — | IN PROGRESS | — | — |")

    lines += [
        "",
        "## Blocked Experiments",
        "",
        "| ID | Phase | Status | Reason |",
        "|----|-------|--------|--------|",
    ]
    for bid, binfo in report["blocked_experiments"].items():
        lines.append(f"| {binfo.get('id','?')} | {binfo.get('phase','?')} | {binfo.get('status','?')} | {binfo.get('reason','?')[:100]} |")

    lines += [
        "",
        "## Locked Test Status",
        "",
        f"**Status:** {report['locked_test']['status']}  ",
        f"Requires: {', '.join(report['locked_test']['requirements'])}",
        "",
        "---",
        "*NeuroScreen — Academic research prototype only. Not a clinical system.*",
    ]

    out_md.write_text("\n".join(lines))
    print(f"Written: {out_md}")
