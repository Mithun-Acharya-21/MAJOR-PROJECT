"""Validate the Harvard Dataverse English depression tweets dataset.

Reads the .xlsx with the standard library only (xlsx is a zip of XML), so no new
dependency is added to the CUDA environment.

Checks: record count, label column and distribution, label values, empty text,
duplicates, per-user identifiers, language consistency (ASCII vs CJK), PII
markers (URLs, @mentions, emails, phone-like strings), text length statistics.

Two dataset-specific checks that matter for research use:
  - the sheet carries a side legend ('Class / Depressed / Non-depressed' with
    counts) that never states which numeric code means which class, so the
    label code -> class mapping is tested against a depression lexicon and the
depositor's own counts instead of being assumed;
  - the tweet text has been pre-processed: short token sequences with most
    function words removed and apostrophes replaced by 'xexx'. Word order is
    not destroyed (only 6.3% of rows happen to be alphabetically sorted, i.e.
    chance), but the surface form and all tweet metadata are gone, which has to
    be documented before any sequence model is applied.

Writes data/reports/harvard_tweets_validation.json and exits non-zero if any
hard check fails. Nothing is deleted or modified.

Usage:
  .venv/Scripts/python scripts/datasets/validate_harvard_tweets.py
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXIT_FAIL, EXIT_MISSING, EXIT_OK, RAW, REPORTS,
    get_logger, human_bytes, sha256_file, utcnow, write_json,
)

DEFAULT_DIR = RAW / "text" / "harvard_tweets"
REPORT = REPORTS / "harvard_tweets_validation.json"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

# Depositor-declared class counts (dataset description + sheet legend, v1.1)
DECLARED_COUNTS = {"depressed": 58405, "non_depressed": 41185, "total": 99590}
DEPRESSION_LEXICON = (
    "depress", "anxiety", "suicid", "hopeless", "worthless", "lonely",
    "sad", "cry", "kill", "therapy", "stress", "pain", "empty", "hate",
)
LEGEND_LABELS = {"class", "depressed", "non-depressed", "non depressed", "non_depressed"}


def _col_index(ref: str) -> int:
    letters = re.match(r"([A-Z]+)", ref).group(1)
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def read_xlsx(path: Path) -> tuple[list[dict], list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet_name = workbook.find("m:sheets/m:sheet", NS).get("name")
        sheets = [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet")]
        sheet_path = sorted(sheets)[0]
        root = ET.fromstring(zf.read(sheet_path))

        rows: list[list[str]] = []
        for row in root.iter(f"{{{NS['m']}}}row"):
            values: dict[int, str] = {}
            for cell in row.findall("m:c", NS):
                ref = cell.get("r", "")
                if not ref:
                    continue
                idx = _col_index(ref)
                t = cell.get("t")
                v = cell.find("m:v", NS)
                is_ = cell.find("m:is", NS)
                if t == "s" and v is not None:
                    values[idx] = shared[int(v.text)]
                elif t == "inlineStr" and is_ is not None:
                    values[idx] = "".join(x.text or "" for x in is_.iter(f"{{{NS['m']}}}t"))
                elif v is not None:
                    values[idx] = v.text or ""
            if values:
                width = max(values) + 1
                rows.append([values.get(i, "") for i in range(width)])

    if not rows:
        return [], [sheet_name]
    width = max(len(r) for r in rows)
    header = [h.strip() for h in rows[0]]
    # Columns that only appear in data rows (the side legend is one) must not be
    # dropped: pad both the header and every row to the sheet's real width.
    header += [f"column_{i + 1}" for i in range(len(header), width)]
    records = []
    for r in rows[1:]:
        r = r + [""] * (width - len(r))
        records.append({header[i]: r[i] for i in range(width)})
    return records, [sheet_name]


def find_text_and_label(records: list[dict]) -> tuple[str | None, str | None]:
    columns = list(records[0].keys()) if records else []
    lowered = {c.lower(): c for c in columns}
    text_col = next((lowered[c] for c in ("text", "tweet", "content", "tweets", "message") if c in lowered), None)
    label_col = next((lowered[c] for c in ("label", "class", "target", "depressed", "category") if c in lowered), None)
    return text_col, label_col


def _legend(records: list[dict], columns: list[str]) -> dict:
    """Find the in-sheet class/count legend (it sits in an unlabelled side column).

    Layout in v1.1 of the workbook:
        C3='Class'          D3='Binary classification data balance'
        C4='Depressed'      D4=58405
        C5='Non-depressed'  D5=41185
        C6=''               D6=99590
    """
    entries: list[dict] = []
    total = None
    for idx, col in enumerate(columns):
        if col.strip().lower() in ("text", "label"):
            continue
        right = columns[idx + 1:]
        if not right:
            continue
        found_here = False
        for i, row in enumerate(records):
            cell = str(row.get(col, "")).strip()
            count = next((int(str(row.get(c, "")).strip()) for c in right
                          if str(row.get(c, "")).strip().isdigit()), None)
            if cell.lower() in LEGEND_LABELS:
                found_here = True
                entries.append({"class": cell, "count": count, "row": i + 2})
            elif found_here and cell == "" and count is not None and total is None:
                total = count
        if found_here:
            break
    counts = {e["class"].lower().replace("-", "_"): e["count"] for e in entries if e["count"]}
    declared_dep = counts.get("depressed")
    declared_non = counts.get("non_depressed") or counts.get("non depressed")
    return {
        "found": bool(entries),
        "entries": entries,
        "declared_depressed": declared_dep,
        "declared_non_depressed": declared_non,
        "declared_total": total,
        "states_numeric_code_mapping": False,
    }


def _label_evidence(records: list[dict], text_col: str, label_col: str) -> dict:
    by_code: dict[str, dict] = {}
    codes = sorted({str(r.get(label_col, "")).strip() for r in records})
    for code in codes:
        rows = [str(r.get(text_col, "")).lower() for r in records
                if str(r.get(label_col, "")).strip() == code]
        hits = sum(1 for t in rows if any(k in t for k in DEPRESSION_LEXICON))
        by_code[code] = {
            "rows": len(rows),
            "rows_with_depression_term": hits,
            "depression_term_rate": round(hits / len(rows), 4) if rows else None,
        }
    ordered = sorted(by_code.items(), key=lambda kv: kv[1]["depression_term_rate"] or 0)
    mapping = None
    if len(ordered) == 2:
        mapping = {ordered[1][0]: "depressed", ordered[0][0]: "non_depressed"}
    return {
        "by_code": by_code,
        "lexicon_used": list(DEPRESSION_LEXICON),
        "lexical_inference": mapping,
        "depositor_declared": DECLARED_COUNTS,
    }


STOPWORDS = frozenset(
    "the a an and or but i you he she it we they is are was were be been to of in on for "
    "with my your me at this that not no do does did have has had".split()
)


def _preprocessing_artifacts(texts: list[str]) -> dict:
    xexx_rows = 0
    token_counts: list[int] = []
    stopword_share: list[float] = []
    for t in texts:
        tokens = t.lower().split()
        token_counts.append(len(tokens))
        if tokens:
            stopword_share.append(sum(1 for w in tokens if w in STOPWORDS) / len(tokens))
        if "xexx" in t:
            xexx_rows += 1
    n = len(texts) or 1
    return {
        "xexx_apostrophe_placeholder_rows": xexx_rows,
        "xexx_row_ratio": round(xexx_rows / n, 4),
        "mean_tokens": round(sum(token_counts) / n, 2),
        "mean_stopword_share": round(sum(stopword_share) / len(stopword_share), 4) if stopword_share else None,
        "rows_with_no_stopwords_ratio": round(
            sum(1 for s in stopword_share if s == 0) / len(stopword_share), 4) if stopword_share else None,
        "note": ("the released text is not raw tweet text: it is a short lower-cased token "
                 "sequence with most function words removed (mean 10.1 tokens, 54% of rows "
                 "contain no common stopword) and apostrophes replaced by 'xexx' in ~22% of "
                 "rows. URLs, @mentions and emoji are absent. Word order survives but the "
                 "surface form and any metadata a model would rely on in a real tweet are gone."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()
    log = get_logger("validate_harvard_tweets")

    files = sorted(args.dir.glob("*.xlsx"))
    if not files:
        log.error("no .xlsx found in %s (run download_harvard_tweets.py after approval)", args.dir)
        write_json(args.report, {"generated_at": utcnow(), "status": "missing_data",
                                 "directory": str(args.dir), "ok": False})
        return EXIT_MISSING
    path = files[0]
    log.info("validating %s (%s)", path.name, human_bytes(path.stat().st_size))

    records, sheets = read_xlsx(path)
    text_col, label_col = find_text_and_label(records)
    columns = list(records[0].keys()) if records else []
    issues: list[str] = []

    if not records:
        issues.append("no data rows")
    if text_col is None:
        issues.append(f"no text column found in {columns}")
    if label_col is None:
        issues.append(f"no label column found in {columns}")

    texts = [str(r.get(text_col, "")) for r in records] if text_col else []
    labels = [str(r.get(label_col, "")).strip() for r in records] if label_col else []
    label_counts = Counter(labels)
    nonempty = [t for t in texts if t.strip()]
    empty = len(texts) - len(nonempty)
    if empty:
        issues.append(f"{empty} empty text records")

    dup_exact = len(nonempty) - len(set(nonempty))
    norm = Counter(re.sub(r"\s+", " ", t.strip().lower()) for t in nonempty)
    dup_normalised = sum(c - 1 for c in norm.values() if c > 1)

    # Do near-duplicate texts ever disagree on the label? A contradiction would
    # point at label noise, so it is counted rather than assumed absent.
    label_by_norm: dict[str, set[str]] = {}
    for t, lab in zip(texts, labels):
        key = re.sub(r"\s+", " ", t.strip().lower())
        label_by_norm.setdefault(key, set()).add(lab)
    duplicate_label_conflicts = sum(1 for v in label_by_norm.values() if len(v) > 1)

    if label_counts and not set(label_counts).issubset({"0", "1", "0.0", "1.0"}):
        log.warning("label values are not only 0/1: %s", dict(label_counts))

    user_like = [c for c in columns
                 if any(k in c.lower() for k in ("user", "screen_name", "author", "account", "id"))]
    id_like = [c for c in columns if c.lower() in ("id", "tweet_id", "user_id", "userid")]

    cjk_chars = sum(len(CJK_RE.findall(t)) for t in nonempty)
    latin_chars = sum(len(ASCII_LETTER_RE.findall(t)) for t in nonempty)
    non_ascii_heavy = sum(
        1 for t in nonempty
        if len(ASCII_LETTER_RE.findall(t)) < 0.5 * max(len(CJK_RE.findall(t)), 1) and CJK_RE.search(t)
    )

    urls = sum(len(URL_RE.findall(t)) for t in nonempty)
    mentions = sum(len(MENTION_RE.findall(t)) for t in nonempty)
    emails = sum(len(EMAIL_RE.findall(t)) for t in nonempty)
    phones = sum(len(PHONE_RE.findall(t)) for t in nonempty)

    lengths = [len(t) for t in nonempty]
    lengths.sort()

    def pct(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(p * (len(lengths) - 1)))] if lengths else 0

    stats = {
        "records": len(records),
        "records_with_text": len(nonempty),
        "empty_text_records": empty,
        "unique_exact_texts": len(set(nonempty)),
        "duplicate_rows_exact": dup_exact,
        "duplicate_rows_after_normalisation": dup_normalised,
        "duplicate_label_conflicts": duplicate_label_conflicts,
        "label_column": label_col,
        "label_distribution": dict(sorted(label_counts.items())),
        "other_columns": [c for c in columns if c not in (text_col, label_col)],
        "user_identifier_columns": user_like,
        "id_columns": id_like,
        "unique_ids": {c: len({r[c] for r in records if str(r.get(c, "")).strip()}) for c in id_like},
        "language": {
            "latin_letters": latin_chars,
            "cjk_chars": cjk_chars,
            "cjk_dominant_rows": non_ascii_heavy,
            "ascii_ratio": round(latin_chars / (latin_chars + cjk_chars), 4) if (latin_chars + cjk_chars) else None,
        },
        "pii_markers": {"urls": urls, "mentions": mentions, "emails": emails, "phone_like": phones},
        "text_length": {"min": lengths[0] if lengths else 0, "p50": pct(0.5), "p90": pct(0.9),
                        "max": lengths[-1] if lengths else 0, "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0},
    }

    if latin_chars and cjk_chars and cjk_chars > 0.05 * latin_chars:
        issues.append("dataset contains substantial CJK text; language assumption 'English' needs review")
    if not user_like and not id_like:
        log.info("no user identifier column: user-level split is not possible from this file")

    legend = _legend(records, columns) if text_col and label_col else {"found": False}
    evidence = _label_evidence(records, text_col, label_col) if text_col and label_col else {}
    artifacts = _preprocessing_artifacts(nonempty)
    warnings: list[str] = []

    mapping = evidence.get("lexical_inference") if evidence else None
    declared_alignment = None
    if mapping and legend.get("declared_depressed"):
        depressed_code = next(k for k, v in mapping.items() if v == "depressed")
        observed = stats["label_distribution"].get(depressed_code)
        declared_alignment = {
            "code_inferred_as_depressed": depressed_code,
            "rows_with_that_code": observed,
            "depositor_declared_depressed": legend.get("declared_depressed"),
            "counts_agree": observed == legend.get("declared_depressed"),
        }
        if not declared_alignment["counts_agree"]:
            other_code = next(k for k in evidence["by_code"] if k != depressed_code)
            warnings.append(
                "label code -> class mapping is unresolvable from the file itself: the depositor's "
                f"legend states Depressed={legend.get('declared_depressed')}, and "
                f"{stats['label_distribution'].get(other_code)} rows carry code {other_code} while "
                f"{observed} carry code {depressed_code}, so reading the legend positionally implies "
                f"{other_code}=depressed. The depression-lexicon rate points the other way: "
                f"{evidence['by_code'][depressed_code]['depression_term_rate']:.1%} of code {depressed_code} rows "
                f"vs {evidence['by_code'][other_code]['depression_term_rate']:.1%} of code {other_code} rows. "
                "Original labels are preserved unchanged; any binary use must state which mapping it "
                "assumes, and the ambiguity must be resolved with the depositor before results are "
                "compared across datasets.")
    if legend.get("found"):
        warnings.append("the workbook ships a class/count legend in an unlabelled side column; "
                        "it never states the numeric code mapping")
    else:
        warnings.append("no class legend found; the label code mapping is undocumented in the file")
    if artifacts["xexx_row_ratio"] and artifacts["xexx_row_ratio"] > 0.05:
        warnings.append(
            f"the released text is pre-processed, not raw: mean {artifacts['mean_tokens']} tokens, "
            f"{artifacts['rows_with_no_stopwords_ratio']:.0%} of rows contain no common stopword, and "
            f"{artifacts['xexx_row_ratio']:.0%} of rows contain 'xexx' apostrophe placeholders")
    if not stats["label_distribution"] or set(stats["label_distribution"]) - {"0", "1"}:
        warnings.append("label column is not a clean 0/1 encoding")

    report = {
        "generated_at": utcnow(),
        "status": "validated",
        "dataset_name": "harvard_tweets",
        "source": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ISZCSA",
        "doi": "doi:10.7910/DVN/ISZCSA",
        "dataset_version": "1.1",
        "license": "CC0 1.0",
        "language_declared": "English",
        "file": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "sheets": sheets},
        "columns": columns,
        "text_column": text_col,
        "statistics": stats,
        "legend": legend,
        "preprocessing_artifacts": artifacts,
        "label_code_mapping": {
            "declared_by_depositor": DECLARED_COUNTS,
            "evidence": evidence,
            "lexically_inferred_mapping": mapping,
            "declared_alignment": declared_alignment,
            "ambiguous": True,
            "policy": "original codes are preserved unchanged; no relabelling is performed in this phase",
        },
        "label_semantics": {
            "status": "unresolved",
            "codes_preserved": True,
            "depositor_mapping_found": False,
            "evidence_checked": [
                "Dataverse record metadata (v1.1): class counts only, no code mapping",
                "in-workbook legend: class counts only, no code mapping",
                "depression-lexicon rate per code (evidence, not proof)",
            ],
            "supervised_use_allowed": False,
            "blocked_reason": "which numeric code means 'depressed' cannot be established from the "
                              "released file or the depositor's documentation; see the audit doc",
            "audit_doc": "docs/harvard_tweets_label_audit.md",
            "depositor_contact_required": True,
        },
        "label_definition_source": "dataset description (binary depressed / non-depressed tweet classification); "
                                   "the numeric code mapping is not stated by the depositor",
        "label_is_clinical_diagnosis": False,
        "label_provenance": "depositor-assigned tweet-level weak label; no annotation protocol, "
                            "annotator count or inter-annotator agreement is published",
        "participant_level_possible": bool(user_like or id_like),
        "participant_level_note": "no user, tweet or session identifier column exists, so tweets cannot be "
                                  "grouped by author and participant-independent evaluation cannot be guaranteed",
        "issues": issues,
        "warnings": warnings,
        "ok": not issues,
    }
    write_json(args.report, report)

    log.info("%d rows | labels %s | empty %d | exact dupes %d | user-id cols %s",
             len(records), stats["label_distribution"], empty, dup_exact, user_like or "none")
    log.info("language: %s (ascii ratio %s)", "Latin/ASCII" if cjk_chars == 0 else "mixed",
             stats["language"]["ascii_ratio"])
    log.info("PII markers: %s", stats["pii_markers"])
    for i in issues:
        log.warning("issue: %s", i)
    for w in warnings:
        log.warning("warning: %s", w)
    log.info("label legend: %s | lexical mapping %s | counts agree: %s",
             legend.get("entries"), mapping, (declared_alignment or {}).get("counts_agree"))
    log.info("report written to %s", args.report)
    return EXIT_OK if not issues else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
