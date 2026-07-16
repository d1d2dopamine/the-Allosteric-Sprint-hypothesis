# -*- coding: utf-8 -*-
"""
UCLA_CNP_VALIDATOR.py — behavioral validator for OpenNeuro ds000030.

Purpose
-------
Downloads ONLY small tabular Stop-Signal/phenotype files (never MRI), validates
ADHD vs healthy-control behavioral heterogeneity without forced clusters, then
runs a secondary ADHD-only clustering analysis with held-out validation.

Default workflow
----------------
    python UCLA_CNP_VALIDATOR.py

Useful options
--------------
    python UCLA_CNP_VALIDATOR.py --keep-data
    python UCLA_CNP_VALIDATOR.py --data-dir PATH --no-download
    python UCLA_CNP_VALIDATOR.py --self-test

Outputs
-------
UCLA_CNP_VALID_output/
    ucla_cnp_academic_diagnostic.txt
    ucla_cnp_participant_metrics.csv
    ucla_cnp_quality_control.csv
    ucla_cnp_cluster_diagnostics.csv
    ucla_cnp_group_distributions.png
    ucla_cnp_cluster_validation.png (if stable clusters are found)
    ucla_cnp_temporal_dynamics.png
    ucla_cnp_analysis_config.json
    ucla_cnp_reproducibility_log.txt
    UCLA_CNP_VALID_output.zip

Methodological commitments
---------------------------
1. Primary analysis is cluster-free: ADHD vs healthy controls.
2. Clusters are fitted ONLY within ADHD and ONLY from go-process measures:
   median go RT and go RT MAD. Diagnosis/control data and inhibition outcomes
   never define the clusters.
3. A two-cluster interpretation is accepted only if preregistered-style gates
   pass (BIC improvement, silhouette, minimum size, bootstrap stability).
4. SSRT is held out for validation and is never used to create clusters.
5. No dopaminergic, metabolic, neural, diagnostic, or clinical mechanism is
   inferred from behavioral clusters.

Dataset
-------
UCLA Consortium for Neuropsychiatric Phenomics LA5c Study, OpenNeuro ds000030.
This script downloads the current public OpenNeuro snapshot from the anonymous
AWS S3 bucket. Cite the original dataset descriptor before publication.

Author: D1D2DOPAMINE
Generated with AI assistance; analytic decisions and interpretations require
independent human review.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
import warnings
import zipfile

# ----------------------------- configuration -----------------------------

SCRIPT_VERSION = "0.1.0"
DATASET_ID = "ds000030"
S3_URI = "s3://openneuro.org/ds000030"
OUTPUT_FOLDER_NAME = "UCLA_CNP_VALID_output"
RAW_FOLDER_NAME = "UCLA_CNP_VALID_raw"
REPORT_NAME = "ucla_cnp_academic_diagnostic.txt"
LOG_NAME = "ucla_cnp_reproducibility_log.txt"
METRICS_NAME = "ucla_cnp_participant_metrics.csv"
QC_NAME = "ucla_cnp_quality_control.csv"
CLUSTER_NAME = "ucla_cnp_cluster_diagnostics.csv"
CONFIG_NAME = "ucla_cnp_analysis_config.json"
GROUP_PLOT_NAME = "ucla_cnp_group_distributions.png"
CLUSTER_PLOT_NAME = "ucla_cnp_cluster_validation.png"
TEMPORAL_PLOT_NAME = "ucla_cnp_temporal_dynamics.png"
ZIP_NAME = "UCLA_CNP_VALID_output.zip"

SEED = 20260716
N_PERMUTATIONS = 5000
N_BOOTSTRAPS = 500
MIN_VALID_GO_TRIALS = 40
MIN_STOP_TRIALS = 12
RT_MIN_SECONDS = 0.10
RT_MAX_SECONDS = 3.00
MIN_GROUP_N = 8

# Gates for allowing a discrete two-subtype interpretation.
GMM_BIC_IMPROVEMENT_MIN = 10.0       # BIC(K=1) - BIC(K=2)
GMM_SILHOUETTE_MIN = 0.25
GMM_MIN_CLUSTER_SHARE = 0.15
GMM_BOOTSTRAP_ARI_MIN = 0.60

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "boto3": "boto3",
}

LOG_BUFFER: list[str] = []


def log(*parts) -> None:
    text = " ".join(str(x) for x in parts)
    print(text, flush=True)
    LOG_BUFFER.append(text)


def ensure_dependencies(no_auto_install: bool = False) -> None:
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(pip_name)
    if not missing:
        return
    if no_auto_install:
        raise RuntimeError("Missing packages: %s" % ", ".join(missing))
    log("Installing missing Python packages:", ", ".join(missing))
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + missing
    subprocess.check_call(cmd)
    importlib.invalidate_caches()


def imports_after_bootstrap():
    global np, pd, stats, plt, GaussianMixture, RobustScaler
    global silhouette_score, adjusted_rand_score
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import silhouette_score, adjusted_rand_score


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().lower()


def find_column(df, aliases, contains=()):
    normalized = {normalize_name(c): c for c in df.columns}
    for alias in aliases:
        if normalize_name(alias) in normalized:
            return normalized[normalize_name(alias)]
    for c in df.columns:
        key = normalize_name(c)
        if any(all(token in key for token in group) for group in contains):
            return c
    return None


def read_table(path: Path):
    attempts = [
        {"sep": "\t"}, {"sep": ";"}, {"sep": ","},
        {"sep": None, "engine": "python"},
    ]
    last = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 2:
                return df
        except Exception as exc:
            last = exc
    raise RuntimeError("Could not parse %s (%s)" % (path, last))


# ------------------------------- download --------------------------------

def download_public_subset(raw_dir: Path) -> None:
    """Download only small public tabular files via anonymous S3.

    Uses boto3 pagination instead of cloning Git/DataLad. Every object is
    filtered before download; NIfTI/MRI files are categorically rejected.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    raw_dir.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket = "openneuro.org"
    prefix = DATASET_ID + "/"

    def wanted(relative: str) -> bool:
        low = relative.lower()
        if low == "participants.tsv":
            return True
        if low == "task-stopsignal_bold.json":
            return True
        if low.startswith("phenotype/") and low.endswith((".tsv", ".csv", ".json")):
            return True
        if re.match(r"sub-[^/]+/func/.*task-stopsignal.*_events\.(tsv|csv)$", low):
            return True
        if re.match(r"sub-[^/]+/beh/.*task-stopsignaltraining.*_events\.(tsv|csv)$", low):
            return True
        return False

    paginator = client.get_paginator("list_objects_v2")
    selected = []
    log("Listing the public OpenNeuro S3 snapshot; MRI objects will not be downloaded...")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix):]
            if wanted(relative):
                selected.append((key, relative, int(obj.get("Size", 0))))

    if not selected:
        raise RuntimeError(
            "OpenNeuro S3 returned no matching behavioral files. The public snapshot "
            "layout may have changed. Use --data-dir with manually downloaded files."
        )

    total = sum(size for _, _, size in selected)
    log("Selected %d tabular/metadata files (%.2f MB total)." %
        (len(selected), total / 1024 / 1024))
    for i, (key, relative, size) in enumerate(selected, 1):
        # Absolute safety gate: never download neuroimaging payloads.
        if relative.lower().endswith((".nii", ".nii.gz", ".dcm", ".bval", ".bvec")):
            raise RuntimeError("Safety gate rejected imaging file: " + relative)
        dest = raw_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == size and size > 0:
            continue
        if i == 1 or i % 50 == 0 or i == len(selected):
            log("Downloading file %d/%d..." % (i, len(selected)))
        client.download_file(bucket, key, str(dest))

    manifest = []
    for _, relative, size in selected:
        path = raw_dir / relative
        manifest.append({
            "path": relative,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    (raw_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def locate_dataset_root(data_dir: Path) -> Path:
    candidates = [data_dir]
    candidates += [p.parent for p in data_dir.rglob("participants.tsv")]
    for root in candidates:
        if (root / "participants.tsv").is_file():
            return root
    raise FileNotFoundError("participants.tsv was not found under %s" % data_dir)


# -------------------------- participant metadata --------------------------

def load_participants(root: Path):
    path = root / "participants.tsv"
    df = read_table(path)
    id_col = find_column(df, ["participant_id", "subject_id", "id"],
                         contains=[("participant", "id"), ("subject", "id")])
    diagnosis_col = find_column(
        df, ["diagnosis", "diagnostic_group", "group", "dx"],
        contains=[("diagnos",), ("group",)]
    )
    if id_col is None or diagnosis_col is None:
        raise RuntimeError(
            "participants.tsv must contain participant ID and diagnosis/group. "
            "Found columns: %s" % list(df.columns)
        )

    def classify_dx(v):
        s = normalize_text(v)
        compact = normalize_name(v)
        if "adhd" in compact or "attentiondeficit" in compact:
            return "ADHD"
        if compact in {"control", "healthycontrol", "healthy", "hc", "normal"}:
            return "Healthy Control"
        if "control" in compact and not any(x in compact for x in ("clinical", "patient")):
            return "Healthy Control"
        return "Other/Unknown"

    out = pd.DataFrame({
        "participant_id": df[id_col].astype(str).str.strip(),
        "diagnosis_raw": df[diagnosis_col].astype(str),
    })
    out["group"] = out["diagnosis_raw"].map(classify_dx)

    age_col = find_column(df, ["age", "age_group"], contains=[("age",)])
    sex_col = find_column(df, ["sex", "gender"], contains=[("sex",), ("gender",)])
    if age_col:
        out["age"] = df[age_col]
    if sex_col:
        out["sex"] = df[sex_col]

    # Preserve scan inventory fields; they help diagnose absent task files.
    for c in df.columns:
        if "stop" in normalize_name(c) and c not in out.columns:
            out["inventory_" + normalize_name(c)] = df[c]

    counts = out["group"].value_counts(dropna=False).to_dict()
    log("Diagnosis mapping from participants.tsv:", counts)
    if counts.get("ADHD", 0) == 0 or counts.get("Healthy Control", 0) == 0:
        examples = df[diagnosis_col].value_counts(dropna=False).head(20).to_dict()
        raise RuntimeError(
            "Could not identify both ADHD and healthy controls. Raw diagnosis values: %s" % examples
        )
    return out


def find_event_files(root: Path):
    scanner = {}
    training = {}
    for path in root.rglob("*_events.tsv"):
        low = path.name.lower()
        match = re.search(r"(sub-[a-z0-9]+)", path.as_posix(), flags=re.I)
        if not match:
            continue
        pid = match.group(1)
        if "task-stopsignaltraining" in low:
            training[pid] = path
        elif "task-stopsignal" in low:
            scanner[pid] = path
    for path in root.rglob("*_events.csv"):
        low = path.name.lower()
        match = re.search(r"(sub-[a-z0-9]+)", path.as_posix(), flags=re.I)
        if not match:
            continue
        pid = match.group(1)
        if "task-stopsignaltraining" in low:
            training.setdefault(pid, path)
        elif "task-stopsignal" in low:
            scanner.setdefault(pid, path)
    log("Event files found: scanner=%d, training=%d" % (len(scanner), len(training)))
    return scanner, training


# ------------------------------ event parser ------------------------------

def numeric_series(df, col):
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def boolish(value):
    s = normalize_text(value)
    if s in {"1", "1.0", "true", "yes", "correct", "success", "successful", "hit"}:
        return True
    if s in {"0", "0.0", "false", "no", "incorrect", "error", "failure", "failed", "miss"}:
        return False
    return None


def infer_event_schema(df):
    trial_col = find_column(
        df, ["trial_type", "trialtype", "condition", "trial", "type"],
        contains=[("trial", "type"), ("condition",)]
    )
    rt_col = find_column(
        df, ["response_time", "reaction_time", "response_time_sec", "rt", "responded_rt"],
        contains=[("response", "time"), ("reaction", "time")]
    )
    ssd_col = find_column(
        df, ["stop_signal_delay", "stopsignaldelay", "stop_signal_onset", "ssd"],
        contains=[("stop", "signal", "delay"), ("ssd",)]
    )
    correct_col = find_column(
        df, ["correct", "accuracy", "response_correct", "correctness"],
        contains=[("correct",), ("accuracy",)]
    )
    response_col = find_column(
        df, ["response", "participant_response", "button", "key_press", "key"],
        contains=[("response",), ("button",), ("keypress",)]
    )
    onset_col = find_column(df, ["onset"], contains=[("onset",)])
    return {
        "trial": trial_col, "rt": rt_col, "ssd": ssd_col,
        "correct": correct_col, "response": response_col, "onset": onset_col,
    }


def classify_trials(df, schema):
    if schema["trial"] is None:
        raise RuntimeError("No trial_type/condition column. Columns=%s" % list(df.columns))
    raw = df[schema["trial"]].map(normalize_text)

    is_stop = raw.str.contains("stop", na=False)
    is_go = raw.str.contains("go", na=False) & ~is_stop

    # Some files encode outcomes directly in trial_type.
    stop_success = is_stop & raw.str.contains("success|correct|inhibit", regex=True, na=False)
    stop_failure = is_stop & raw.str.contains("fail|incorrect|error|respond", regex=True, na=False)

    rt = numeric_series(df, schema["rt"])
    finite_rt = rt.notna() & (rt > 0)

    if schema["correct"] is not None:
        parsed = df[schema["correct"]].map(boolish)
        parsed_true = parsed.map(lambda x: x is True)
        parsed_false = parsed.map(lambda x: x is False)
        stop_success = stop_success | (is_stop & parsed_true & ~finite_rt)
        stop_failure = stop_failure | (is_stop & parsed_false & finite_rt)

    # For a stop trial, no registered response generally means successful stop.
    unresolved = is_stop & ~(stop_success | stop_failure)
    stop_success = stop_success | (unresolved & ~finite_rt)
    stop_failure = stop_failure | (unresolved & finite_rt)

    if not is_go.any() and (~is_stop).any():
        # Conservative fallback: non-stop event rows with an RT are go trials.
        is_go = (~is_stop) & finite_rt

    return is_go.to_numpy(bool), is_stop.to_numpy(bool), \
           stop_success.to_numpy(bool), stop_failure.to_numpy(bool), rt


def rt_to_seconds(rt):
    rt = pd.to_numeric(rt, errors="coerce").astype(float)
    positive = rt[(rt > 0) & np.isfinite(rt)]
    if positive.empty:
        return rt, "unknown"
    median = float(positive.median())
    if median > 20:  # typical millisecond encoding
        return rt / 1000.0, "milliseconds->seconds"
    return rt, "seconds"


def ssd_to_seconds(ssd):
    ssd = pd.to_numeric(ssd, errors="coerce").astype(float)
    positive = ssd[(ssd > 0) & np.isfinite(ssd)]
    if positive.empty:
        return ssd, "unavailable"
    if float(positive.median()) > 10:
        return ssd / 1000.0, "milliseconds->seconds"
    return ssd, "seconds"


def robust_mad(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def compute_ssrt(go_rt, ssd, stop_failure_rate):
    """Integration-method SSRT: quantile(go RT, p(respond|stop)) - mean SSD."""
    go = np.asarray(go_rt, dtype=float)
    go = go[np.isfinite(go)]
    delays = np.asarray(ssd, dtype=float)
    delays = delays[np.isfinite(delays) & (delays >= 0)]
    if len(go) < MIN_VALID_GO_TRIALS or len(delays) < MIN_STOP_TRIALS:
        return np.nan
    p = float(np.clip(stop_failure_rate, 0.01, 0.99))
    nth_rt = float(np.quantile(np.sort(go), p))
    value = nth_rt - float(np.mean(delays))
    return value if 0.02 <= value <= 1.5 else np.nan


def parse_participant_events(pid: str, path: Path, group: str):
    row = {
        "participant_id": pid, "group": group, "event_file": str(path),
        "included_primary": False, "qc_reason": "",
    }
    try:
        df = read_table(path)
        schema = infer_event_schema(df)
        row["event_columns"] = " | ".join(map(str, df.columns))
        row["schema"] = json.dumps(schema, ensure_ascii=False)
        is_go, is_stop, stop_success, stop_failure, rt_raw = classify_trials(df, schema)
        rt, rt_unit = rt_to_seconds(rt_raw)
        row["rt_unit_handling"] = rt_unit

        valid_rt = np.isfinite(rt.to_numpy(float)) & (rt.to_numpy(float) >= RT_MIN_SECONDS) & \
                   (rt.to_numpy(float) <= RT_MAX_SECONDS)
        go_all = is_go
        go_valid = is_go & valid_rt
        stop_all = is_stop

        go_rt = rt.to_numpy(float)[go_valid]
        row["n_rows"] = int(len(df))
        row["n_go_trials"] = int(go_all.sum())
        row["n_valid_go_rt"] = int(go_valid.sum())
        row["n_stop_trials"] = int(stop_all.sum())
        row["median_go_rt_s"] = float(np.median(go_rt)) if len(go_rt) else np.nan
        row["mean_go_rt_s"] = float(np.mean(go_rt)) if len(go_rt) else np.nan
        row["go_rt_sd_s"] = float(np.std(go_rt, ddof=1)) if len(go_rt) > 1 else np.nan
        row["go_rt_mad_s"] = robust_mad(go_rt)
        row["go_rt_iqr_s"] = float(np.subtract(*np.percentile(go_rt, [75, 25]))) if len(go_rt) else np.nan
        row["go_omission_rate"] = float(1 - go_valid.sum() / go_all.sum()) if go_all.sum() else np.nan
        row["stop_success_rate"] = float(stop_success.sum() / stop_all.sum()) if stop_all.sum() else np.nan
        row["stop_failure_rate"] = float(stop_failure.sum() / stop_all.sum()) if stop_all.sum() else np.nan

        ssd_raw = numeric_series(df, schema["ssd"])
        ssd, ssd_unit = ssd_to_seconds(ssd_raw)
        row["ssd_unit_handling"] = ssd_unit
        stop_ssd = ssd.to_numpy(float)[stop_all]
        row["mean_ssd_s"] = float(np.nanmean(stop_ssd)) if np.isfinite(stop_ssd).any() else np.nan
        row["ssrt_s"] = compute_ssrt(go_rt, stop_ssd, row["stop_failure_rate"])

        reasons = []
        if row["n_valid_go_rt"] < MIN_VALID_GO_TRIALS:
            reasons.append("fewer than %d valid go RTs" % MIN_VALID_GO_TRIALS)
        if row["n_stop_trials"] < MIN_STOP_TRIALS:
            reasons.append("fewer than %d stop trials" % MIN_STOP_TRIALS)
        if not np.isfinite(row["median_go_rt_s"]) or not np.isfinite(row["go_rt_mad_s"]):
            reasons.append("missing go RT metrics")
        row["qc_reason"] = "; ".join(reasons)
        row["included_primary"] = len(reasons) == 0

        # Four equal sequential blocks, preserving trial order.
        task_indices = np.flatnonzero(is_go | is_stop)
        block_assign = np.full(len(df), -1, dtype=int)
        for block_no, indices in enumerate(np.array_split(task_indices, 4), 1):
            block_assign[indices] = block_no
        for b in range(1, 5):
            mask = (block_assign == b) & go_valid
            vals = rt.to_numpy(float)[mask]
            row["block%d_median_go_rt_s" % b] = float(np.median(vals)) if len(vals) else np.nan
            row["block%d_go_rt_mad_s" % b] = robust_mad(vals)
            block_go = (block_assign == b) & is_go
            row["block%d_go_omission_rate" % b] = \
                float(1 - mask.sum() / block_go.sum()) if block_go.sum() else np.nan
        return row
    except Exception as exc:
        row["qc_reason"] = "parser error: %s" % exc
        row["parser_traceback"] = traceback.format_exc(limit=5)
        return row


# ----------------------------- statistics ---------------------------------

def rank_biserial_from_u(u, n1, n2):
    return float(1.0 - (2.0 * u) / (n1 * n2))


def bootstrap_median_difference(a, b, n_boot=N_BOOTSTRAPS, seed=SEED):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    observed = float(np.median(a) - np.median(b))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = np.median(rng.choice(a, len(a), replace=True)) - \
                   np.median(rng.choice(b, len(b), replace=True))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return observed, float(lo), float(hi)


def permutation_median_p(a, b, n_perm=N_PERMUTATIONS, seed=SEED):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    observed = abs(float(np.median(a) - np.median(b)))
    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        stat = abs(float(np.median(perm[:n_a]) - np.median(perm[n_a:])))
        exceed += stat >= observed - 1e-15
    return float((exceed + 1) / (n_perm + 1))


def compare_groups(metrics, metric):
    a = pd.to_numeric(metrics.loc[metrics["group"] == "ADHD", metric], errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(metrics.loc[metrics["group"] == "Healthy Control", metric], errors="coerce").dropna().to_numpy()
    result = {"metric": metric, "n_adhd": len(a), "n_control": len(b)}
    if len(a) < MIN_GROUP_N or len(b) < MIN_GROUP_N:
        result.update({"u": np.nan, "mw_p": np.nan, "rank_biserial_r": np.nan,
                       "median_difference": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                       "permutation_p": np.nan})
        return result
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    diff, lo, hi = bootstrap_median_difference(a, b)
    result.update({
        "u": float(u), "mw_p": float(p),
        "rank_biserial_r": rank_biserial_from_u(u, len(a), len(b)),
        "median_adhd": float(np.median(a)), "median_control": float(np.median(b)),
        "median_difference": diff, "ci_low": lo, "ci_high": hi,
        "permutation_p": permutation_median_p(a, b),
    })
    return result


def brown_forsythe(metrics, metric):
    a = pd.to_numeric(metrics.loc[metrics["group"] == "ADHD", metric], errors="coerce").dropna()
    b = pd.to_numeric(metrics.loc[metrics["group"] == "Healthy Control", metric], errors="coerce").dropna()
    if len(a) < MIN_GROUP_N or len(b) < MIN_GROUP_N:
        return {"metric": metric, "statistic": np.nan, "p": np.nan}
    stat, p = stats.levene(a, b, center="median")
    return {"metric": metric, "statistic": float(stat), "p": float(p)}


# ------------------------------ clustering --------------------------------

def bootstrap_cluster_stability(X_scaled, reference_labels, n_boot=N_BOOTSTRAPS, seed=SEED):
    """Refit K=2 GMM on bootstrap samples and predict all original points."""
    rng = np.random.default_rng(seed)
    scores = []
    n = len(X_scaled)
    for i in range(n_boot):
        idx = rng.choice(np.arange(n), n, replace=True)
        # Require enough unique observations to fit a covariance matrix.
        if len(np.unique(idx)) < max(6, X_scaled.shape[1] * 3):
            continue
        try:
            model = GaussianMixture(n_components=2, covariance_type="full",
                                    n_init=20, random_state=seed + i,
                                    reg_covar=1e-5).fit(X_scaled[idx])
            predicted = model.predict(X_scaled)
            scores.append(adjusted_rand_score(reference_labels, predicted))
        except Exception:
            continue
    return float(np.median(scores)) if scores else np.nan, len(scores)


def run_adhd_clustering(metrics):
    feature_cols = ["median_go_rt_s", "go_rt_mad_s"]
    adhd = metrics[(metrics["group"] == "ADHD") & metrics["included_primary"]].copy()
    adhd = adhd.dropna(subset=feature_cols).reset_index(drop=True)
    diagnostics = []
    outcome = {
        "stable_two_cluster": False, "reason": "", "adhd": adhd,
        "feature_cols": feature_cols, "models": {},
    }
    if len(adhd) < 20:
        outcome["reason"] = "fewer than 20 eligible ADHD participants"
        return outcome, pd.DataFrame(diagnostics)

    scaler = RobustScaler(quantile_range=(25, 75))
    X = adhd[feature_cols].to_numpy(float)
    Xs = scaler.fit_transform(X)
    models = {}
    labels_by_k = {}
    for k in (1, 2, 3):
        if len(adhd) <= k * 3:
            continue
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=50,
                                random_state=SEED, reg_covar=1e-5).fit(Xs)
        labels = model.predict(Xs)
        models[k] = model
        labels_by_k[k] = labels
        shares = np.bincount(labels, minlength=k) / len(labels)
        sil = silhouette_score(Xs, labels) if k > 1 and len(np.unique(labels)) > 1 else np.nan
        diagnostics.append({
            "k": k, "bic": float(model.bic(Xs)), "aic": float(model.aic(Xs)),
            "silhouette": float(sil) if np.isfinite(sil) else np.nan,
            "minimum_cluster_share": float(shares.min()),
        })

    outcome["models"] = models
    outcome["scaler_center"] = scaler.center_.tolist()
    outcome["scaler_scale"] = scaler.scale_.tolist()
    if 1 not in models or 2 not in models:
        outcome["reason"] = "K=1 or K=2 model unavailable"
        return outcome, pd.DataFrame(diagnostics)

    labels2 = labels_by_k[2]
    bic_improvement = float(models[1].bic(Xs) - models[2].bic(Xs))
    sil2 = float(silhouette_score(Xs, labels2))
    shares2 = np.bincount(labels2, minlength=2) / len(labels2)
    ari, valid_boots = bootstrap_cluster_stability(Xs, labels2)

    for row in diagnostics:
        if row["k"] == 2:
            row["bic_improvement_over_k1"] = bic_improvement
            row["bootstrap_median_ari"] = ari
            row["valid_bootstraps"] = valid_boots

    bic2 = float(models[2].bic(Xs))
    bic3 = float(models[3].bic(Xs)) if 3 in models else np.nan
    gates = {
        "bic": bic_improvement >= GMM_BIC_IMPROVEMENT_MIN,
        "k2_best_bic": (not np.isfinite(bic3)) or bic2 <= bic3,
        "silhouette": sil2 >= GMM_SILHOUETTE_MIN,
        "minimum_share": float(shares2.min()) >= GMM_MIN_CLUSTER_SHARE,
        "stability": np.isfinite(ari) and ari >= GMM_BOOTSTRAP_ARI_MIN,
    }
    outcome.update({
        "bic_improvement": bic_improvement, "bic_k2": bic2, "bic_k3": bic3,
        "silhouette": sil2, "minimum_share": float(shares2.min()), "bootstrap_median_ari": ari,
        "valid_bootstraps": valid_boots, "gates": gates,
    })
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        outcome["reason"] = "failed gates: " + ", ".join(failed)
        return outcome, pd.DataFrame(diagnostics)

    # Do not force hypothesis labels unless centroid geometry is compatible.
    centers_original = scaler.inverse_transform(models[2].means_)
    fast_idx = int(np.argmin(centers_original[:, 0]))
    other_idx = 1 - fast_idx
    fast = centers_original[fast_idx]
    other = centers_original[other_idx]
    interpretable = bool(fast[0] < other[0] and fast[1] <= other[1])
    label_map = {
        fast_idx: "Sprint-like" if interpretable else "ADHD Cluster A",
        other_idx: "Crash-like" if interpretable else "ADHD Cluster B",
    }
    adhd["adhd_cluster"] = [label_map[int(x)] for x in labels2]
    adhd["cluster_numeric"] = labels2
    outcome.update({
        "stable_two_cluster": True, "reason": "all gates passed",
        "adhd": adhd, "centers_original": centers_original.tolist(),
        "interpretable_as_sprint_crash": interpretable,
        "label_map": {str(k): v for k, v in label_map.items()},
    })
    return outcome, pd.DataFrame(diagnostics)


def compare_clusters(clustered_adhd, metric):
    labels = list(clustered_adhd["adhd_cluster"].dropna().unique())
    if len(labels) != 2:
        return None
    a = pd.to_numeric(clustered_adhd.loc[clustered_adhd["adhd_cluster"] == labels[0], metric], errors="coerce").dropna()
    b = pd.to_numeric(clustered_adhd.loc[clustered_adhd["adhd_cluster"] == labels[1], metric], errors="coerce").dropna()
    if len(a) < 3 or len(b) < 3:
        return {"metric": metric, "groups": labels, "n": [len(a), len(b)], "p": np.nan}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "metric": metric, "groups": labels, "n": [len(a), len(b)],
        "medians": [float(a.median()), float(b.median())],
        "u": float(u), "p": float(p),
        "rank_biserial_r": rank_biserial_from_u(u, len(a), len(b)),
    }


# -------------------------------- plots -----------------------------------

def create_group_plot(metrics, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    groups = ["Healthy Control", "ADHD"]
    colors = ["#2a9d8f", "#e76f51"]
    variables = [
        ("median_go_rt_s", "Median go RT (s)"),
        ("go_rt_mad_s", "Go RT MAD (s)"),
        ("ssrt_s", "SSRT (s; held out)"),
    ]
    rng = np.random.default_rng(SEED)
    for ax, (col, label) in zip(axes, variables):
        data = []
        for i, (group, color) in enumerate(zip(groups, colors)):
            vals = pd.to_numeric(metrics.loc[metrics["group"] == group, col], errors="coerce").dropna().to_numpy()
            data.append(vals)
            if len(vals):
                x = rng.normal(i + 1, 0.045, size=len(vals))
                ax.scatter(x, vals, s=22, alpha=0.55, color=color, edgecolor="none")
        valid_positions = [i + 1 for i, vals in enumerate(data) if len(vals)]
        valid_data = [vals for vals in data if len(vals)]
        if valid_data:
            bp = ax.boxplot(valid_data, positions=valid_positions, widths=0.45,
                            showfliers=False, patch_artist=True)
            for patch, pos in zip(bp["boxes"], valid_positions):
                patch.set_facecolor(colors[pos - 1]); patch.set_alpha(0.25)
        ax.set_xticks([1, 2], groups, rotation=12)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("UCLA CNP ds000030: cluster-free behavioral comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_cluster_plot(cluster_outcome, output_path: Path):
    adhd = cluster_outcome["adhd"]
    if not cluster_outcome["stable_two_cluster"]:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    palette = ["#d62828", "#457b9d"]
    labels = list(adhd["adhd_cluster"].unique())
    for color, label in zip(palette, labels):
        sub = adhd[adhd["adhd_cluster"] == label]
        axes[0].scatter(sub["median_go_rt_s"], sub["go_rt_mad_s"],
                        label="%s (n=%d)" % (label, len(sub)), alpha=0.75,
                        s=42, color=color, edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Median go RT (s)")
    axes[0].set_ylabel("Go RT MAD (s)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    ssrt_data = [pd.to_numeric(adhd.loc[adhd["adhd_cluster"] == label, "ssrt_s"], errors="coerce").dropna()
                 for label in labels]
    valid = [(label, vals, color) for label, vals, color in zip(labels, ssrt_data, palette) if len(vals)]
    if valid:
        boxes = axes[1].boxplot([x[1] for x in valid], labels=[x[0] for x in valid],
                                showfliers=True, patch_artist=True)
        for patch, item in zip(boxes["boxes"], valid):
            patch.set_facecolor(item[2]); patch.set_alpha(0.35)
    axes[1].set_ylabel("SSRT (s; not used for clustering)")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("ADHD-only GMM and held-out inhibition outcome")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def create_temporal_plot(metrics, cluster_outcome, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    source = metrics.copy()
    series_groups = [("Healthy Control", source[source["group"] == "Healthy Control"], "#2a9d8f"),
                     ("ADHD", source[source["group"] == "ADHD"], "#e76f51")]
    if cluster_outcome["stable_two_cluster"]:
        adhd = cluster_outcome["adhd"]
        series_groups = [(label, adhd[adhd["adhd_cluster"] == label], color)
                         for label, color in zip(adhd["adhd_cluster"].unique(), ["#d62828", "#457b9d"])]
    for label, sub, color in series_groups:
        means, sems = [], []
        for b in range(1, 5):
            vals = pd.to_numeric(sub["block%d_median_go_rt_s" % b], errors="coerce").dropna()
            means.append(float(vals.mean()) if len(vals) else np.nan)
            sems.append(float(vals.sem()) if len(vals) > 1 else np.nan)
        ax.errorbar(range(1, 5), means, yerr=sems, marker="o", capsize=3,
                    label="%s (participant-level blocks)" % label, color=color)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Sequential task block")
    ax.set_ylabel("Mean of participant median go RT (s)")
    ax.set_title("Within-session dynamics (descriptive; minutes, not longitudinal phases)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -------------------------------- report ----------------------------------

def fmt(value, digits=4):
    try:
        if value is None or not np.isfinite(float(value)):
            return "NA"
        return ("%%.%df" % digits) % float(value)
    except Exception:
        return str(value)


def write_report(output_dir, participants, metrics, comparisons, variance_tests,
                 cluster_outcome, cluster_comparisons, schema_audit, args):
    lines = []
    lines.append("UCLA CNP ds000030 Behavioral Validator — Final Report")
    lines.append("Script version: %s" % SCRIPT_VERSION)
    lines.append("Generated: %s" % dt.datetime.now().astimezone().isoformat())
    lines.append("Exploratory research only — not a diagnostic or clinical tool.")
    lines.append("")
    lines.append("1. DATA AND SAMPLE")
    lines.append("  Dataset: UCLA Consortium for Neuropsychiatric Phenomics LA5c / OpenNeuro ds000030")
    lines.append("  Only tabular metadata and Stop-Signal event files were requested; MRI was excluded.")
    lines.append("  participants.tsv rows: %d" % len(participants))
    for group in ["ADHD", "Healthy Control", "Other/Unknown"]:
        lines.append("  %-18s: metadata n=%d; eligible behavioral n=%d" % (
            group,
            int((participants["group"] == group).sum()),
            int(((metrics["group"] == group) & metrics["included_primary"]).sum()) if len(metrics) else 0,
        ))
    lines.append("")
    lines.append("2. EVENT-SCHEMA AUDIT")
    for key, value in schema_audit.items():
        lines.append("  %s: %s" % (key, value))
    lines.append("  RT inclusion window: %.2f–%.2f seconds" % (RT_MIN_SECONDS, RT_MAX_SECONDS))
    lines.append("  Minimum valid trials: go=%d, stop=%d" % (MIN_VALID_GO_TRIALS, MIN_STOP_TRIALS))
    lines.append("")
    lines.append("3. PRIMARY CLUSTER-FREE ANALYSIS: ADHD VS HEALTHY CONTROL")
    lines.append("  No cluster assignment is used in this section.")
    lines.append("  Mann–Whitney tests are two-sided; permutation tests compare median differences.")
    for res in comparisons:
        lines.append("  %s: n_ADHD=%d, n_control=%d, medians=%s vs %s, U=%s, p=%s, "
                     "rank-biserial r=%s, permutation p=%s, median difference 95%% bootstrap CI=[%s, %s]" % (
            res["metric"], res["n_adhd"], res["n_control"],
            fmt(res.get("median_adhd")), fmt(res.get("median_control")),
            fmt(res.get("u"), 2), fmt(res.get("mw_p")), fmt(res.get("rank_biserial_r"), 3),
            fmt(res.get("permutation_p")), fmt(res.get("ci_low")), fmt(res.get("ci_high"))))
    lines.append("")
    lines.append("4. DISTRIBUTIONAL HETEROGENEITY (BROWN–FORSYTHE, MEDIAN-CENTERED)")
    for res in variance_tests:
        lines.append("  %s: statistic=%s, p=%s" %
                     (res["metric"], fmt(res["statistic"], 3), fmt(res["p"])))
    lines.append("")
    lines.append("5. SECONDARY ADHD-ONLY CLUSTERING")
    lines.append("  Features: median go RT and go RT MAD only.")
    lines.append("  Controls, diagnosis labels, SSRT, stop accuracy, and clinical outcomes did not define clusters.")
    lines.append("  Candidate GMMs: K=1, K=2, K=3; robust scaling fitted only on ADHD.")
    lines.append("  Stable two-cluster result: %s" % cluster_outcome["stable_two_cluster"])
    lines.append("  Decision: %s" % cluster_outcome["reason"])
    if "bic_improvement" in cluster_outcome:
        lines.append("  Gates: BIC improvement=%s (required >= %.1f); K=2 BIC=%s, K=3 BIC=%s "
                     "(K=2 must be best); silhouette=%s (>= %.2f); minimum share=%s (>= %.2f); "
                     "bootstrap median ARI=%s (>= %.2f)" % (
            fmt(cluster_outcome["bic_improvement"], 2), GMM_BIC_IMPROVEMENT_MIN,
            fmt(cluster_outcome.get("bic_k2"), 2), fmt(cluster_outcome.get("bic_k3"), 2),
            fmt(cluster_outcome["silhouette"], 3), GMM_SILHOUETTE_MIN,
            fmt(cluster_outcome["minimum_share"], 3), GMM_MIN_CLUSTER_SHARE,
            fmt(cluster_outcome["bootstrap_median_ari"], 3), GMM_BOOTSTRAP_ARI_MIN))
    if cluster_outcome["stable_two_cluster"]:
        lines.append("  Sprint/Crash-compatible centroid geometry: %s" %
                     cluster_outcome["interpretable_as_sprint_crash"])
        counts = cluster_outcome["adhd"]["adhd_cluster"].value_counts().to_dict()
        lines.append("  Cluster counts: %s" % counts)
        lines.append("  Held-out comparisons (not cluster-defining):")
        for res in cluster_comparisons:
            if res:
                lines.append("    %s: groups=%s, n=%s, medians=%s, p=%s, r=%s" % (
                    res["metric"], res.get("groups"), res.get("n"), res.get("medians"),
                    fmt(res.get("p")), fmt(res.get("rank_biserial_r"), 3)))
    else:
        lines.append("  Interpretation: no stable evidence for discrete ADHD behavioral subtypes was accepted.")
        lines.append("  The script deliberately did not force participants into Sprint/Crash labels.")
    lines.append("")
    lines.append("6. TEMPORAL SCOPE")
    lines.append("  Four sequential blocks describe within-session changes over minutes only.")
    lines.append("  They cannot test hypothesized phase transitions over weeks or months.")
    lines.append("")
    lines.append("7. LIMITATIONS")
    lines.append("  - Stop-Signal is not the same task as BALLADEER or CPT-II; this is a conceptual replication.")
    lines.append("  - Adaptive stop-signal procedures target ~50%% stopping success; stop-success rate is not a primary outcome.")
    lines.append("  - SSRT is reported only when trial-level SSD and sufficient data permit integration-method estimation.")
    lines.append("  - GMM clusters are exploratory unless independently preregistered and replicated.")
    lines.append("  - Behavioral patterns do not validate dopamine, ATP, receptor, metabolic, or neural mechanisms.")
    lines.append("  - Cross-sectional data cannot establish cyclical within-person states.")
    lines.append("")
    lines.append("8. REPRODUCIBILITY")
    lines.append("  Random seed: %d" % SEED)
    lines.append("  Permutations: %d; bootstraps: %d" % (N_PERMUTATIONS, N_BOOTSTRAPS))
    lines.append("  Raw data kept after run: %s" % args.keep_data)
    lines.append("  See %s, %s, and %s for machine-readable details." %
                 (CONFIG_NAME, QC_NAME, CLUSTER_NAME))
    lines.append("")
    lines.append("CITATION")
    lines.append("  Cite the UCLA CNP/OpenNeuro ds000030 data descriptor and dataset accession.")
    lines.append("  This script does not redistribute raw participant data.")

    (output_dir / REPORT_NAME).write_text("\n".join(lines), encoding="utf-8")


# ------------------------------ self test ---------------------------------

def synthetic_events(pid, group, rng, cluster=None):
    n_go, n_stop = 96, 32
    if group == "Healthy Control":
        center, spread, ssrt = 0.50, 0.055, 0.22
    elif cluster == 0:
        center, spread, ssrt = 0.43, 0.065, 0.27
    else:
        center, spread, ssrt = 0.66, 0.13, 0.34
    go_rt = np.clip(rng.normal(center, spread, n_go), 0.15, 1.5)
    ssd = np.clip(rng.normal(0.25, 0.04, n_stop), 0.08, 0.5)
    fail_prob = np.clip(np.mean(go_rt <= (ssd.mean() + ssrt)), 0.25, 0.75)
    fail = rng.random(n_stop) < fail_prob
    stop_rt = np.where(fail, np.clip(rng.normal(center, spread, n_stop), 0.15, 1.5), np.nan)
    trial = np.array(["go"] * n_go + ["stop"] * n_stop, dtype=object)
    rt = np.concatenate([go_rt, stop_rt])
    ssd_full = np.concatenate([np.full(n_go, np.nan), ssd])
    order = rng.permutation(n_go + n_stop)
    return pd.DataFrame({
        "onset": np.arange(n_go + n_stop) * 2.0,
        "duration": 1.0,
        "trial_type": trial[order],
        "response_time": rt[order],
        "stop_signal_delay": ssd_full[order],
    })


def build_synthetic_dataset(root: Path):
    rng = np.random.default_rng(SEED)
    rows = []
    specs = []
    for i in range(36):
        specs.append(("sub-a%03d" % i, "ADHD", i % 2))
    for i in range(50):
        specs.append(("sub-c%03d" % i, "CONTROL", None))
    for pid, dx, cluster in specs:
        rows.append({"participant_id": pid, "diagnosis": dx, "age": 30})
        path = root / pid / "func" / (pid + "_task-stopsignal_events.tsv")
        path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_events(pid, "ADHD" if dx == "ADHD" else "Healthy Control", rng, cluster).to_csv(
            path, sep="\t", index=False)
    pd.DataFrame(rows).to_csv(root / "participants.tsv", sep="\t", index=False)
    (root / "task-stopsignal_bold.json").write_text(
        json.dumps({"TaskName": "stopsignal"}), encoding="utf-8")


# --------------------------------- main -----------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate OpenNeuro ds000030 behavioral data without MRI.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Existing ds000030 root or parent directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true",
                        help="Never access OpenNeuro; require local files")
    parser.add_argument("--keep-data", action="store_true",
                        help="Keep downloaded raw tabular files after success")
    parser.add_argument("--no-auto-install", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="Run on generated synthetic data; no network")
    parser.add_argument("--no-colab-download", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_dependencies(args.no_auto_install)
    imports_after_bootstrap()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    base = Path.cwd()
    output_dir = (args.output_dir or (base / OUTPUT_FOLDER_NAME)).resolve()
    raw_dir = (args.data_dir or (base / RAW_FOLDER_NAME)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    success = False

    config = {
        "script_version": SCRIPT_VERSION, "dataset": DATASET_ID,
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "python": sys.version, "platform": platform.platform(), "seed": SEED,
        "n_permutations": N_PERMUTATIONS, "n_bootstraps": N_BOOTSTRAPS,
        "features": ["median_go_rt_s", "go_rt_mad_s"],
        "held_out_primary": "ssrt_s",
        "gates": {
            "bic_improvement_min": GMM_BIC_IMPROVEMENT_MIN,
            "silhouette_min": GMM_SILHOUETTE_MIN,
            "minimum_cluster_share": GMM_MIN_CLUSTER_SHARE,
            "bootstrap_median_ari_min": GMM_BOOTSTRAP_ARI_MIN,
        },
    }

    try:
        log("UCLA CNP ds000030 behavioral validator v%s" % SCRIPT_VERSION)
        log("Primary analysis: cluster-free. Secondary clustering: ADHD only.")
        if args.self_test:
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            raw_dir.mkdir(parents=True)
            build_synthetic_dataset(raw_dir)
            args.keep_data = True
            log("Synthetic self-test dataset created.")
        elif not args.no_download:
            download_public_subset(raw_dir)

        root = locate_dataset_root(raw_dir)
        participants = load_participants(root)
        scanner_files, training_files = find_event_files(root)
        if not scanner_files:
            raise RuntimeError(
                "No scanner Stop-Signal events files were found. The script will not use MRI "
                "or fabricate behavioral measures. Check the OpenNeuro snapshot/layout."
            )

        rows = []
        metadata = participants.set_index("participant_id")
        target = participants[participants["group"].isin(["ADHD", "Healthy Control"])]
        for i, rec in enumerate(target.itertuples(index=False), 1):
            pid = rec.participant_id
            path = scanner_files.get(pid)
            if path is None:
                rows.append({
                    "participant_id": pid, "group": rec.group,
                    "included_primary": False, "qc_reason": "scanner events file not found",
                })
                continue
            rows.append(parse_participant_events(pid, path, rec.group))
            if i % 25 == 0 or i == len(target):
                log("Parsed %d/%d target participants..." % (i, len(target)))

        metrics = pd.DataFrame(rows)
        if metrics.empty:
            raise RuntimeError("No target participant records were produced.")
        eligible = metrics[metrics["included_primary"]].copy()
        n_adhd = int((eligible["group"] == "ADHD").sum())
        n_control = int((eligible["group"] == "Healthy Control").sum())
        log("Eligible behavioral sample: ADHD=%d, healthy control=%d" % (n_adhd, n_control))
        if n_adhd < MIN_GROUP_N or n_control < MIN_GROUP_N:
            raise RuntimeError(
                "Too few eligible participants for group analysis (ADHD=%d, control=%d). "
                "Inspect the QC and schema columns." % (n_adhd, n_control)
            )

        primary_metrics = ["median_go_rt_s", "go_rt_mad_s", "go_rt_iqr_s",
                           "go_omission_rate", "ssrt_s"]
        comparisons = [compare_groups(eligible, m) for m in primary_metrics]
        variance_tests = [brown_forsythe(eligible, m) for m in
                          ["median_go_rt_s", "go_rt_mad_s", "go_rt_iqr_s", "ssrt_s"]]

        cluster_outcome, cluster_diag = run_adhd_clustering(eligible)
        cluster_comparisons = []
        if cluster_outcome["stable_two_cluster"]:
            for m in ["ssrt_s", "go_omission_rate", "stop_failure_rate"]:
                cluster_comparisons.append(compare_clusters(cluster_outcome["adhd"], m))
            mapping = cluster_outcome["adhd"].set_index("participant_id")["adhd_cluster"]
            metrics["adhd_cluster"] = metrics["participant_id"].map(mapping)
        else:
            metrics["adhd_cluster"] = np.nan

        # Schema audit reports actual mappings rather than silently guessing.
        included_rows = metrics[metrics["included_primary"]]
        schema_examples = included_rows.get("schema", pd.Series(dtype=str)).dropna().value_counts().head(5).to_dict()
        column_examples = included_rows.get("event_columns", pd.Series(dtype=str)).dropna().value_counts().head(3).to_dict()
        schema_audit = {
            "unique accepted schemas": len(included_rows.get("schema", pd.Series(dtype=str)).dropna().unique()),
            "most common schema mappings": schema_examples,
            "most common event column sets": column_examples,
            "scanner event files": len(scanner_files),
            "training event files (not used in primary metrics)": len(training_files),
        }

        metrics.to_csv(output_dir / METRICS_NAME, index=False)
        metrics[[c for c in ["participant_id", "group", "included_primary", "qc_reason",
                             "n_rows", "n_go_trials", "n_valid_go_rt", "n_stop_trials",
                             "rt_unit_handling", "ssd_unit_handling", "schema", "event_columns"]
                 if c in metrics.columns]].to_csv(output_dir / QC_NAME, index=False)
        cluster_diag.to_csv(output_dir / CLUSTER_NAME, index=False)
        config["schema_audit"] = schema_audit
        config["cluster_result"] = {k: v for k, v in cluster_outcome.items()
                                    if k not in {"adhd", "models"}}
        (output_dir / CONFIG_NAME).write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str),
                                              encoding="utf-8")

        create_group_plot(eligible, output_dir / GROUP_PLOT_NAME)
        create_cluster_plot(cluster_outcome, output_dir / CLUSTER_PLOT_NAME)
        create_temporal_plot(eligible, cluster_outcome, output_dir / TEMPORAL_PLOT_NAME)
        write_report(output_dir, participants, metrics, comparisons, variance_tests,
                     cluster_outcome, cluster_comparisons, schema_audit, args)

        (output_dir / LOG_NAME).write_text("\n".join(LOG_BUFFER), encoding="utf-8")
        zip_path = output_dir.parent / ZIP_NAME
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in output_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(output_dir.parent))
        log("Analysis complete. Report:", output_dir / REPORT_NAME)
        log("Results archive:", zip_path)
        success = True

        # In Colab, automatically open the browser download unless disabled.
        if not args.no_colab_download:
            try:
                from google.colab import files  # type: ignore
                files.download(str(zip_path))
            except ImportError:
                pass
            except Exception as exc:
                log("Colab auto-download was unavailable:", exc)
        return 0

    except Exception as exc:
        log("FATAL ERROR:", exc)
        LOG_BUFFER.append(traceback.format_exc())
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "crash_log.txt").write_text("\n".join(LOG_BUFFER), encoding="utf-8")
        log("Crash log:", output_dir / "crash_log.txt")
        return 1
    finally:
        if success and not args.keep_data and not args.self_test and args.data_dir is None:
            with contextlib.suppress(Exception):
                shutil.rmtree(raw_dir)
                log("Downloaded raw tabular files deleted after successful analysis.")


if __name__ == "__main__":
    raise SystemExit(main())
