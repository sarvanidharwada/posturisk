"""Preprocessing pipeline for the LTMM (Long Term Movement Monitoring) dataset.

This module handles:
  1. Loading clinical / demographic metadata from the Excel spreadsheet.
  2. Reading raw WFDB-format accelerometer signals (lab walks) using only
     numpy (no ``wfdb`` dependency required).
  3. Label encoding (faller vs non-faller).
  4. Cleaning and imputing missing values.
  5. Extracting per-subject time-domain and frequency-domain gait features
     from 1-min lab-walk accelerometer windows.
  6. Producing a single clean DataFrame saved to ``data/processed/``.

Usage
-----
    python -m posturisk.preprocess                   # default paths
    python -m posturisk.preprocess --raw data/raw/   # custom raw dir
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from posturisk.features import extract_signal_features

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLINICAL_FILE = "ClinicalDemogData_COFL.xlsx"

# Signals available in each lab-walk recording
SIGNAL_NAMES = [
    "v_acc",    # vertical acceleration
    "ml_acc",   # medio-lateral acceleration
    "ap_acc",   # antero-posterior acceleration
    "yaw_vel",  # yaw angular velocity
    "pitch_vel",  # pitch angular velocity
    "roll_vel",   # roll angular velocity
]

# Clinical columns to keep (after renaming)
CLINICAL_KEEP_COLS = [
    "subject_id",
    "gender",
    "age",
    "year_fall",
    "six_month_fall",
    "gds",
    "abc_pct",
    "sf36",
    "pase",
    "mmse",
    "moca",
    "fab",
    "tmta",
    "tmtb",
    "tug",
    "fsst",
    "berg",
    "dgi",
]


# ── WFDB Reader (no external dependency) ─────────────────────────────────────


@dataclass
class WFDBHeader:
    """Parsed WFDB header (.hea) metadata for one record."""

    record_name: str
    n_signals: int
    sample_rate: int
    n_samples: int
    gains: list[float]
    baselines: list[int]
    signal_names: list[str]
    fmt: list[int]
    comments: dict[str, str]


def parse_hea(hea_path: Path) -> WFDBHeader:
    """Parse a WFDB ``.hea`` header file.

    Parameters
    ----------
    hea_path : Path
        Path to the ``.hea`` file.

    Returns
    -------
    WFDBHeader
        Parsed header metadata.
    """
    text = hea_path.read_text().strip().splitlines()

    # First line: record_name n_signals sample_rate n_samples
    parts = text[0].split()
    record_name = parts[0]
    n_signals = int(parts[1])
    sample_rate = int(parts[2])
    n_samples = int(parts[3])

    gains: list[float] = []
    baselines: list[int] = []
    signal_names: list[str] = []
    fmts: list[int] = []
    comments: dict[str, str] = {}

    for line in text[1:]:
        if line.startswith("#"):
            # Comment line — parse key:value
            m = re.match(r"#\s*(\w+)\s*:\s*(.*)", line)
            if m:
                comments[m.group(1).lower()] = m.group(2).strip()
            continue

        # Signal specification line
        # filename fmt gain(baseline)/units ...  signal_name
        sig_parts = line.split()
        fmts.append(int(sig_parts[1]))

        # Parse gain(baseline)/units
        gain_str = sig_parts[2]
        gain_match = re.match(r"([\d.eE+-]+)\((-?\d+)\)", gain_str)
        if gain_match:
            gains.append(float(gain_match.group(1)))
            baselines.append(int(gain_match.group(2)))
        else:
            # Fallback: gain without baseline
            gain_val = re.match(r"([\d.eE+-]+)", gain_str)
            gains.append(float(gain_val.group(1)) if gain_val else 200.0)
            baselines.append(0)

        signal_names.append(sig_parts[-1])

    return WFDBHeader(
        record_name=record_name,
        n_signals=n_signals,
        sample_rate=sample_rate,
        n_samples=n_samples,
        gains=gains,
        baselines=baselines,
        signal_names=signal_names,
        fmt=fmts,
        comments=comments,
    )


def read_wfdb_signal(dat_path: Path, header: WFDBHeader) -> NDArray[np.float64]:
    """Read a WFDB ``.dat`` binary signal file and convert to physical units.

    Parameters
    ----------
    dat_path : Path
        Path to the ``.dat`` binary file.
    header : WFDBHeader
        Parsed header for this record.

    Returns
    -------
    NDArray[np.float64]
        Array of shape ``(n_samples, n_signals)`` in physical units (g, deg/s).
    """
    # All LTMM files are format 16 (16-bit signed integer, little-endian)
    raw = np.fromfile(dat_path, dtype=np.int16)
    raw = raw.reshape(-1, header.n_signals)

    # Cast to int32 to avoid overflow when subtracting large baselines
    raw32 = raw.astype(np.int32)

    # Convert to physical units: physical = (digital - baseline) / gain
    physical = np.empty((raw32.shape[0], header.n_signals), dtype=np.float64)
    for i in range(header.n_signals):
        physical[:, i] = (raw32[:, i] - header.baselines[i]) / header.gains[i]

    return physical


# ── Clinical Data Loading ─────────────────────────────────────────────────────

# Map raw Excel column names -> clean snake_case names
_COLUMN_RENAME = {
    "#": "subject_id",
    "Gender(1-female, 0-male)": "gender",
    "Age": "age",
    "Year Fall ": "year_fall",
    "Year Fall": "year_fall",
    "6 Months Fall": "six_month_fall",
    "GDS": "gds",
    "ABC Tot %": "abc_pct",
    "SF-36": "sf36",
    "PASE": "pase",
    "MMSE": "mmse",
    "MoCa": "moca",
    "FAB": "fab",
    "TMTa": "tmta",
    "TMTb": "tmtb",
    "TUG": "tug",
    "FSST": "fsst",
    "BERG": "berg",
    "DGI": "dgi",
}


def load_clinical_data(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load and clean the clinical/demographic spreadsheet.

    Parameters
    ----------
    raw_dir : Path
        Directory containing ``ClinicalDemogData_COFL.xlsx``.

    Returns
    -------
    pd.DataFrame
        Cleaned clinical data with standardised column names.
    """
    filepath = raw_dir / CLINICAL_FILE
    if not filepath.exists():
        raise FileNotFoundError(
            f"Clinical data not found at {filepath}. "
            "Run `posturisk-fetch --lab-walks-only` first."
        )

    df = pd.read_excel(filepath)
    df = df.rename(columns=_COLUMN_RENAME)

    # Keep only the columns we need
    available = [c for c in CLINICAL_KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # Encode faller label from subject_id prefix
    df["is_faller"] = df["subject_id"].str.startswith("FL").astype(int)

    # Clean MMSE — convert "N/A" strings to NaN, then to numeric
    if "mmse" in df.columns:
        df["mmse"] = pd.to_numeric(df["mmse"], errors="coerce")

    # Convert all numeric-coercible columns
    for col in df.columns:
        if col not in ("subject_id",):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── Signal Feature Extraction ────────────────────────────────────────────────
# Moved to src/posturisk/features.py


# ── Subject-Level Pipeline ────────────────────────────────────────────────────


def _find_lab_walk_records(raw_dir: Path) -> list[Path]:
    """Find all lab-walk .hea files in raw_dir/LabWalks/."""
    lab_dir = raw_dir / "LabWalks"
    if not lab_dir.exists():
        return []
    return sorted(lab_dir.glob("*_base.hea"))


def _subject_id_from_record(record_name: str) -> str:
    """Convert a record name like ``co001_base`` to a subject ID like ``CO-001``."""
    m = re.match(r"(co|fl)(\d+)_base", record_name, re.IGNORECASE)
    if m:
        prefix = m.group(1).upper()
        num = m.group(2)
        return f"{prefix}-{num}"
    return record_name


def load_lab_walk_features(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load all lab-walk recordings and extract per-subject signal features.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the raw LTMM data.

    Returns
    -------
    pd.DataFrame
        One row per subject, with signal-derived features as columns.
    """
    hea_files = _find_lab_walk_records(raw_dir)
    if not hea_files:
        logger.warning("No lab-walk recordings found in %s/LabWalks/", raw_dir)
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for hea_path in hea_files:
        dat_path = hea_path.with_suffix(".dat")
        if not dat_path.exists():
            logger.warning("Missing .dat for %s — skipping", hea_path.name)
            continue

        try:
            header = parse_hea(hea_path)
            signals = read_wfdb_signal(dat_path, header)
            features = extract_signal_features(signals, fs=header.sample_rate)
            subject_id = _subject_id_from_record(header.record_name)
            features["subject_id"] = subject_id
            features["sample_rate"] = header.sample_rate
            features["n_samples"] = header.n_samples
            features["duration_s"] = header.n_samples / header.sample_rate
            rows.append(features)
        except Exception:
            logger.exception("Failed to process %s", hea_path.name)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Move subject_id to first column
    cols = ["subject_id"] + [c for c in df.columns if c != "subject_id"]
    return df[cols]


# ── Merging and Cleaning ─────────────────────────────────────────────────────


def merge_and_clean(
    clinical: pd.DataFrame,
    signal_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge clinical and signal feature data, handle missing values.

    Parameters
    ----------
    clinical : pd.DataFrame
        Cleaned clinical/demographic data (from :func:`load_clinical_data`).
    signal_features : pd.DataFrame
        Per-subject signal features (from :func:`load_lab_walk_features`).

    Returns
    -------
    pd.DataFrame
        Merged, cleaned DataFrame ready for modelling.
    """
    if signal_features.empty:
        logger.warning("No signal features -- returning clinical data only.")
        df = clinical.copy()
    else:
        # Left-join from signal features: keeps all subjects with recordings.
        # Clinical metadata only covers CO (non-faller) subjects in the LTMM
        # dataset, so FL subjects will have NaN for clinical columns.
        df = signal_features.merge(clinical, on="subject_id", how="left")
        logger.info(
            "Merged %d signal x %d clinical -> %d subjects",
            len(signal_features),
            len(clinical),
            len(df),
        )

    # Ensure faller label is always set from prefix (works even without
    # clinical data, since subject_id comes from the record filename)
    df["is_faller"] = df["subject_id"].str.upper().str.startswith("FL").astype(int)

    # Drop columns with >50% missing
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > 0.5].index.tolist()
    if drop_cols:
        logger.info("Dropping columns with >50%% missing: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    # Impute remaining numeric NaNs with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            n_missing = df[col].isnull().sum()
            logger.info(
                "Imputing %d missing values in '%s' with median=%.3f",
                n_missing,
                col,
                median_val,
            )
            df[col] = df[col].fillna(median_val)

    return df


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def run_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_filename: str = "features.csv",
) -> pd.DataFrame:
    """Execute the full preprocessing pipeline.

    1. Load clinical data
    2. Load and extract lab-walk signal features
    3. Merge, clean, impute
    4. Save to ``processed_dir / output_filename``

    Parameters
    ----------
    raw_dir : Path
        Directory with raw LTMM files.
    processed_dir : Path
        Directory to write the output CSV.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    pd.DataFrame
        The final processed DataFrame.
    """
    logger.info("Loading clinical data from %s …", raw_dir)
    clinical = load_clinical_data(raw_dir)
    logger.info("  → %d subjects, %d columns", len(clinical), len(clinical.columns))

    logger.info("Extracting lab-walk signal features …")
    signal_features = load_lab_walk_features(raw_dir)
    if not signal_features.empty:
        logger.info(
            "  → %d subjects, %d features", len(signal_features), len(signal_features.columns) - 1
        )

    logger.info("Merging and cleaning …")
    df = merge_and_clean(clinical, signal_features)
    logger.info("  → Final shape: %s", df.shape)

    # Save
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / output_filename
    df.to_csv(out_path, index=False)
    logger.info("Saved processed data to %s", out_path)

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        prog="posturisk-preprocess",
        description="Preprocess LTMM data: merge clinical + signal features.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Path to raw data directory.",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Path to output directory.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("  PostuRisk -- Preprocessing Pipeline")
    print("=" * 60)

    df = run_pipeline(raw_dir=args.raw, processed_dir=args.processed)

    print(f"\nDone! Output: {args.processed / 'features.csv'}")
    print(f"  Subjects: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Fallers:  {df['is_faller'].sum()} / {len(df)}")


if __name__ == "__main__":
    main()
