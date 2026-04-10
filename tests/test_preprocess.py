"""Unit tests for posturisk.preprocess."""

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from posturisk.preprocess import (
    SIGNAL_NAMES,
    WFDBHeader,
    _rms,
    _subject_id_from_record,
    extract_signal_features,
    load_clinical_data,
    merge_and_clean,
    parse_hea,
    read_wfdb_signal,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_hea_file(tmp_path: Path) -> Path:
    """Create a minimal WFDB .hea file for testing."""
    content = dedent("""\
        test_rec 3 100 500
        test_rec.dat 16 200.0(0)/g 0 0 0 0 0 v-acceleration
        test_rec.dat 16 200.0(0)/g 0 0 0 0 0 ml-acceleration
        test_rec.dat 16 200.0(100)/deg/s 0 0 0 0 0 yaw-velocity
        #Age:75.0
        #Sex:F
    """)
    hea = tmp_path / "test_rec.hea"
    hea.write_text(content)
    return hea


@pytest.fixture()
def sample_dat_file(tmp_path: Path) -> Path:
    """Create a matching .dat binary file for 3 signals × 500 samples."""
    rng = np.random.default_rng(42)
    # 3 signals, 500 samples, 16-bit signed
    data = rng.integers(-1000, 1000, size=(500, 3), dtype=np.int16)
    dat = tmp_path / "test_rec.dat"
    data.tofile(dat)
    return dat


@pytest.fixture()
def sample_clinical_xlsx(tmp_path: Path) -> Path:
    """Create a minimal clinical Excel file for testing."""
    data = {
        "#": ["CO-001", "CO-002", "FL-001", "FL-002"],
        "Gender(1-female, 0-male)": [1, 0, 1, 0],
        "Age": [75.0, 80.0, 78.0, np.nan],
        "Year Fall ": [0, 0, 1, 1],
        "6 Months Fall": [0, 0, 1, 0],
        "GDS": [1, 2, np.nan, 3],
        "ABC Tot %": [88.0, 97.5, 82.0, 90.0],
        "SF-36": [82.0, 90.0, np.nan, 70.0],
        "PASE": [100.0, 110.0, 90.0, 120.0],
        "MMSE": [29, "N/A", 27, 28],
        "MoCa": [25, 26, 23, np.nan],
        "FAB": [16, 14, 15, 16],
        "TMTa": [60, 80, 90, 70],
        "TMTb": [140, 150, 160, 130],
        "TUG": [8.0, 7.5, 10.0, 9.0],
        "FSST": [9.0, 9.5, 11.0, 10.0],
        "BERG": [52, 54, 50, 53],
        "DGI": [22, 23, 20, 21],
    }
    df = pd.DataFrame(data)
    xlsx_path = tmp_path / "ClinicalDemogData_COFL.xlsx"
    df.to_excel(xlsx_path, index=False)
    return xlsx_path


# ── Tests: WFDB Reader ───────────────────────────────────────────────────────


class TestParseHea:
    """Tests for parse_hea()."""

    def test_parses_record_metadata(self, sample_hea_file: Path):
        header = parse_hea(sample_hea_file)
        assert header.record_name == "test_rec"
        assert header.n_signals == 3
        assert header.sample_rate == 100
        assert header.n_samples == 500

    def test_parses_signal_gains(self, sample_hea_file: Path):
        header = parse_hea(sample_hea_file)
        assert header.gains == [200.0, 200.0, 200.0]

    def test_parses_baselines(self, sample_hea_file: Path):
        header = parse_hea(sample_hea_file)
        assert header.baselines == [0, 0, 100]

    def test_parses_signal_names(self, sample_hea_file: Path):
        header = parse_hea(sample_hea_file)
        assert header.signal_names == ["v-acceleration", "ml-acceleration", "yaw-velocity"]

    def test_parses_comments(self, sample_hea_file: Path):
        header = parse_hea(sample_hea_file)
        assert header.comments["age"] == "75.0"
        assert header.comments["sex"] == "F"


class TestReadWfdbSignal:
    """Tests for read_wfdb_signal()."""

    def test_output_shape(self, sample_dat_file: Path, sample_hea_file: Path):
        header = parse_hea(sample_hea_file)
        signals = read_wfdb_signal(sample_dat_file, header)
        assert signals.shape == (500, 3)

    def test_output_dtype(self, sample_dat_file: Path, sample_hea_file: Path):
        header = parse_hea(sample_hea_file)
        signals = read_wfdb_signal(sample_dat_file, header)
        assert signals.dtype == np.float64

    def test_physical_units_conversion(self, tmp_path: Path):
        """Verify gain/baseline conversion: physical = (digital - baseline) / gain."""
        hea_content = dedent("""\
            simple 1 100 3
            simple.dat 16 100.0(50)/g 0 0 0 0 0 test-signal
        """)
        (tmp_path / "simple.hea").write_text(hea_content)

        # Digital values: [150, 250, 50]
        raw = np.array([150, 250, 50], dtype=np.int16)
        (tmp_path / "simple.dat").write_bytes(raw.tobytes())

        header = parse_hea(tmp_path / "simple.hea")
        signals = read_wfdb_signal(tmp_path / "simple.dat", header)

        # Expected: (150-50)/100=1.0, (250-50)/100=2.0, (50-50)/100=0.0
        np.testing.assert_allclose(signals[:, 0], [1.0, 2.0, 0.0])


# ── Tests: Feature Extraction ────────────────────────────────────────────────


class TestExtractSignalFeatures:
    """Tests for extract_signal_features()."""

    def test_returns_dict(self):
        rng = np.random.default_rng(0)
        signals = rng.standard_normal((1000, 3))
        features = extract_signal_features(signals, fs=100, signal_names=["a", "b", "c"])
        assert isinstance(features, dict)

    def test_feature_count(self):
        rng = np.random.default_rng(0)
        signals = rng.standard_normal((1000, 6))
        features = extract_signal_features(signals, fs=100)
        # 11 features per signal × 6 signals + 3 cross-channel = 69
        assert len(features) == 69

    def test_feature_names_prefix(self):
        signals = np.random.default_rng(0).standard_normal((500, 2))
        features = extract_signal_features(signals, fs=100, signal_names=["x", "y"])
        assert "x_mean" in features
        assert "x_std" in features
        assert "y_rms" in features
        assert "y_dom_freq" in features

    def test_no_nan_features(self):
        signals = np.random.default_rng(0).standard_normal((500, 3))
        features = extract_signal_features(signals, fs=100, signal_names=["a", "b", "c"])
        for name, val in features.items():
            assert np.isfinite(val), f"Non-finite value for feature '{name}': {val}"


class TestRms:
    """Tests for _rms()."""

    def test_known_value(self):
        x = np.array([1.0, -1.0, 1.0, -1.0])
        assert _rms(x) == pytest.approx(1.0)

    def test_zeros(self):
        assert _rms(np.zeros(100)) == pytest.approx(0.0)


# ── Tests: Clinical Data Loading ─────────────────────────────────────────────


class TestLoadClinicalData:
    """Tests for load_clinical_data()."""

    def test_loads_and_renames(self, sample_clinical_xlsx: Path):
        df = load_clinical_data(sample_clinical_xlsx.parent)
        assert "subject_id" in df.columns
        assert "gender" in df.columns
        assert "age" in df.columns

    def test_faller_label_encoding(self, sample_clinical_xlsx: Path):
        df = load_clinical_data(sample_clinical_xlsx.parent)
        assert df[df["subject_id"] == "CO-001"]["is_faller"].values[0] == 0
        assert df[df["subject_id"] == "FL-001"]["is_faller"].values[0] == 1

    def test_mmse_na_cleaned(self, sample_clinical_xlsx: Path):
        df = load_clinical_data(sample_clinical_xlsx.parent)
        # "N/A" string should become NaN
        co002 = df[df["subject_id"] == "CO-002"]
        assert pd.isna(co002["mmse"].values[0])

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Clinical data not found"):
            load_clinical_data(tmp_path)


# ── Tests: Subject ID Conversion ─────────────────────────────────────────────


class TestSubjectIdFromRecord:
    """Tests for _subject_id_from_record()."""

    def test_control_subject(self):
        assert _subject_id_from_record("co001_base") == "CO-001"

    def test_faller_subject(self):
        assert _subject_id_from_record("fl015_base") == "FL-015"

    def test_unknown_format(self):
        assert _subject_id_from_record("unknown") == "unknown"


# ── Tests: Merge and Clean ───────────────────────────────────────────────────


class TestMergeAndClean:
    """Tests for merge_and_clean()."""

    def test_imputes_missing_with_median(self):
        clinical = pd.DataFrame(
            {
                "subject_id": ["CO-001", "CO-002", "FL-001"],
                "age": [75.0, np.nan, 80.0],
                "is_faller": [0, 0, 1],
            }
        )
        df = merge_and_clean(clinical, pd.DataFrame())
        assert df["age"].isnull().sum() == 0
        # Median of 75 and 80 = 77.5
        assert df.loc[df["subject_id"] == "CO-002", "age"].values[0] == pytest.approx(77.5)

    def test_drops_mostly_missing_columns(self):
        clinical = pd.DataFrame(
            {
                "subject_id": ["A", "B", "C", "D"],
                "is_faller": [0, 0, 1, 1],
                "ok_col": [1.0, 2.0, 3.0, 4.0],
                "bad_col": [np.nan, np.nan, np.nan, 1.0],
            }
        )
        df = merge_and_clean(clinical, pd.DataFrame())
        assert "bad_col" not in df.columns
        assert "ok_col" in df.columns

    def test_left_join_with_signal_features(self):
        clinical = pd.DataFrame(
            {
                "subject_id": ["CO-001", "CO-002"],
                "is_faller": [0, 0],
                "age": [75.0, 80.0],
            }
        )
        signals = pd.DataFrame(
            {
                "subject_id": ["CO-001", "FL-001"],
                "v_acc_mean": [0.1, 0.2],
            }
        )
        df = merge_and_clean(clinical, signals)
        # Left join from signals: keeps CO-001 and FL-001
        assert len(df) == 2
        assert "v_acc_mean" in df.columns
        # FL-001 should be labelled as faller
        assert df.loc[df["subject_id"] == "FL-001", "is_faller"].values[0] == 1
        assert df.loc[df["subject_id"] == "CO-001", "is_faller"].values[0] == 0
