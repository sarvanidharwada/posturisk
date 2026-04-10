"""Smoke tests for posturisk.fetch_data."""

from pathlib import Path

from posturisk.fetch_data import (
    BASE_URL,
    DEFAULT_OUTPUT_DIR,
    METADATA_FILES,
    _build_file_list,
    _parse_args,
)


class TestBuildFileList:
    """Tests for _build_file_list()."""

    def test_lab_walks_only_excludes_3day_recordings(self):
        files = _build_file_list(include_3day=False)
        # Should contain lab-walk files and metadata, but no top-level .dat
        three_day = [f for f in files if not f.startswith("LabWalks/") and f.endswith(".dat")]
        assert len(three_day) == 0, f"Expected no 3-day .dat files, got: {three_day}"

    def test_full_includes_3day_recordings(self):
        files = _build_file_list(include_3day=True)
        three_day = [f for f in files if not f.startswith("LabWalks/") and f.endswith(".dat")]
        assert len(three_day) > 0, "Expected 3-day .dat files in full download"

    def test_metadata_always_included(self):
        for include_3day in (True, False):
            files = _build_file_list(include_3day=include_3day)
            for meta in METADATA_FILES:
                assert meta in files, f"Missing metadata file {meta!r}"

    def test_lab_walks_present(self):
        files = _build_file_list(include_3day=False)
        lab_files = [f for f in files if f.startswith("LabWalks/")]
        # Should have both .dat and .hea for both CO and FL subjects
        assert len(lab_files) > 20, f"Expected >20 lab-walk files, got {len(lab_files)}"

    def test_file_list_no_duplicates(self):
        for include_3day in (True, False):
            files = _build_file_list(include_3day=include_3day)
            assert len(files) == len(set(files)), "Duplicate entries in file list"


class TestCLI:
    """Tests for CLI argument parsing."""

    def test_defaults(self):
        args = _parse_args([])
        assert args.output_dir == DEFAULT_OUTPUT_DIR
        assert args.lab_walks_only is False
        assert args.no_verify is False
        assert args.force is False

    def test_lab_walks_only_flag(self):
        args = _parse_args(["--lab-walks-only"])
        assert args.lab_walks_only is True

    def test_custom_output_dir(self):
        args = _parse_args(["-o", "/tmp/test_data"])
        assert args.output_dir == Path("/tmp/test_data")

    def test_force_and_no_verify(self):
        args = _parse_args(["--force", "--no-verify"])
        assert args.force is True
        assert args.no_verify is True


class TestConstants:
    """Sanity checks on module constants."""

    def test_base_url_is_https(self):
        assert BASE_URL.startswith("https://")

    def test_base_url_ends_with_slash(self):
        assert BASE_URL.endswith("/")

    def test_default_output_dir_is_data_raw(self):
        assert DEFAULT_OUTPUT_DIR.parts[-2:] == ("data", "raw")
