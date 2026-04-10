"""Download the Long Term Movement Monitoring (LTMM) dataset from PhysioNet.

Dataset
-------
Long Term Movement Monitoring Database v1.0.0
https://physionet.org/content/ltmm/1.0.0/

71 community-living elders (age 65-87) wearing a 3D accelerometer on the
lower back.  Recordings include 3-day free-living data and 1-minute lab
walks.  Subjects are labelled as **fallers** (CO prefix = non-faller,
FL prefix = faller) based on self-reported fall history (≥ 2 falls/year).

Files are in MIT/WFDB format (.dat + .hea) plus Excel clinical metadata.

License: Open Data Commons Attribution License v1.0
DOI:     https://doi.org/10.13026/C2S59C

Usage
-----
    # Download everything (~20 GB uncompressed)
    python -m posturisk.fetch_data

    # Download lab walks + metadata only (~50 MB)
    python -m posturisk.fetch_data --lab-walks-only

    # Or via the installed console script
    posturisk-fetch --lab-walks-only
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://physionet.org/files/ltmm/1.0.0/"

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/posturisk
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

# Metadata / ancillary files (always downloaded)
METADATA_FILES = [
    "RECORDS",
    "SHA256SUMS.txt",
    "ClinicalDemogData_COFL.xlsx",
    "ReportHome75h.xlsx",
]

# Non-faller 3-day recordings (CO = control)
_CO_IDS = [
    "001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
    "011", "012", "013", "014", "015", "016", "017", "018", "019", "020",
    "021", "022", "023", "024", "025", "027", "028", "029", "030", "031",
    "032", "035", "036", "037", "038", "039", "040", "041", "042", "044",
]

# Faller 3-day recordings (FL = faller)
_FL_IDS = [
    "001", "004", "005", "006", "007", "008", "009", "010", "011", "014",
    "015", "016", "018", "019", "020", "021", "022", "023", "024", "025",
    "026", "027", "028", "029", "030", "031", "032", "033", "034", "035",
    "036",
]


def _build_file_list(include_3day: bool = True) -> list[str]:
    """Return the list of remote file paths to download.

    Parameters
    ----------
    include_3day : bool
        If True, include the large 3-day free-living recordings.
        If False, only metadata and lab-walk files are included.
    """
    files: list[str] = list(METADATA_FILES)

    # Lab-walk files (always included — they live in the LabWalks/ subdirectory)
    for prefix, ids in [("CO", _CO_IDS), ("FL", _FL_IDS)]:
        for sid in ids:
            files.append(f"LabWalks/{prefix}{sid}.dat")
            files.append(f"LabWalks/{prefix}{sid}.hea")

    if include_3day:
        for prefix, ids in [("CO", _CO_IDS), ("FL", _FL_IDS)]:
            for sid in ids:
                files.append(f"{prefix}{sid}.dat")
                files.append(f"{prefix}{sid}.hea")
        # A few subjects have supplemental segment files
        for extra in ["CO028_n.dat", "CO029_n.dat", "CO038_2.dat"]:
            files.append(extra)

    return files


# ── Download helpers ──────────────────────────────────────────────────────────


def _download_file(
    url: str,
    dest: Path,
    *,
    chunk_size: int = 1024 * 256,
    timeout: int = 30,
) -> None:
    """Stream-download *url* to *dest* with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with (
            open(dest, "wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                leave=False,
            ) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))


def _verify_sha256(filepath: Path, expected_hash: str) -> bool:
    """Return True if *filepath* matches *expected_hash* (hex-encoded SHA-256)."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 256), b""):
            sha.update(block)
    return sha.hexdigest() == expected_hash


def _load_checksums(output_dir: Path) -> dict[str, str]:
    """Parse the SHA256SUMS.txt file (if present) into a {filename: hash} map."""
    sums_file = output_dir / "SHA256SUMS.txt"
    checksums: dict[str, str] = {}
    if sums_file.exists():
        for line in sums_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                checksums[parts[1]] = parts[0]
    return checksums


# ── Main logic ────────────────────────────────────────────────────────────────


def fetch(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    lab_walks_only: bool = False,
    verify: bool = True,
    skip_existing: bool = True,
) -> None:
    """Download the LTMM dataset to *output_dir*.

    Parameters
    ----------
    output_dir : Path
        Local directory to save files into.
    lab_walks_only : bool
        If True, skip the large 3-day recordings and download only
        the 1-minute lab walks plus clinical metadata (~50 MB).
    verify : bool
        If True, verify SHA-256 checksums after download.
    skip_existing : bool
        If True, skip files that already exist locally.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_list = _build_file_list(include_3day=not lab_walks_only)
    logger.info(
        "Downloading %d files to %s (lab_walks_only=%s)",
        len(file_list),
        output_dir,
        lab_walks_only,
    )

    failed: list[str] = []
    skipped = 0

    for relpath in tqdm(file_list, desc="Overall progress", unit="file"):
        dest = output_dir / relpath
        url = urljoin(BASE_URL, relpath)

        if skip_existing and dest.exists():
            skipped += 1
            continue

        try:
            _download_file(url, dest)
        except (requests.RequestException, OSError) as exc:
            logger.warning("Failed to download %s: %s", relpath, exc)
            failed.append(relpath)

    # ── Checksum verification ─────────────────────────────────────────────
    if verify:
        checksums = _load_checksums(output_dir)
        if checksums:
            logger.info("Verifying checksums for %d files …", len(checksums))
            bad: list[str] = []
            for fname, expected in checksums.items():
                fpath = output_dir / fname
                if fpath.exists() and not _verify_sha256(fpath, expected):
                    bad.append(fname)
            if bad:
                logger.warning(
                    "Checksum mismatch for %d file(s): %s",
                    len(bad),
                    ", ".join(bad[:5]),
                )
            else:
                logger.info("All checksums verified ✓")
        else:
            logger.info("No checksum file found — skipping verification.")

    # ── Summary ───────────────────────────────────────────────────────────
    total = len(file_list)
    downloaded = total - skipped - len(failed)
    print(
        f"\nDone — {downloaded} downloaded, {skipped} skipped (existing), "
        f"{len(failed)} failed out of {total} total files."
    )
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  • {f}")
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="posturisk-fetch",
        description=(
            "Download the PhysioNet LTMM (Long Term Movement Monitoring) "
            "dataset for fall-risk classification."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save downloaded files (default: data/raw/).",
    )
    parser.add_argument(
        "--lab-walks-only",
        action="store_true",
        help=(
            "Download only the 1-minute lab walk recordings and metadata "
            "(~50 MB) instead of the full 3-day recordings (~20 GB)."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 checksum verification after download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``posturisk-fetch`` console script."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("  PostuRisk — LTMM Dataset Downloader")
    print("  https://physionet.org/content/ltmm/1.0.0/")
    print("=" * 60)

    fetch(
        output_dir=args.output_dir,
        lab_walks_only=args.lab_walks_only,
        verify=not args.no_verify,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
