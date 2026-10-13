from __future__ import annotations

import csv
import json
import subprocess
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"


def git_blob(path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"HEAD:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_blob_size(blob: str) -> int:
    result = subprocess.run(
        ["git", "cat-file", "-s", blob],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def csv_shape(path: Path, day_column: str | None = None) -> tuple[int, Counter[str]]:
    rows = 0
    days: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError(f"Missing CSV header: {path}")
        for row in reader:
            rows += 1
            if day_column is not None:
                days[row[day_column]] += 1
    return rows, days


class RepositoryIntegrityTests(unittest.TestCase):
    def test_manifested_artifacts_match_archived_git_objects(self) -> None:
        manifest = json.loads((CODE / "ARTIFACTS.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        for artifact in manifest["artifacts"]:
            path = artifact["path"]
            expected_blob = artifact["git_blob"]
            actual_blob = git_blob(path)
            self.assertEqual(actual_blob, expected_blob, path)
            self.assertEqual(git_blob_size(actual_blob), artifact["bytes"], path)
            self.assertLess(artifact["bytes"], 100 * 1024 * 1024, path)

    def test_restricted_and_raw_track_inputs_are_not_committed(self) -> None:
        forbidden = [
            CODE / "data" / "BIMTRA.csv",
            CODE / "data" / "raw-data" / "traffic-data" / "KPI08_2022.rda",
            CODE / "data" / "raw-data" / "traffic-data" / "KPI08_2023.rda",
            CODE / "data" / "raw-data" / "traffic-data" / "KPI17_2022.rda",
            CODE / "data" / "raw-data" / "traffic-data" / "KPI17_2023.rda",
        ]
        for path in forbidden:
            self.assertFalse(path.exists(), str(path.relative_to(ROOT)))

        self.assertEqual(list(ROOT.glob("**/BIMTRA.csv")), [])
        self.assertEqual(list(ROOT.glob("**/KPI08_*.rda")), [])
        self.assertEqual(list(ROOT.glob("**/KPI17_*.rda")), [])

    def test_conventional_sbgr_archive_has_expected_operational_grid(self) -> None:
        rows, days = csv_shape(CODE / "data" / "data.csv", "day")
        self.assertEqual(rows, 66816)
        self.assertEqual(len(days), 696)
        self.assertTrue(all(count == 96 for count in days.values()))

        # The archived CSV stores operational-day labels without zero padding.
        # These five days are present in the archived input even though the
        # retained historical R script currently excludes them.
        for day in (
            "2023-4-24",
            "2023-4-25",
            "2023-4-26",
            "2023-4-27",
            "2023-4-28",
        ):
            self.assertEqual(days[day], 96)

    def test_uam_archive_uses_same_96_slot_operational_grid(self) -> None:
        rows, days = csv_shape(CODE / "data_uam.csv", "day")
        self.assertEqual(rows, 66816)
        self.assertEqual(len(days), 696)
        self.assertTrue(all(count == 96 for count in days.values()))

    def test_flight_level_arrival_tables_have_expected_size_and_fields(self) -> None:
        raw_rows, _ = csv_shape(CODE / "data" / "arrival_data.csv")
        corrected_rows, _ = csv_shape(
            CODE / "arrival_time_estimation" / "arrival_data_with_estimated_times_40.csv"
        )
        self.assertEqual(raw_rows, 235075)
        self.assertEqual(corrected_rows, 235075)

        with (CODE / "arrival_time_estimation" / "arrival_data_with_estimated_times_40.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            header = set(next(csv.reader(handle)))
        required = {
            "flight_number",
            "scheduled_departure_time",
            "estimated_real_arrival_time",
            "estimated_real_arrival_40_time",
            "day_slot_40",
            "slot_time_40",
            "scheduled_arrivals_40",
        }
        self.assertTrue(required.issubset(header), required - header)

    def test_required_experiment_sources_are_present(self) -> None:
        required = [
            CODE / "maskable_ppo_test.py",
            CODE / "maskable_ppo_test_uam.py",
            CODE / "improved_gym_sdcboptimizer_specific.py",
            CODE / "improved_gym_sdcboptimizer_specific_uam.py",
            CODE / "generate_uam_synthetic_data_bimtra_based.py",
            CODE / "profile_implementation" / "routes_to_scn.py",
            CODE / "requirements.txt",
        ]
        for path in required:
            self.assertTrue(path.is_file(), str(path.relative_to(ROOT)))

    def test_repository_metadata_points_to_current_name(self) -> None:
        expected = "https://github.com/Orbolato/Strategic-DCB-RL"
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        page = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn(expected, citation)
        self.assertIn(expected, page)
        self.assertNotIn("dasc2026-paper-276", citation + readme + page)

    def test_no_repository_license_is_asserted(self) -> None:
        names = {path.name.lower() for path in ROOT.iterdir() if path.is_file()}
        self.assertFalse(any(name == "license" or name.startswith("license.") for name in names))


if __name__ == "__main__":
    unittest.main(verbosity=2)
