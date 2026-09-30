"""End-to-end regression test for the complete Webots mission.

The simulation is intentionally opt-in because a full run is much slower than
the deterministic unit suite and requires a local Webots installation.

Windows PowerShell::

    pwsh -NoProfile -File tests\run_webots_regression.ps1

On other platforms, set ``RUN_WEBOTS_REGRESSION=1`` and run this module. Set
``WEBOTS_EXECUTABLE`` when Webots is not installed in a standard location.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "worlds" / "amr_search_rescue.wbt"
RUN_FULL_REGRESSION = os.environ.get("RUN_WEBOTS_REGRESSION") == "1"


@dataclass(frozen=True)
class EvaluationResult:
    outcome: str
    elapsed_s: float | None = None
    phase: str | None = None
    targets_reached: int | None = None
    targets_required: int | None = None
    home_distance_m: float | None = None
    home_limit_m: float | None = None
    closest_person_m: float | None = None
    person_limit_m: float | None = None
    time_limit_s: float | None = None
    reason: str = ""
    structured: bool = False


def parse_evaluation(output: str) -> EvaluationResult | None:
    """Return the final evaluator result from combined Webots output."""
    structured_lines = [
        line.partition("EVALUATION_JSON:")[2].strip()
        for line in output.splitlines()
        if "EVALUATION_JSON:" in line
    ]
    if structured_lines:
        payload = json.loads(structured_lines[-1])
        return EvaluationResult(
            outcome=str(payload["outcome"]),
            elapsed_s=float(payload["elapsed_s"]),
            phase=str(payload["phase"]),
            targets_reached=int(payload["targets_reached"]),
            targets_required=int(payload["targets_required"]),
            home_distance_m=float(payload["home_distance_m"]),
            home_limit_m=float(payload["home_limit_m"]),
            closest_person_m=float(payload["closest_person_m"]),
            person_limit_m=float(payload["person_limit_m"]),
            time_limit_s=float(payload["time_limit_s"]),
            reason=str(payload.get("reason", "")),
            structured=True,
        )

    terminal_lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("EVALUATION:")
    ]
    if not terminal_lines:
        return None
    terminal = terminal_lines[-1]
    match = re.match(r"EVALUATION:\s+(PASS|FAIL|TIMEOUT)\s*\|?\s*(.*)", terminal)
    if match is None:
        return None
    return EvaluationResult(outcome=match.group(1), reason=match.group(2))


def find_webots() -> Path | None:
    configured = os.environ.get("WEBOTS_EXECUTABLE")
    candidates = [
        Path(configured) if configured else None,
        Path(r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"),
        Path(r"C:\Program Files\Webots\webots.exe"),
    ]
    discovered = shutil.which("webots")
    if discovered:
        candidates.append(Path(discovered))
    return next((path for path in candidates if path and path.is_file()), None)


def output_tail(output: str, line_count: int = 80) -> str:
    return "\n".join(output.splitlines()[-line_count:])


class EvaluationParserTests(unittest.TestCase):
    def test_parses_structured_evaluator_result(self):
        result = parse_evaluation(
            'noise\nEVALUATION_JSON: {"outcome":"PASS","elapsed_s":210.5,'
            '"phase":"COMPLETE","targets_reached":3,"targets_required":3,'
            '"home_distance_m":0.12,"home_limit_m":0.32,'
            '"closest_person_m":0.41,"person_limit_m":0.27,'
            '"time_limit_s":440.0,"reason":"mission complete"}\n'
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.structured)
        self.assertEqual(result.outcome, "PASS")
        self.assertEqual(result.targets_reached, 3)
        self.assertAlmostEqual(result.closest_person_m, 0.41)

    def test_legacy_failure_is_still_visible(self):
        result = parse_evaluation(
            "EVALUATION: FAIL | unsafe separation from person 2: 0.263m\n"
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.structured)
        self.assertEqual(result.outcome, "FAIL")
        self.assertIn("0.263m", result.reason)


@unittest.skipUnless(
    RUN_FULL_REGRESSION,
    "set RUN_WEBOTS_REGRESSION=1 to run the full Webots mission",
)
@unittest.skipIf(
    os.name == "nt",
    "use tests/run_webots_regression.ps1 on Windows to avoid Qt pipe issues",
)
class FullWebotsMissionRegression(unittest.TestCase):
    def test_all_targets_safe_and_home_before_deadline(self):
        webots = find_webots()
        if webots is None:
            self.fail(
                "Webots executable not found; set WEBOTS_EXECUTABLE to its absolute path"
            )

        command = [
            str(webots),
            "--batch",
            "--mode=fast",
            "--no-rendering",
            "--stdout",
            "--stderr",
            str(WORLD),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            captured = error.stdout or ""
            self.fail(
                "Webots regression exceeded the 900s wall-clock guard.\n"
                + output_tail(captured)
            )

        output = completed.stdout
        result = parse_evaluation(output)
        self.assertIsNotNone(
            result,
            "Evaluator produced no terminal result.\n" + output_tail(output),
        )
        assert result is not None
        self.assertEqual(
            result.outcome,
            "PASS",
            f"Mission outcome was {result.outcome}: {result.reason}\n"
            + output_tail(output),
        )
        self.assertTrue(
            result.structured,
            "Evaluator must emit EVALUATION_JSON for complete metric assertions.",
        )
        self.assertEqual(result.phase, "COMPLETE")
        self.assertEqual(result.targets_reached, result.targets_required)
        self.assertLessEqual(result.elapsed_s, result.time_limit_s)
        self.assertLess(result.home_distance_m, result.home_limit_m)
        self.assertGreaterEqual(result.closest_person_m, result.person_limit_m)
        self.assertEqual(
            completed.returncode,
            0,
            "Webots returned a non-zero process status despite evaluator PASS.",
        )


if __name__ == "__main__":
    unittest.main()
