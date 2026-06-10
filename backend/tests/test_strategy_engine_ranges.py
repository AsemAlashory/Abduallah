import sys
import unittest
from collections import Counter
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.market_data import fetch_local_nq_dataset
from app.services.strategy_engine import run_analysis


PARAMS = {
    "major_length": 50,
    "internal_length": 20,
    "micro_length": 6,
    "break_confirmation": "close",
    "min_fvg_size": 0.0,
    "sweep_lookback": 30,
    "retest_tolerance_pct": 0.0015,
}


class StrategyRangeLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset = fetch_local_nq_dataset()
        cls.analysis = run_analysis(
            dataset["internal_candles"],
            PARAMS,
            dataset["external_candles"],
            dataset["internal_candles"],
        )

    def test_each_sweep_seeds_only_one_internal_range_cycle(self) -> None:
        counts = Counter((rng["direction"], rng["a"]["index"]) for rng in self.analysis["ranges"])
        duplicates = {key: count for key, count in counts.items() if count > 1}
        self.assertEqual(
            duplicates,
            {},
            f"Internal ranges must not duplicate the same sweep A point: {duplicates}",
        )

    def test_each_sweep_seeds_only_one_external_range_cycle(self) -> None:
        counts = Counter((rng["direction"], rng["a"]["index"]) for rng in self.analysis["external_ranges"])
        duplicates = {key: count for key, count in counts.items() if count > 1}
        self.assertEqual(
            duplicates,
            {},
            f"External ranges must not duplicate the same sweep A point: {duplicates}",
        )

    def test_validated_ranges_require_a_swept_idm(self) -> None:
        validated = [
            rng
            for rng in self.analysis["ranges"] + self.analysis["external_ranges"]
            if str(rng.get("status", "")).startswith("validated")
        ]
        self.assertGreater(len(validated), 0, "Expected at least one validated range in the fixture dataset")

        for rng in validated:
            self.assertIsNotNone(rng.get("validation_index"), "Validated range must store validation index")
            self.assertTrue((rng.get("b") or {}).get("swept"), "Validated range must have swept IDM at point B")
            self.assertEqual(
                rng.get("validation_index"),
                (rng.get("b") or {}).get("swept_at"),
                "Validated range should lock only when the linked IDM sweep occurs",
            )


if __name__ == "__main__":
    unittest.main()
