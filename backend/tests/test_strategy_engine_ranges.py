import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.strategy_engine import run_analysis


PARAMS = {
    "n_candles": 2,
    "major_length": 50,
    "internal_length": 20,
    "micro_length": 6,
    "break_confirmation": "close",
    "min_fvg_size": 0.0,
    "retest_tolerance_pct": 0.0015,
}


def make_candles(prices: list[float], *, hours: int = 1) -> list[dict]:
    base = datetime(2024, 1, 1)
    candles: list[dict] = []
    for index, price in enumerate(prices):
        candles.append(
            {
                "timestamp": (base + timedelta(hours=index * hours)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price,
                "volume": 1,
            }
        )
    return candles


STRUCTURAL_PRICES = [
    100,
    98,
    96,
    98,
    103,
    108,
    105,
    102,
    106,
    111,
    116,
    114,
    112,
    115,
    120,
    125,
    122,
    119,
    123,
    128,
    133,
    130,
    127,
    131,
    136,
    141,
    138,
    135,
    139,
    144,
]


STOP_HUNT_PRICES = [100, 98, 96, 98, 103, 108, 105, 102, 109, 107, 106, 105, 104, 103]


class StrategyPhaseOneTests(unittest.TestCase):
    def run_phase_one(self, prices: list[float]) -> dict:
        internal = make_candles(prices)
        daily = make_candles(prices, hours=24)
        return run_analysis(
            candles=internal,
            params=PARAMS,
            external_candles=internal,
            internal_candles=internal,
            daily_candles=daily,
        )

    def test_phase_one_does_not_build_later_phase_outputs(self) -> None:
        analysis = self.run_phase_one(STRUCTURAL_PRICES)

        self.assertEqual(analysis["sweeps"], [])
        self.assertEqual(analysis["idms"], [])
        self.assertEqual(analysis["external_ranges"], [])
        self.assertEqual(analysis["ranges"], [])
        self.assertEqual(analysis["pois"], [])
        self.assertEqual(analysis["setups"], [])
        self.assertEqual(analysis["liquidity_targets"], [])
        self.assertEqual(analysis["summary"]["pois"], 0)
        self.assertEqual(analysis["summary"]["ranges"], 0)
        self.assertEqual(analysis["summary"]["sweeps"], 0)

    def test_poi_allowed_is_only_a_green_red_permission(self) -> None:
        analysis = self.run_phase_one(STRUCTURAL_PRICES)
        state = analysis["strategy_state"]

        self.assertIs(state["poi_allowed"], True)
        self.assertEqual(state["currentLegType"], "STRUCTURAL")
        self.assertEqual(state["gate_status"], "open")
        self.assertIn("permission only", state["poi_allowed_meaning"])
        self.assertEqual(analysis["pois"], [], "Phase 1 must not create actual POI records")
        self.assertEqual(state["official_outputs"]["poi_allowed"], state["currentLegType"] == "STRUCTURAL")

    def test_weak_leg_blocks_phase_three_permission(self) -> None:
        analysis = self.run_phase_one(STOP_HUNT_PRICES)
        state = analysis["strategy_state"]

        self.assertIs(state["poi_allowed"], False)
        self.assertEqual(state["currentLegType"], "WEAK")
        self.assertEqual(state["gate_status"], "blocked")
        self.assertEqual(state["gate_reason"], "weak_leg_no_permission_for_phase3_poi")
        self.assertGreater(len(analysis["stop_hunts"]), 0)
        self.assertTrue(all(record["updates_structure"] is False for record in analysis["stop_hunts"]))

    def test_structure_events_require_body_close_plus_continuation(self) -> None:
        analysis = self.run_phase_one(STRUCTURAL_PRICES)

        self.assertGreater(len(analysis["structure_events"]), 0)
        self.assertTrue(all(event["body_close_required"] for event in analysis["structure_events"]))
        self.assertTrue(all(event["continuation_required"] for event in analysis["structure_events"]))
        self.assertTrue(all(event["continuation_confirmed"] for event in analysis["structure_events"]))
        self.assertTrue(
            any(swing.get("strength_class") == "STRUCTURAL" for swing in analysis["swings"]),
            "At least one swing should be scored structural when it produces a confirmed BOS",
        )

    def test_phase_one_official_outputs_match_pdf_contract(self) -> None:
        analysis = self.run_phase_one(STRUCTURAL_PRICES)
        state = analysis["phase_1"]
        official = state["official_outputs"]

        expected_keys = {
            "trend_direction",
            "structure_bias",
            "last_bos",
            "current_external_swing",
            "activeWeakHighs",
            "activeWeakLows",
            "currentLegType",
            "poi_allowed",
            "shift_detected",
            "swing_strength_map",
        }
        self.assertEqual(set(official), expected_keys)
        self.assertIn(official["trend_direction"], {"BULLISH", "BEARISH", "NEUTRAL"})
        self.assertIn(official["structure_bias"], {"BULLISH", "BEARISH", "NEUTRAL"})
        self.assertIn(official["currentLegType"], {"WEAK", "STRUCTURAL"})
        self.assertIsInstance(official["poi_allowed"], bool)
        self.assertEqual(official["poi_allowed"], official["currentLegType"] == "STRUCTURAL")
        self.assertTrue({"high", "low"}.issubset(official["current_external_swing"]))
        self.assertIsInstance(official["activeWeakHighs"], list)
        self.assertIsInstance(official["activeWeakLows"], list)
        self.assertIsInstance(official["swing_strength_map"], dict)
        self.assertIsInstance(official["shift_detected"], bool)

        if official["last_bos"]:
            self.assertEqual(set(official["last_bos"]), {"type", "level", "time", "direction", "index", "timeframe", "confirmed"})
            self.assertEqual(official["last_bos"]["type"], "BOS")
            self.assertIs(official["last_bos"]["confirmed"], True)

    def test_unbroken_swings_are_not_scored_before_bos_close(self) -> None:
        analysis = self.run_phase_one(STOP_HUNT_PRICES)
        pending = [s for s in analysis["swings"] if not s.get("score_ready")]

        self.assertGreater(len(pending), 0)
        self.assertTrue(all(s.get("strength_score") == 0.0 for s in pending))


if __name__ == "__main__":
    unittest.main()
