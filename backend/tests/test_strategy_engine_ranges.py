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

REVERSAL_PRICES = [100, 98, 96, 99, 105, 103, 101, 106, 112, 108, 104, 109, 110, 106, 102, 98, 94]

DIRECT_COUNTER_BREAK_WITHOUT_SHIFT = [100, 98, 96, 99, 105, 103, 101, 106, 112, 108, 104, 100, 96, 92, 90]


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
            "protected_high",
            "protected_low",
            "activeWeakHighs",
            "activeWeakLows",
            "strongHighs",
            "strongLows",
            "currentLegType",
            "poi_allowed",
            "shift_detected",
            "latest_shift",
            "swing_strength_map",
        }
        self.assertEqual(set(official), expected_keys)
        self.assertIn(official["trend_direction"], {"BULLISH", "BEARISH", "NEUTRAL"})
        self.assertIn(official["structure_bias"], {"BULLISH", "BEARISH", "NEUTRAL"})
        self.assertIn(official["currentLegType"], {"WEAK", "STRUCTURAL"})
        self.assertIsInstance(official["poi_allowed"], bool)
        self.assertEqual(official["poi_allowed"], official["currentLegType"] == "STRUCTURAL")
        self.assertTrue({"high", "low"}.issubset(official["current_external_swing"]))
        self.assertTrue(official["protected_high"] is None or isinstance(official["protected_high"], dict))
        self.assertTrue(official["protected_low"] is None or isinstance(official["protected_low"], dict))
        self.assertIsInstance(official["activeWeakHighs"], list)
        self.assertIsInstance(official["activeWeakLows"], list)
        self.assertIsInstance(official["strongHighs"], list)
        self.assertIsInstance(official["strongLows"], list)
        self.assertIsInstance(official["swing_strength_map"], dict)
        self.assertIsInstance(official["shift_detected"], bool)
        self.assertTrue(official["latest_shift"] is None or isinstance(official["latest_shift"], dict))

        if official["last_bos"]:
            self.assertEqual(set(official["last_bos"]), {"type", "level", "time", "direction", "index", "timeframe", "confirmed"})
            self.assertEqual(official["last_bos"]["type"], "BOS")
            self.assertIs(official["last_bos"]["confirmed"], True)

    def test_unbroken_swings_are_not_scored_before_bos_close(self) -> None:
        analysis = self.run_phase_one(STOP_HUNT_PRICES)
        pending = [s for s in analysis["swings"] if not s.get("score_ready")]

        self.assertGreater(len(pending), 0)
        self.assertTrue(all(s.get("strength_score") == 0.0 for s in pending))

    def test_choch_requires_confirmed_shift(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)
        state = analysis["phase_1"]
        choch_events = [event for event in analysis["structure_events"] if event["event"] == "CHoCH"]

        self.assertGreater(len(state["shift_events"]), 0)
        self.assertGreater(len(choch_events), 0)
        self.assertTrue(all(event.get("shift") for event in choch_events))
        self.assertTrue(all(event.get("shift_index") is not None for event in choch_events))
        self.assertTrue(all(event["shift_index"] < event["index"] for event in choch_events))
        self.assertTrue(all((event.get("shift") or {}).get("confirmation_index") <= event["index"] for event in choch_events))
        self.assertTrue(state["choch_requires_shift"])
        self.assertEqual(state["official_outputs"]["latest_shift"], state["latest_shift"])

    def test_shift_detected_is_only_active_before_followup_structure_event(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)
        state = analysis["phase_1"]

        self.assertIsNotNone(state["latest_shift"])
        self.assertGreater(len([event for event in analysis["structure_events"] if event["event"] == "CHoCH"]), 0)
        self.assertFalse(state["shift_detected"])
        self.assertFalse(state["official_outputs"]["shift_detected"])

    def test_weekly_and_daily_trend_components_are_reported(self) -> None:
        analysis = self.run_phase_one(STRUCTURAL_PRICES)
        components = analysis["phase_1"]["trend_components"]
        frames = {component["timeframe"] for component in components}

        self.assertIn("weekly", frames)
        self.assertIn("daily", frames)
        self.assertEqual(analysis["phase_1"]["trend_timeframe"], "weekly+daily")

    def test_weekly_and_daily_trend_do_not_feed_structure_outputs(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)
        state = analysis["phase_1"]

        self.assertGreater(len(state["trend_events"]), 0)
        self.assertTrue(all(event["timeframe"] in {"external_4h", "internal_1h"} for event in analysis["structure_events"]))
        self.assertTrue(all(item["timeframe"] in {"external_4h", "internal_1h"} for item in analysis["stop_hunts"]))
        self.assertTrue(all(not key.startswith(("weekly:", "daily:")) for key in state["swing_strength_map"]))
        if state["last_bos"]:
            self.assertIn(state["last_bos"]["timeframe"], {"external_4h", "internal_1h"})
        self.assertTrue(all(event["timeframe"] in {"external_4h", "internal_1h"} for event in state["shift_events"]))

    def test_peaks_and_valleys_use_n_candles_left_and_right(self) -> None:
        analysis = self.run_phase_one(STRUCTURAL_PRICES)
        market_swings = [s for s in analysis["swings"] if s["timeframe"] in {"external_4h", "internal_1h"}]

        self.assertGreater(len(market_swings), 0)
        self.assertTrue(any(s["kind"] == "high" for s in market_swings))
        self.assertTrue(any(s["kind"] == "low" for s in market_swings))
        self.assertTrue(all(s["length"] == PARAMS["n_candles"] for s in market_swings))
        self.assertTrue(all(s["confirmed_after"] == s["index"] + PARAMS["n_candles"] for s in market_swings))
        self.assertTrue(all("local" in s["detection_rule"] for s in market_swings))

    def test_internal_and_external_swing_scope_contract(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)
        external_swings = [s for s in analysis["swings"] if s["timeframe"] == "external_4h"]

        self.assertGreater(len(external_swings), 0)
        self.assertTrue(any(s["structure_scope"] == "External" and s["bos_produced"] for s in external_swings))
        self.assertTrue(any(s["structure_scope"] == "Internal" and not s["bos_produced"] for s in external_swings))
        self.assertFalse(any(s["structure_scope"] == "External" and not s["bos_produced"] for s in external_swings))

    def test_swing_validation_requires_liquidity_and_bos_production(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)
        swings = analysis["swings"]
        valid = [s for s in swings if s.get("valid_swing")]
        invalid = [s for s in swings if not s.get("valid_swing")]

        self.assertGreater(len(valid), 0)
        self.assertGreater(len(invalid), 0)
        self.assertTrue(all(s["liquidity_taken"] and s["bos_produced"] for s in valid))
        self.assertTrue(all((not s["liquidity_taken"]) or (not s["bos_produced"]) for s in invalid))

    def test_swing_strength_score_classes_follow_thresholds(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)

        for swing in analysis["swings"]:
            score = float(swing.get("strength_score", 0.0))
            strength_entry = analysis["phase_1"]["swing_strength_map"].get(swing["id"])
            self.assertIsInstance(strength_entry, dict)
            self.assertEqual(strength_entry["score"], swing["strength_score"])
            self.assertEqual(strength_entry["class"], swing["strength_class"])
            self.assertEqual(
                set(swing["score_weights"]),
                {"liquidity_taken", "move_size", "bos_produced", "continuation_quality", "structural_impact"},
            )
            if score >= 0.70:
                self.assertEqual(swing["strength_class"], "STRUCTURAL")
            elif score >= 0.40:
                self.assertEqual(swing["strength_class"], "MAJOR")
            else:
                self.assertEqual(swing["strength_class"], "MINOR")

        self.assertTrue(any(s["strength_class"] == "STRUCTURAL" for s in analysis["swings"]))
        self.assertTrue(any(s["strength_class"] == "MINOR" for s in analysis["swings"]))

    def test_bos_updates_last_bos_and_stop_hunt_does_not(self) -> None:
        structural = self.run_phase_one(STRUCTURAL_PRICES)
        stop_hunt = self.run_phase_one(STOP_HUNT_PRICES)

        self.assertIsNotNone(structural["phase_1"]["last_bos"])
        self.assertEqual(structural["phase_1"]["last_bos"]["type"], "BOS")
        self.assertEqual(structural["phase_1"]["last_bos"]["direction"], "bullish")
        self.assertEqual(stop_hunt["phase_1"]["last_bos"], None)
        self.assertEqual(stop_hunt["structure_events"], [])
        self.assertGreater(len(stop_hunt["stop_hunts"]), 0)

    def test_current_external_swing_updates_only_after_confirmed_bos(self) -> None:
        no_bos = self.run_phase_one(STOP_HUNT_PRICES)
        structural = self.run_phase_one(STRUCTURAL_PRICES)

        self.assertEqual(no_bos["phase_1"]["current_external_swing"], {"high": None, "low": None})
        current = structural["phase_1"]["current_external_swing"]
        last_external_bos = next(event for event in reversed(structural["structure_events"]) if event["timeframe"] == "external_4h" and event["event"] == "BOS")
        self.assertIsNone(current["high"])
        self.assertIsNotNone(current["low"])
        self.assertEqual(current["low"]["id"], last_external_bos["responsible_structure"]["id"])
        self.assertTrue(current["low"]["bos_produced"])
        self.assertEqual(current["low"]["structure_scope"], "External")

    def test_opposite_break_without_shift_is_not_bos_or_choch(self) -> None:
        analysis = self.run_phase_one(DIRECT_COUNTER_BREAK_WITHOUT_SHIFT)
        events = analysis["structure_events"]
        rejected = [item for item in analysis["stop_hunts"] if item.get("classification") == "invalid_counter_break_without_shift"]

        self.assertGreater(len(rejected), 0)
        self.assertTrue(all(event["direction"] == "bullish" for event in events))
        self.assertFalse(any(event["event"] == "CHoCH" for event in events))
        self.assertEqual(analysis["phase_1"]["structure_bias"], "BULLISH")

    def test_weak_and_strong_high_low_outputs_are_separate_from_hh_hl_labels(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)
        swings = analysis["swings"]
        weak_ids = {item["id"] for item in analysis["phase_1"]["activeWeakHighs"] + analysis["phase_1"]["activeWeakLows"]}
        strong_swings = [s for s in swings if s.get("structure_role") in {"STRONG_HIGH", "STRONG_LOW"}]

        self.assertGreater(len(weak_ids), 0)
        self.assertGreater(len(strong_swings), 0)
        self.assertTrue(all(s["id"] not in weak_ids for s in strong_swings))
        self.assertTrue(any(s["structure_role"] == "STRONG_HIGH" and s["label"] in {"HH", "LH"} for s in strong_swings))
        self.assertTrue(any(s["structure_role"] == "STRONG_LOW" and s["label"] in {"HL", "LL"} for s in strong_swings))
        self.assertGreater(len(analysis["phase_1"]["strongHighs"]), 0)
        self.assertGreater(len(analysis["phase_1"]["strongLows"]), 0)

    def test_mss_stays_inactive_without_phase_two_sweep(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)
        state = analysis["phase_1"]

        self.assertEqual(analysis["sweeps"], [])
        self.assertFalse(state["mss_detected"])
        self.assertEqual(state["mss_events"], [])
        self.assertIn("Phase 2 Sweep", state["mss_note"])

    def test_structural_leg_allows_only_permission_not_poi_creation(self) -> None:
        structural = self.run_phase_one(STRUCTURAL_PRICES)
        weak = self.run_phase_one(STOP_HUNT_PRICES)

        self.assertEqual(structural["phase_1"]["currentLegType"], "STRUCTURAL")
        self.assertTrue(structural["phase_1"]["poi_allowed"])
        self.assertEqual(structural["pois"], [])
        self.assertEqual(weak["phase_1"]["currentLegType"], "WEAK")
        self.assertFalse(weak["phase_1"]["poi_allowed"])
        self.assertEqual(weak["pois"], [])


if __name__ == "__main__":
    unittest.main()
