import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.strategy_engine import StrategyEngine, run_analysis


PARAMS = {
    "n_candles": 2,
    "major_length": 2,
    "internal_length": 2,
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


def make_ohlc(rows: list[tuple[float, float, float, float]], *, hours: int = 1) -> list[dict]:
    base = datetime(2024, 1, 1)
    candles: list[dict] = []
    for index, (open_, high, low, close) in enumerate(rows):
        candles.append(
            {
                "timestamp": (base + timedelta(hours=index * hours)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
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
        external = make_candles(prices, hours=4)
        internal = make_candles(prices, hours=1)
        daily = make_candles(prices, hours=24)
        return run_analysis(
            candles=external,
            params=PARAMS,
            external_candles=external,
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
            "At least one swing should be classified structural when it produces a confirmed BOS",
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
        self.assertTrue(state["swing_settings"]["feeds_separated"])
        self.assertIn("never auto-reused", state["swing_settings"]["same_feed_guard"])
        self.assertEqual(state["timeframe_candle_counts"]["chart"], state["timeframe_candle_counts"]["external"])


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
        self.assertTrue(all((event.get("source_shift_index", event["shift_index"]) < event.get("source_index", event["index"])) for event in choch_events))
        self.assertTrue(all((event.get("shift") or {}).get("confirmation_index") <= event.get("source_index", event["index"]) for event in choch_events))
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

    def test_peaks_and_valleys_use_pine_formula_lengths(self) -> None:
        analysis = self.run_phase_one(STRUCTURAL_PRICES)
        external_swings = [s for s in analysis["swings"] if s["timeframe"] == "external_4h"]
        internal_swings = [s for s in analysis["swings"] if s["timeframe"] == "internal_1h"]
        market_swings = external_swings + internal_swings

        self.assertGreater(len(market_swings), 0)
        self.assertTrue(any(s["kind"] == "high" for s in market_swings))
        self.assertTrue(any(s["kind"] == "low" for s in market_swings))
        self.assertTrue(all(s["length"] == PARAMS["major_length"] for s in external_swings))
        self.assertTrue(all(s["length"] == PARAMS["internal_length"] for s in internal_swings))
        self.assertTrue(all(s.get("source_confirmed_after", s["confirmed_after"]) == s.get("source_index", s["index"]) + s["length"] for s in market_swings))
        self.assertTrue(any(s.get("source_index") != s.get("index") for s in internal_swings))
        self.assertTrue(all("Pine swings(len)" in s["detection_rule"] for s in market_swings))

    def test_phase_one_auto_pairs_official_external_internal_timeframes(self) -> None:
        internal = make_candles(STRUCTURAL_PRICES)
        external = make_candles(STRUCTURAL_PRICES, hours=24 * 7)
        daily = make_candles(STRUCTURAL_PRICES, hours=24)
        params = {**PARAMS, "external_timeframe": "1wk", "internal_timeframe": "1d", "chart_timeframe": "1d"}

        analysis = run_analysis(
            candles=internal,
            params=params,
            external_candles=external,
            internal_candles=internal,
            daily_candles=daily,
        )
        state = analysis["phase_1"]
        frames = {s["timeframe"] for s in analysis["swings"]}

        self.assertIn("external_1wk", frames)
        self.assertIn("internal_4h", frames)
        self.assertNotIn("internal_1d", frames)
        self.assertEqual(state["swing_settings"]["external_timeframe"], "1wk")
        self.assertEqual(state["swing_settings"]["internal_timeframe"], "4h")
        self.assertEqual(state["swing_settings"]["external_used_length"], PARAMS["major_length"])
        self.assertEqual(state["swing_settings"]["internal_used_length"], PARAMS["internal_length"])
        self.assertEqual(state["timeframe_candle_counts"]["external"], len(external))
        self.assertEqual(state["timeframe_candle_counts"]["internal"], len(internal))

    def test_internal_and_external_swing_scope_contract(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)
        external_swings = [s for s in analysis["swings"] if s["timeframe"] == "external_4h"]

        self.assertGreater(len(external_swings), 0)
        self.assertTrue(any(s["structure_scope"] == "External" and s["bos_produced"] for s in external_swings))
        self.assertTrue(any(s["structure_scope"] == "Internal" and not s["bos_produced"] for s in external_swings))
        self.assertFalse(any(s["structure_scope"] == "External" and not s["bos_produced"] for s in external_swings))

    def test_swing_validation_requires_confirmed_structure_continuation(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)
        swings = analysis["swings"]
        valid = [s for s in swings if s.get("valid_swing")]
        invalid = [s for s in swings if not s.get("valid_swing")]

        self.assertGreater(len(valid), 0)
        self.assertGreater(len(invalid), 0)
        self.assertTrue(all(s["bos_produced"] and s["continuation_confirmed"] for s in valid))
        self.assertTrue(all((not s["bos_produced"]) or (not s["continuation_confirmed"]) for s in invalid))

    def test_swing_strength_classes_follow_phase_one_direct_rule(self) -> None:
        analysis = self.run_phase_one(REVERSAL_PRICES)

        for swing in analysis["swings"]:
            score = float(swing.get("strength_score", 0.0))
            strength_entry = analysis["phase_1"]["swing_strength_map"].get(swing["id"])
            self.assertIsInstance(strength_entry, dict)
            self.assertEqual(strength_entry["score"], swing["strength_score"])
            self.assertEqual(strength_entry["class"], swing["strength_class"])
            self.assertIn(swing["strength_class"], {"STRUCTURAL", "WEAK"})
            self.assertIn(score, {0.0, 1.0})
            self.assertIn("no weighted scoring", swing.get("strength_rule", ""))
            if swing["bos_produced"] and swing["continuation_confirmed"]:
                self.assertEqual(swing["strength_class"], "STRUCTURAL")
                self.assertEqual(score, 1.0)
            else:
                self.assertEqual(swing["strength_class"], "WEAK")
                self.assertEqual(score, 0.0)

        self.assertTrue(any(s["strength_class"] == "STRUCTURAL" for s in analysis["swings"]))
        self.assertTrue(any(s["strength_class"] == "WEAK" for s in analysis["swings"]))

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

    def test_shiftless_counter_break_does_not_consume_later_choch_swing(self) -> None:
        prices = [100, 104, 110, 106, 102, 95, 98, 94, 90, 91, 88, 90, 93, 94, 96, 94, 96.5, 97.5, 92, 94, 96.8, 98.5]
        candles = make_candles(prices)
        engine = StrategyEngine(candles, PARAMS, external_candles=candles, internal_candles=candles)
        df = engine._prepare_df(candles)

        def swing(kind: str, index: int, label: str, confirmed_after: int) -> dict:
            row = df.iloc[index]
            price = float(row["high"] if kind == "high" else row["low"])
            return {
                "id": f"internal_1h:{kind}:{index}",
                "index": index,
                "timestamp": row["timestamp"],
                "price": price,
                "kind": kind,
                "label": label,
                "tier": "internal",
                "timeframe": "internal_1h",
                "length": 2,
                "confirmed_after": confirmed_after,
                "confirmation_timestamp": df.iloc[confirmed_after]["timestamp"],
            }

        counter_high = swing("high", 14, "LH", 16)
        swings = [
            swing("high", 2, "HH", 4),
            swing("low", 5, "HL", 7),
            swing("low", 10, "LL", 12),
            counter_high,
            swing("low", 18, "HL", 20),
        ]

        events, stop_hunts, shifts = engine._detect_phase1_structure(df, swings, "internal_1h")
        rejected = [item for item in stop_hunts if item.get("classification") == "invalid_counter_break_without_shift"]
        choch = [event for event in events if event["event"] == "CHoCH" and event.get("swing_id") == counter_high["id"]]

        self.assertTrue(any(item.get("swing_id") == counter_high["id"] for item in rejected))
        self.assertGreater(len(shifts), 0)
        self.assertGreater(len(choch), 0)
        self.assertEqual(choch[-1]["direction"], "bullish")
        self.assertIsNotNone(choch[-1].get("shift"))

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


class StrategyPhaseTwoTests(unittest.TestCase):
    def phase_two_fixture(self, *, reset_before_mss: bool = False) -> tuple[StrategyEngine, list[dict], list[dict], list[dict]]:
        rows = [
            (100, 101, 99, 100),
            (98, 99, 96, 97),
            (96, 97, 95, 96),
            (99, 100, 98, 99),
            (103, 104, 102, 103),
            (101, 102, 100, 101),
            (97, 100, 94, 96),
            (102, 103, 101, 102) if not reset_before_mss else (94, 96, 92, 93),
            (104, 107, 103, 106),
            (99, 100, 97, 98),
        ]
        candles = make_ohlc(rows)
        engine = StrategyEngine(candles, {**PARAMS, "analysis_phase": 2}, external_candles=candles, internal_candles=candles)
        df = engine._prepare_df(candles)
        swing_low = {
            "id": "internal_1h:low:2",
            "index": 2,
            "timestamp": df.iloc[2]["timestamp"],
            "price": float(df.iloc[2]["low"]),
            "kind": "low",
            "label": "HL",
            "tier": "internal",
            "timeframe": "internal_1h",
            "length": 2,
            "confirmed_after": 3,
            "confirmation_timestamp": df.iloc[3]["timestamp"],
        }
        event = {
            "index": 8,
            "source_index": 8,
            "timestamp": df.iloc[8]["timestamp"],
            "event": "BOS",
            "direction": "bullish",
            "broken_level": 104.0,
            "swing_index": 4,
            "swing_id": "internal_1h:high:4",
            "swing_timestamp": df.iloc[4]["timestamp"],
            "swing_label": "HH",
            "responsible_structure": {"id": swing_low["id"], "index": 2, "price": swing_low["price"], "kind": "low"},
            "protected_level": swing_low["price"],
            "timeframe": "internal_1h",
            "body_close_required": True,
            "continuation_required": True,
            "continuation_confirmed": True,
            "confirmation_index": 8,
            "confirmation_timestamp": df.iloc[8]["timestamp"],
            "continuation_score": 0.85,
        }
        engine.phase_1_state = {
            "swing_strength_map": {
                swing_low["id"]: {
                    "score": 0.85,
                    "class": "STRUCTURAL",
                    "liquidity_taken": True,
                    "bos_produced": True,
                    "continuation_quality": 0.85,
                }
            }
        }
        return engine, [swing_low], [event], candles

    def test_phase_two_confirms_sweep_before_building_range(self) -> None:
        engine, swings, events, _ = self.phase_two_fixture()
        sweeps = engine._detect_phase2_sweeps(engine.df, swings, events, "internal_1h", "IRL")
        ranges = engine._build_phase2_ranges(engine.df, sweeps, events, "internal_1h", "internal")

        confirmed = [sweep for sweep in sweeps if sweep.get("confirmed_sweep")]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["sweep_phase"], 5)
        self.assertEqual(confirmed[0]["sweep_state"], "CONFIRMED")
        self.assertTrue(confirmed[0]["range_allowed"])
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0]["status"], "ACTIVE")
        self.assertTrue(ranges[0]["valid"])
        self.assertIsNone(ranges[0]["b"])
        self.assertIsNone(ranges[0]["c"])
        self.assertAlmostEqual(ranges[0]["eq"], ranges[0]["low"] + ((ranges[0]["high"] - ranges[0]["low"]) * 0.5))
        self.assertEqual(ranges[0]["current_zone"], "DISCOUNT")
        self.assertTrue(ranges[0]["long_allowed"])
        self.assertFalse(ranges[0]["short_allowed"])
        self.assertFalse(ranges[0]["counter_trading_blocked"])

    def test_phase_two_resets_if_original_direction_resumes_before_mss(self) -> None:
        engine, swings, events, _ = self.phase_two_fixture(reset_before_mss=True)
        sweeps = engine._detect_phase2_sweeps(engine.df, swings, events, "internal_1h", "IRL")
        ranges = engine._build_phase2_ranges(engine.df, sweeps, events, "internal_1h", "internal")

        self.assertGreater(len(sweeps), 0)
        self.assertTrue(all(not sweep.get("confirmed_sweep") for sweep in sweeps))
        self.assertTrue(any(sweep.get("sweep_state") == "RESET" for sweep in sweeps))
        self.assertEqual(ranges, [])

    def test_phase_two_response_keeps_phase_three_outputs_disabled(self) -> None:
        external = make_candles(STRUCTURAL_PRICES, hours=4)
        internal = make_candles(STRUCTURAL_PRICES, hours=1)
        daily = make_candles(STRUCTURAL_PRICES, hours=24)
        analysis = run_analysis(
            candles=external,
            params={**PARAMS, "analysis_phase": 2},
            external_candles=external,
            internal_candles=internal,
            daily_candles=daily,
        )

        self.assertIn("phase_2", analysis)
        self.assertEqual(analysis["idms"], [])
        self.assertEqual(analysis["pois"], [])
        self.assertEqual(analysis["setups"], [])
        self.assertEqual(analysis["liquidity_targets"], [])
        self.assertIn("Sweep + Range", analysis["phase_2"]["phase_name"])

    def test_phase_two_state_reports_latest_range_cycle_even_when_invalid(self) -> None:
        candles = make_candles([100, 101, 102, 101, 99])
        engine = StrategyEngine(candles, {**PARAMS, "analysis_phase": 2}, external_candles=candles, internal_candles=candles)
        engine.external_sweeps = []
        engine.internal_sweeps = []
        engine.external_ranges = [
            {
                "timestamp": "2024-01-01T01:00:00Z",
                "timeframe": "external_4h",
                "from_event_index": 1,
                "status": "ACTIVE",
                "valid": True,
                "range_bias": "BULLISH",
                "premium_zone": {"low": 101, "high": 102},
                "discount_zone": {"low": 100, "high": 101},
                "current_zone": "DISCOUNT",
            }
        ]
        engine.ranges = [
            {
                "timestamp": "2024-01-01T03:00:00Z",
                "timeframe": "internal_1h",
                "from_event_index": 3,
                "status": "INVALID",
                "valid": False,
                "range_bias": "BEARISH",
                "premium_zone": {"low": 101, "high": 102},
                "discount_zone": {"low": 100, "high": 101},
                "current_zone": "EQ",
            }
        ]

        state = engine._build_phase2_state()

        self.assertEqual(state["range_status"], "INVALID")
        self.assertFalse(state["range_valid"])
        self.assertEqual(state["gate_status"], "blocked")


if __name__ == "__main__":
    unittest.main()
