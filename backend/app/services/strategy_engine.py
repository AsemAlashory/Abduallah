from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Optional

import pandas as pd


Direction = Literal["bullish", "bearish"]
Side = Literal["buy", "sell"]

PHASE1_AUTO_PAIRS = {
    "1mo": "1d",
    "1wk": "4h",
    "1d": "2h",
    "4h": "1h",
    "1h": "15m",
}
PHASE1_DEFAULT_SWING_LENGTH = 45


class StrategyEngine:
    """SMC sweep-range engine based on the client PDF rules.

    Phase 1 keeps chart, external, and internal feeds separate. The internal
    feed is selected by the official auto-pair table and must not silently
    reuse the chart/external feed.
    """

    def __init__(
        self,
        candles: list[dict],
        params: dict,
        external_candles: Optional[list[dict]] = None,
        internal_candles: Optional[list[dict]] = None,
        micro_candles: Optional[list[dict]] = None,
        weekly_candles: Optional[list[dict]] = None,
        daily_candles: Optional[list[dict]] = None,
    ):
        self.params = params
        self.lightweight_backtest = bool(params.get("_lightweight_backtest"))
        self.chart_df = self._prepare_df(candles)
        self.external_timeframe_label = self._clean_timeframe_label(params.get("external_timeframe") or params.get("chart_timeframe") or "4h")
        self.internal_timeframe_label = self._phase1_internal_pair(self.external_timeframe_label)
        self.chart_timeframe_label = self._clean_timeframe_label(params.get("chart_timeframe") or self.external_timeframe_label)
        self.internal_df = self._prepare_df(internal_candles or [])
        self.external_df = self._prepare_df(external_candles or [])
        self.micro_df = self._prepare_df(micro_candles or [])
        self.weekly_df = self._prepare_df(weekly_candles or [])
        self.daily_df = self._prepare_df(daily_candles or [])
        self.df = self.chart_df
        self.external_timeframe_key = self._phase1_timeframe_key("external", self.external_timeframe_label)
        self.internal_timeframe_key = self._phase1_timeframe_key("internal", self.internal_timeframe_label)
        self.phase1_swing_settings: dict[str, Any] = {}

        self.trend_swings: list[dict] = []
        self.trend_events: list[dict] = []
        self.external_swings: list[dict] = []
        self.internal_swings: list[dict] = []
        self.stop_hunts: list[dict] = []
        self.external_events: list[dict] = []
        self.internal_events: list[dict] = []
        self.phase1_shift_events: list[dict] = []
        self.external_sweeps: list[dict] = []
        self.internal_sweeps: list[dict] = []
        self.external_idms: list[dict] = []
        self.idms: list[dict] = []
        self.external_ranges: list[dict] = []
        self.ranges: list[dict] = []
        self.pois: list[dict] = []
        self.setups: list[dict] = []
        self.movement_legs: list[dict] = []
        self.liquidity_targets: list[dict] = []
        self.trendline_liquidity: list[dict] = []
        self.session_liquidity: list[dict] = []
        self.correction_protocols: list[dict] = []
        self.strategy_state: dict[str, Any] = {}
        self.phase_1_state: dict[str, Any] = {}

    def run(self) -> dict:
        if self.df.empty:
            return self._empty_result()

        trend_requested_length = int(self.params.get("n_candles", self.params.get("swing_length", 2)) or 2)
        external_requested_length = int(self.params.get("major_length", PHASE1_DEFAULT_SWING_LENGTH) or PHASE1_DEFAULT_SWING_LENGTH)
        internal_requested_length = int(self.params.get("internal_length", PHASE1_DEFAULT_SWING_LENGTH) or PHASE1_DEFAULT_SWING_LENGTH)
        external_length = self._phase1_swing_length(self.external_df, external_requested_length)
        internal_length = self._phase1_swing_length(self.internal_df, internal_requested_length)
        self.phase1_swing_settings = {
            "model": "pine_swings_state_flip",
            "source_formula": "indicator.pine swings(len): high[len] > ta.highest(len) / low[len] < ta.lowest(len)",
            "auto_pairing": True,
            "chart_timeframe": self.chart_timeframe_label,
            "external_timeframe": self.external_timeframe_label,
            "internal_timeframe": self.internal_timeframe_label,
            "external_requested_length": max(1, external_requested_length),
            "internal_requested_length": max(1, internal_requested_length),
            "external_used_length": external_length,
            "internal_used_length": internal_length,
            "external_candles": len(self.external_df),
            "internal_candles": len(self.internal_df),
            "feeds_separated": True,
            "same_feed_guard": "external/internal feeds are never auto-reused in Phase 1",
            "trend_requested_length": max(1, trend_requested_length),
        }
        self.phase1_shift_events = []
        trend_components: list[dict] = []
        for trend_df, trend_timeframe, trend_source in self._trend_engine_sources():
            trend_length = self._phase1_swing_length(trend_df, trend_requested_length)
            component_swings = self._detect_phase1_swings(
                trend_df,
                tier="trend",
                timeframe=trend_timeframe,
                length=trend_length,
            )
            component_events, _, _ = self._detect_phase1_structure(trend_df, component_swings, trend_timeframe)
            component_bias = self._derive_phase1_bias(trend_df, component_swings, component_events)
            self._score_phase1_swings(component_swings, trend_df, component_events, trend_timeframe)
            self.trend_swings.extend(component_swings)
            self.trend_events.extend(component_events)
            trend_components.append(
                {
                    "timeframe": trend_timeframe,
                    "source": trend_source,
                    "bias": component_bias or "neutral",
                    "swings": len(component_swings),
                    "events": len(component_events),
                }
            )

        trend_direction = self._combine_trend_biases(trend_components)
        trend_timeframe = "+".join(component["timeframe"] for component in trend_components) if trend_components else "daily"
        trend_source = "+".join(component["source"] for component in trend_components) if trend_components else "fallback_market_data_no_daily_weekly"

        self.external_swings = self._detect_phase1_swings(
            self.external_df,
            tier="major",
            timeframe=self.external_timeframe_key,
            length=external_length,
        )
        self.internal_swings = self._detect_phase1_swings(
            self.internal_df,
            tier="internal",
            timeframe=self.internal_timeframe_key,
            length=internal_length,
        )

        self.external_events, external_stop_hunts, external_shifts = self._detect_phase1_structure(self.external_df, self.external_swings, self.external_timeframe_key)
        self.internal_events, internal_stop_hunts, internal_shifts = self._detect_phase1_structure(self.internal_df, self.internal_swings, self.internal_timeframe_key)
        self.phase1_shift_events.extend(external_shifts + internal_shifts)
        self.stop_hunts = external_stop_hunts + internal_stop_hunts

        structure_bias = self._derive_phase1_bias(self.external_df, self.external_swings, self.external_events)
        if structure_bias is None:
            structure_bias = self._derive_phase1_bias(self.internal_df, self.internal_swings, self.internal_events)

        self._score_phase1_swings(self.external_swings, self.external_df, self.external_events, self.external_timeframe_key)
        self._score_phase1_swings(self.internal_swings, self.internal_df, self.internal_events, self.internal_timeframe_key)
        self._classify_phase1_swing_scopes()

        self.phase_1_state = self._build_phase1_state(
            trend_direction=trend_direction,
            structure_bias=structure_bias,
            trend_timeframe=trend_timeframe,
            trend_source=trend_source,
            trend_components=trend_components,
        )
        self.strategy_state = self.phase_1_state

        # Phase 1 owns structure and trend only. Sweep, Range, IDM, POI, ABC,
        # and Entry are intentionally left empty for later phases.
        self.external_sweeps = []
        self.internal_sweeps = []
        self.external_idms = []
        self.idms = []
        self.external_ranges = []
        self.ranges = []
        self.trendline_liquidity = []
        self.session_liquidity = []
        self.liquidity_targets = []
        self.pois = []
        self.setups = []
        self.movement_legs = []
        self.correction_protocols = []

        swings = self._map_for_chart(self.external_swings, self.external_timeframe_key) + self._map_for_chart(self.internal_swings, self.internal_timeframe_key)
        events = self._map_for_chart(self.external_events, self.external_timeframe_key) + self._map_for_chart(self.internal_events, self.internal_timeframe_key)

        return {
            "summary": {
                "candles": len(self.df),
                "swings": len(swings),
                "structure_events": len(events),
                "stop_hunts": len(self.stop_hunts),
                "active_weak_highs": len(self.phase_1_state.get("activeWeakHighs", [])),
                "active_weak_lows": len(self.phase_1_state.get("activeWeakLows", [])),
                "structural_swings": len([s for s in swings if s.get("strength_class") == "STRUCTURAL"]),
                "poi_allowed": 1 if self.phase_1_state.get("poi_allowed") else 0,
                "sweeps": 0,
                "idms": 0,
                "external_ranges": 0,
                "ranges": 0,
                "pois": 0,
                "range_authorized_pois": 0,
                "aligned_range_authorized_pois": 0,
                "active_pois": 0,
                "setups": 0,
                "movement_legs": 0,
                "trendline_liquidity": 0,
                "session_liquidity": 0,
                "correction_protocols": 0,
            },
            "swings": sorted(swings, key=lambda x: (x["index"], x["timeframe"])),
            "structure_events": sorted(events, key=lambda x: (x["index"], x["timeframe"])),
            "sweeps": [],
            "idms": [],
            "external_ranges": [],
            "ranges": [],
            "pois": [],
            "liquidity_targets": [],
            "phase_1": self.phase_1_state,
            "stop_hunts": self.stop_hunts,
            "strategy_state": self.strategy_state,
            "setups": [],
            "movement_legs": [],
            "trendline_liquidity": [],
            "session_liquidity": [],
            "correction_protocols": [],
        }

    def _empty_result(self) -> dict:
        return {
            "summary": {"candles": 0},
            "swings": [],
            "structure_events": [],
            "sweeps": [],
            "idms": [],
            "external_ranges": [],
            "ranges": [],
            "pois": [],
            "liquidity_targets": [],
            "phase_1": {
                "phase": 1,
                "phase_name": "Structure Foundation",
                "trend_direction": "NEUTRAL",
                "trend_components": [],
                "structure_bias": "NEUTRAL",
                "last_bos": None,
                "current_external_swing": {"high": None, "low": None},
                "protected_high": None,
                "protected_low": None,
                "activeWeakHighs": [],
                "activeWeakLows": [],
                "strongHighs": [],
                "strongLows": [],
                "currentLegType": "WEAK",
                "poi_allowed": False,
                "shift_detected": False,
                "latest_shift": None,
                "latest_choch": None,
                "shift_events": [],
                "choch_requires_shift": True,
                "swing_strength_map": {},
                "gate_status": "blocked",
                "gate_reason": "no_market_data",
                "official_outputs": {
                    "trend_direction": "NEUTRAL",
                    "structure_bias": "NEUTRAL",
                    "last_bos": None,
                    "current_external_swing": {"high": None, "low": None},
                    "protected_high": None,
                    "protected_low": None,
                    "activeWeakHighs": [],
                    "activeWeakLows": [],
                    "strongHighs": [],
                    "strongLows": [],
                    "currentLegType": "WEAK",
                    "poi_allowed": False,
                    "shift_detected": False,
                    "latest_shift": None,
                    "swing_strength_map": {},
                },
            },
            "stop_hunts": [],
            "strategy_state": {
                "phase": 1,
                "phase_name": "Structure Foundation",
                "trend_direction": "NEUTRAL",
                "trend_components": [],
                "structure_bias": "NEUTRAL",
                "last_bos": None,
                "current_external_swing": {"high": None, "low": None},
                "protected_high": None,
                "protected_low": None,
                "activeWeakHighs": [],
                "activeWeakLows": [],
                "strongHighs": [],
                "strongLows": [],
                "currentLegType": "WEAK",
                "poi_allowed": False,
                "shift_detected": False,
                "latest_shift": None,
                "latest_choch": None,
                "shift_events": [],
                "choch_requires_shift": True,
                "swing_strength_map": {},
                "gate_status": "blocked",
                "gate_reason": "no_market_data",
            },
            "setups": [],
            "movement_legs": [],
            "trendline_liquidity": [],
            "session_liquidity": [],
            "correction_protocols": [],
        }

    def _clean_timeframe_label(self, value: Any) -> str:
        label = str(value or "").strip().lower().replace(" ", "")
        aliases = {
            "": "1h",
            "60m": "1h",
            "h1": "1h",
            "h2": "2h",
            "h4": "4h",
            "m15": "15m",
            "w": "1wk",
            "1w": "1wk",
            "week": "1wk",
            "weekly": "1wk",
            "month": "1mo",
            "monthly": "1mo",
            "1mth": "1mo",
            "mo": "1mo",
            "d": "1d",
            "daily": "1d",
        }
        return aliases.get(label, label)

    def _phase1_internal_pair(self, external_label: str) -> str:
        return PHASE1_AUTO_PAIRS.get(self._clean_timeframe_label(external_label), "1h")

    def _phase1_timeframe_key(self, scope: str, label: str) -> str:
        safe = "".join(ch for ch in self._clean_timeframe_label(label) if ch.isalnum()) or scope
        return f"{scope}_{safe}"

    def _phase1_swing_length(self, df: pd.DataFrame, requested: int) -> int:
        requested = max(1, int(requested or 1))
        if df.empty or len(df) > requested:
            return requested
        return max(1, len(df) - 1)

    def _prepare_df(self, candles: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(candles).reset_index(drop=True)
        if df.empty:
            return df

        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "timestamp" not in df.columns:
            return pd.DataFrame()

        df["timestamp"] = df["timestamp"].astype(str)
        df["_time"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = (
            df.dropna(subset=["_time", "open", "high", "low", "close"])
            .sort_values("_time")
            .drop_duplicates(subset=["_time"], keep="last")
            .reset_index(drop=True)
        )
        df["timestamp"] = df["_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return df

    def _trend_engine_sources(self) -> list[tuple[pd.DataFrame, str, str]]:
        sources: list[tuple[pd.DataFrame, str, str]] = []
        market_source = self.external_df if not self.external_df.empty else self.internal_df

        weekly = self.weekly_df if not self.weekly_df.empty else self._resample_ohlc(market_source, "1W")
        daily = self.daily_df if not self.daily_df.empty else self._resample_ohlc(market_source, "1D")

        if not weekly.empty:
            sources.append((weekly, "weekly", "OHLC_Weekly" if not self.weekly_df.empty else "derived_weekly_from_market_data"))
        if not daily.empty:
            sources.append((daily, "daily", "OHLC_Daily" if not self.daily_df.empty else "derived_daily_from_market_data"))
        if not sources and not market_source.empty:
            sources.append((market_source, "daily", "fallback_market_data_no_daily_weekly"))
        return sources

    def _combine_trend_biases(self, components: list[dict]) -> Optional[Direction]:
        weekly = next((c.get("bias") for c in components if c.get("timeframe") == "weekly"), None)
        daily = next((c.get("bias") for c in components if c.get("timeframe") == "daily"), None)
        weekly_bias = weekly if weekly in ("bullish", "bearish") else None
        daily_bias = daily if daily in ("bullish", "bearish") else None

        if weekly_bias and daily_bias:
            return weekly_bias if weekly_bias == daily_bias else weekly_bias
        return weekly_bias or daily_bias

    def _resample_ohlc(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        if df.empty or "_time" not in df:
            return pd.DataFrame()

        indexed = df.sort_values("_time").set_index("_time")
        agg: dict[str, str] = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in indexed.columns:
            agg["volume"] = "sum"

        out = indexed.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"]).reset_index()
        if out.empty:
            return pd.DataFrame()
        out["timestamp"] = out["_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        cols = ["timestamp", "open", "high", "low", "close"]
        if "volume" in out.columns:
            cols.append("volume")
        return self._prepare_df(out[cols].to_dict("records"))

    def _detect_phase1_swings(self, df: pd.DataFrame, tier: str, timeframe: str, length: int) -> list[dict]:
        """Detect swings with the same state-flip logic used by the client Pine formula.

        Pine reference:
            upper = ta.highest(len)
            lower = ta.lowest(len)
            os := high[len] > upper ? 0 : low[len] < lower ? 1 : os[1]
            top = os == 0 and os[1] != 0 ? high[len] : 0
            btm = os == 1 and os[1] != 1 ? low[len] : 0
        """
        swings: list[dict] = []
        length = max(1, int(length))
        if df.empty or len(df) <= length:
            return swings

        last_high_price: Optional[float] = None
        last_low_price: Optional[float] = None
        highs = df["high"].astype(float).to_numpy()
        lows = df["low"].astype(float).to_numpy()
        timestamps = df["timestamp"].tolist()
        os_prev = 0

        for current_index in range(length, len(df)):
            pivot_index = current_index - length
            upper = float(highs[pivot_index + 1 : current_index + 1].max())
            lower = float(lows[pivot_index + 1 : current_index + 1].min())
            os_current = 0 if float(highs[pivot_index]) > upper else (1 if float(lows[pivot_index]) < lower else os_prev)
            confirmed_after = int(current_index)

            if os_current == 0 and os_prev != 0:
                high = float(highs[pivot_index])
                label = "HH" if last_high_price is None or high > last_high_price else "LH"
                last_high_price = high
                swings.append(
                    {
                        "id": f"{timeframe}:high:{pivot_index}",
                        "index": int(pivot_index),
                        "timestamp": timestamps[pivot_index],
                        "price": high,
                        "kind": "high",
                        "label": label,
                        "tier": tier,
                        "timeframe": timeframe,
                        "length": length,
                        "confirmed_after": confirmed_after,
                        "confirmation_timestamp": timestamps[confirmed_after],
                        "detection_rule": "Pine swings(len) state flip: high[len] > ta.highest(len)",
                    }
                )

            if os_current == 1 and os_prev != 1:
                low = float(lows[pivot_index])
                label = "LL" if last_low_price is not None and low < last_low_price else "HL"
                last_low_price = low
                swings.append(
                    {
                        "id": f"{timeframe}:low:{pivot_index}",
                        "index": int(pivot_index),
                        "timestamp": timestamps[pivot_index],
                        "price": low,
                        "kind": "low",
                        "label": label,
                        "tier": tier,
                        "timeframe": timeframe,
                        "length": length,
                        "confirmed_after": confirmed_after,
                        "confirmation_timestamp": timestamps[confirmed_after],
                        "detection_rule": "Pine swings(len) state flip: low[len] < ta.lowest(len)",
                    }
                )

            os_prev = os_current

        return sorted(swings, key=lambda x: (x["index"], x["kind"]))

    def _detect_phase1_structure(self, df: pd.DataFrame, swings: list[dict], timeframe: str) -> tuple[list[dict], list[dict], list[dict]]:
        events: list[dict] = []
        stop_hunts: list[dict] = []
        shift_events: list[dict] = []
        bias: Optional[Direction] = None
        broken_swing_ids: set[str] = set()
        stop_hunt_keys: set[tuple[str, int]] = set()
        processed_shift_swing_ids: set[str] = set()
        pending_shift: Optional[dict] = None

        if df.empty:
            return events, stop_hunts, shift_events

        def register_shift(swing: dict, protected: dict, direction: Direction, index: int, row: pd.Series) -> dict:
            record = {
                "index": int(swing["index"]),
                "source_index": int(index),
                "timestamp": swing["timestamp"],
                "confirmation_index": int(index),
                "confirmation_timestamp": row["timestamp"],
                "event": "Shift",
                "direction": direction,
                "timeframe": timeframe,
                "swing_id": swing.get("id"),
                "swing_index": int(swing["index"]),
                "swing_timestamp": swing["timestamp"],
                "swing_kind": swing["kind"],
                "swing_label": swing.get("label"),
                "weak_level": float(swing["price"]),
                "protected_structure": self._phase1_structure_point(protected),
                "protected_level": float(protected["price"]),
                "produces": "Weak High" if swing["kind"] == "high" else "Weak Low",
                "reason": "price protected the prior high/low instead of breaking it; CHoCH can only be confirmed after this shift is confirmed",
            }
            shift_events.append(record)
            return record

        def reject_shiftless_counter_break(swing: dict, direction: Direction, index: int, row: pd.Series) -> None:
            key = (str(swing.get("id")), int(index))
            if key in stop_hunt_keys:
                return
            stop_hunt_keys.add(key)
            stop_hunts.append(
                {
                    "index": int(index),
                    "timestamp": row["timestamp"],
                    "direction": direction,
                    "tested_level": float(swing["price"]),
                    "swing_id": swing.get("id"),
                    "swing_index": int(swing["index"]),
                    "timeframe": timeframe,
                    "updates_structure": False,
                    "classification": "invalid_counter_break_without_shift",
                    "reason": "opposite-direction structure break was rejected because Phase 1 requires a confirmed Shift before CHoCH",
                }
            )

        confirmation_map: dict[int, list[dict]] = {}
        for swing in swings:
            confirmation_index = int(swing["confirmed_after"])
            if 0 <= confirmation_index < len(df):
                confirmation_map.setdefault(confirmation_index, []).append(swing)

        all_confirmed_highs: list[dict] = []
        all_confirmed_lows: list[dict] = []
        available_highs: list[dict] = []
        available_lows: list[dict] = []
        closes = df["close"].astype(float).to_numpy()

        def latest_prior_swing(candidates: list[dict], swing: dict) -> Optional[dict]:
            swing_index = int(swing["index"])
            return next((candidate for candidate in reversed(candidates) if int(candidate["index"]) < swing_index), None)

        def discard_broken_tail(candidates: list[dict]) -> None:
            while candidates and str(candidates[-1].get("id")) in broken_swing_ids:
                candidates.pop()

        for i in range(len(df)):
            row = df.iloc[i]
            newly_confirmed = confirmation_map.get(int(i), [])
            for swing in newly_confirmed:
                if swing["kind"] == "high":
                    all_confirmed_highs.append(swing)
                    available_highs.append(swing)
                else:
                    all_confirmed_lows.append(swing)
                    available_lows.append(swing)

            discard_broken_tail(available_highs)
            discard_broken_tail(available_lows)
            last_high = available_highs[-1] if available_highs else None
            last_low = available_lows[-1] if available_lows else None

            if last_high is None or last_low is None:
                continue

            for swing in newly_confirmed:
                swing_id = str(swing.get("id"))
                if swing_id in processed_shift_swing_ids:
                    continue
                if bias == "bullish" and swing["kind"] == "high":
                    protected = latest_prior_swing(all_confirmed_highs, swing)
                    if protected and float(swing["price"]) <= float(protected["price"]):
                        pending_shift = register_shift(swing, protected, "bearish", int(i), row)
                        processed_shift_swing_ids.add(swing_id)
                elif bias == "bearish" and swing["kind"] == "low":
                    protected = latest_prior_swing(all_confirmed_lows, swing)
                    if protected and float(swing["price"]) >= float(protected["price"]):
                        pending_shift = register_shift(swing, protected, "bullish", int(i), row)
                        processed_shift_swing_ids.add(swing_id)

            close = float(closes[i])
            high_broken = bool(close > float(last_high["price"]) and last_high["id"] not in broken_swing_ids)
            low_broken = bool(close < float(last_low["price"]) and last_low["id"] not in broken_swing_ids)

            if high_broken:
                continuation = self._break_continuation(df, int(i), "bullish", float(last_high["price"]))
                if continuation["confirmed"]:
                    shift_context = pending_shift if pending_shift and pending_shift.get("direction") == "bullish" else None
                    if bias not in (None, "bullish") and shift_context is None:
                        reject_shiftless_counter_break(last_high, "bullish", int(i), row)
                        broken_swing_ids.add(last_high["id"])
                        continue
                    event_type = "BOS" if bias in (None, "bullish") else "CHoCH"
                    responsible = self._phase1_structure_point(last_low)
                    events.append(
                        {
                            "index": int(i),
                            "source_index": int(i),
                            "timestamp": row["timestamp"],
                            "event": event_type,
                            "direction": "bullish",
                            "broken_level": float(last_high["price"]),
                            "swing_index": int(last_high["index"]),
                            "swing_id": last_high["id"],
                            "swing_timestamp": last_high["timestamp"],
                            "swing_label": last_high.get("label"),
                            "responsible_structure": responsible,
                            "protected_level": responsible["price"],
                            "timeframe": timeframe,
                            "confirmation": "close_plus_continuation",
                            "body_close_required": True,
                            "continuation_required": True,
                            "continuation_confirmed": True,
                            "confirmation_index": continuation["confirmation_index"],
                            "confirmation_timestamp": continuation["confirmation_timestamp"],
                            "continuation_score": continuation["quality"],
                            "shift_detected": event_type == "CHoCH",
                            "shift_required": event_type == "CHoCH",
                            "shift": shift_context,
                            "shift_index": shift_context.get("index") if shift_context else None,
                            "shift_timestamp": shift_context.get("timestamp") if shift_context else None,
                        }
                    )
                    broken_swing_ids.add(last_high["id"])
                    bias = "bullish"
                    if pending_shift and pending_shift.get("direction") == "bearish":
                        pending_shift = None
                    if event_type == "CHoCH":
                        pending_shift = None
                else:
                    key = (last_high["id"], int(i))
                    if key not in stop_hunt_keys:
                        stop_hunt_keys.add(key)
                        stop_hunts.append(self._phase1_stop_hunt_record(row, int(i), last_high, "bullish", timeframe))

            if low_broken:
                continuation = self._break_continuation(df, int(i), "bearish", float(last_low["price"]))
                if continuation["confirmed"]:
                    shift_context = pending_shift if pending_shift and pending_shift.get("direction") == "bearish" else None
                    if bias not in (None, "bearish") and shift_context is None:
                        reject_shiftless_counter_break(last_low, "bearish", int(i), row)
                        broken_swing_ids.add(last_low["id"])
                        continue
                    event_type = "BOS" if bias in (None, "bearish") else "CHoCH"
                    responsible = self._phase1_structure_point(last_high)
                    events.append(
                        {
                            "index": int(i),
                            "source_index": int(i),
                            "timestamp": row["timestamp"],
                            "event": event_type,
                            "direction": "bearish",
                            "broken_level": float(last_low["price"]),
                            "swing_index": int(last_low["index"]),
                            "swing_id": last_low["id"],
                            "swing_timestamp": last_low["timestamp"],
                            "swing_label": last_low.get("label"),
                            "responsible_structure": responsible,
                            "protected_level": responsible["price"],
                            "timeframe": timeframe,
                            "confirmation": "close_plus_continuation",
                            "body_close_required": True,
                            "continuation_required": True,
                            "continuation_confirmed": True,
                            "confirmation_index": continuation["confirmation_index"],
                            "confirmation_timestamp": continuation["confirmation_timestamp"],
                            "continuation_score": continuation["quality"],
                            "shift_detected": event_type == "CHoCH",
                            "shift_required": event_type == "CHoCH",
                            "shift": shift_context,
                            "shift_index": shift_context.get("index") if shift_context else None,
                            "shift_timestamp": shift_context.get("timestamp") if shift_context else None,
                        }
                    )
                    broken_swing_ids.add(last_low["id"])
                    bias = "bearish"
                    if pending_shift and pending_shift.get("direction") == "bullish":
                        pending_shift = None
                    if event_type == "CHoCH":
                        pending_shift = None
                else:
                    key = (last_low["id"], int(i))
                    if key not in stop_hunt_keys:
                        stop_hunt_keys.add(key)
                        stop_hunts.append(self._phase1_stop_hunt_record(row, int(i), last_low, "bearish", timeframe))

        return events, stop_hunts, shift_events

    def _phase1_structure_point(self, swing: dict) -> dict:
        return {
            "id": swing.get("id"),
            "index": int(swing["index"]),
            "timestamp": swing["timestamp"],
            "price": float(swing["price"]),
            "kind": swing["kind"],
            "label": swing.get("label"),
            "timeframe": swing.get("timeframe"),
            "definition": "responsible strong high/low that produced the confirmed body-close structure break",
        }

    def _phase1_stop_hunt_record(self, row: pd.Series, index: int, swing: dict, direction: Direction, timeframe: str) -> dict:
        return {
            "index": index,
            "timestamp": row["timestamp"],
            "direction": direction,
            "tested_level": float(swing["price"]),
            "swing_id": swing.get("id"),
            "swing_index": int(swing["index"]),
            "timeframe": timeframe,
            "updates_structure": False,
            "reason": "body close broke the swing level without continuation; Phase 1 keeps structure unchanged",
        }

    def _break_continuation(self, df: pd.DataFrame, break_index: int, direction: Direction, broken_level: float) -> dict:
        row = df.iloc[break_index]
        close = float(row["close"])
        body = abs(float(row["close"] - row["open"]))
        lookback = df.iloc[max(0, break_index - 20) : break_index]
        ranges = lookback["high"] - lookback["low"] if not lookback.empty else pd.Series(dtype=float)
        bodies = (lookback["close"] - lookback["open"]).abs() if not lookback.empty else pd.Series(dtype=float)
        avg_range = float(ranges.mean()) if not ranges.empty else max(abs(close) * 0.001, 1e-9)
        avg_body = float(bodies.mean()) if not bodies.empty else avg_range * 0.5
        threshold = max(avg_range * 0.25, abs(close) * 0.0001, 1e-9)
        displacement_ok = body >= max(avg_body * 1.15, avg_range * 0.35)

        best_progress = 0.0
        confirmation_index = break_index
        end = min(len(df) - 1, break_index + 3)
        for j in range(break_index + 1, end + 1):
            next_row = df.iloc[j]
            next_close = float(next_row["close"])
            if direction == "bullish":
                if next_close <= broken_level:
                    break
                progress = max(float(next_row["high"]) - close, next_close - close)
            else:
                if next_close >= broken_level:
                    break
                progress = max(close - float(next_row["low"]), close - next_close)

            if progress > best_progress:
                best_progress = progress
                confirmation_index = j

        progress_ok = best_progress >= threshold
        confirmed = bool(progress_ok or displacement_ok)
        if displacement_ok and confirmation_index == break_index:
            best_progress = max(best_progress, threshold)
        quality = 0.0
        if confirmed:
            displacement_quality = min(body / max(avg_body * 1.5, 1e-9), 1.0) if displacement_ok else 0.0
            progress_quality = min(best_progress / max(threshold * 2.0, 1e-9), 1.0) if progress_ok else 0.0
            quality = max(progress_quality, displacement_quality, 0.55)

        return {
            "confirmed": confirmed,
            "quality": float(round(min(quality, 1.0), 4)),
            "confirmation_index": int(confirmation_index),
            "confirmation_timestamp": df.iloc[confirmation_index]["timestamp"],
            "displacement_ok": displacement_ok,
            "progress_ok": progress_ok,
            "threshold": float(threshold),
        }

    def _derive_phase1_bias(self, df: pd.DataFrame, swings: list[dict], events: list[dict]) -> Optional[Direction]:
        if events:
            return events[-1]["direction"]

        highs = [s for s in swings if s["kind"] == "high"]
        lows = [s for s in swings if s["kind"] == "low"]
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1]["price"] > highs[-2]["price"] and lows[-1]["price"] > lows[-2]["price"]:
                return "bullish"
            if highs[-1]["price"] < highs[-2]["price"] and lows[-1]["price"] < lows[-2]["price"]:
                return "bearish"

        if not df.empty and len(df) >= 2:
            first = float(df.iloc[0]["close"])
            last = float(df.iloc[-1]["close"])
            change = (last - first) / first if first else 0.0
            if abs(change) >= 0.01:
                return "bullish" if change > 0 else "bearish"
        return None

    def _score_phase1_swings(self, swings: list[dict], df: pd.DataFrame, events: list[dict], timeframe: str) -> None:
        if df.empty:
            return

        ranges = df["high"] - df["low"]
        median_range = float(ranges.median()) if not ranges.empty else 0.0
        median_range = max(median_range, abs(float(df.iloc[-1]["close"])) * 0.0001, 1e-9)

        broken_by = {event.get("swing_id"): event for event in events if event.get("swing_id")}
        produced_by: dict[str, list[dict]] = {}
        for event in events:
            responsible = event.get("responsible_structure") or {}
            swing_id = responsible.get("id")
            if swing_id:
                produced_by.setdefault(str(swing_id), []).append(event)

        for swing in swings:
            swing_id = str(swing.get("id"))
            liquidity_taken = self._swing_liquidity_taken(df, swing)
            body_broken = swing_id in broken_by
            produced_events = produced_by.get(swing_id, [])
            bos_produced = bool(produced_events)
            continuation_confirmed = any(bool(event.get("continuation_confirmed")) for event in produced_events)
            continuation_quality = max([float(event.get("continuation_score", 0.0) or 0.0) for event in produced_events] or [0.0])
            is_structural = bool(bos_produced and continuation_confirmed)
            strength_score = 1.0 if is_structural else 0.0
            strength_class = "STRUCTURAL" if is_structural else "WEAK"

            role = f"WEAK_{str(swing['kind']).upper()}"
            if produced_events:
                event_direction = produced_events[-1]["direction"]
                if swing["kind"] == "low" and event_direction == "bullish":
                    role = "STRONG_LOW"
                elif swing["kind"] == "high" and event_direction == "bearish":
                    role = "STRONG_HIGH"
            elif body_broken:
                role = f"TAKEN_{str(swing['kind']).upper()}"

            swing.update(
                {
                    "liquidity_taken": liquidity_taken,
                    "body_broken": body_broken,
                    "bos_produced": bos_produced,
                    "continuation_confirmed": continuation_confirmed,
                    "continuation_quality": float(round(continuation_quality, 4)),
                    "strength_score": strength_score,
                    "score_ready": bos_produced,
                    "strength_class": strength_class,
                    "structure_role": role,
                    "valid_swing": is_structural,
                    "validation_status": "structural" if is_structural else "weak_or_pending",
                    "is_active_weak": role.startswith("WEAK_") and not body_broken and not liquidity_taken,
                    "strength_rule": "body-close BOS/CHoCH with continuation confirmed; no weighted scoring in Phase 1",
                }
            )

    def _swing_liquidity_taken(self, df: pd.DataFrame, swing: dict) -> bool:
        start = int(swing.get("confirmed_after", swing.get("index", 0)))
        if start >= len(df):
            return False
        future = df.iloc[start:]
        if swing["kind"] == "high":
            return bool((future["high"] > float(swing["price"])).any())
        return bool((future["low"] < float(swing["price"])).any())

    def _phase1_move_size_score(self, df: pd.DataFrame, swing: dict, median_range: float) -> float:
        start = int(swing["index"])
        end = min(len(df), start + 30)
        window = df.iloc[start:end]
        if window.empty:
            return 0.0
        if swing["kind"] == "low":
            move = float(window["high"].max()) - float(swing["price"])
        else:
            move = float(swing["price"]) - float(window["low"].min())
        return float(min(max(move / max(median_range * 4.0, 1e-9), 0.0), 1.0))

    def _phase1_timeframe_weight(self, timeframe: str) -> float:
        if timeframe == "weekly":
            return 1.0
        if timeframe == "daily":
            return 0.92
        if str(timeframe).startswith("external_"):
            return 0.82
        return 0.72

    def _classify_phase1_swing_scopes(self) -> None:
        external_window = self._current_external_swing()
        high = external_window.get("high")
        low = external_window.get("low")
        external_high = float(high["price"]) if high else None
        external_low = float(low["price"]) if low else None

        for swing in self.trend_swings:
            swing["structure_scope"] = "Trend"

        for swing in self.external_swings:
            is_external = bool(swing.get("bos_produced"))
            swing["structure_scope"] = "External" if is_external else "Internal"
            swing["structure_scope_reason"] = "confirmed_bos_producer" if is_external else "waiting_for_confirmed_bos"

        for swing in self.internal_swings:
            inside_external = False
            if external_high is not None and external_low is not None:
                price = float(swing["price"])
                inside_external = min(external_low, external_high) <= price <= max(external_low, external_high)
            swing["structure_scope"] = "Internal" if inside_external else "ExternalCandidate"

    def _active_phase1_weak_levels(self, kind: Literal["high", "low"]) -> list[dict]:
        records: list[dict] = []
        for swing in self.external_swings + self.internal_swings:
            if swing.get("kind") != kind or not swing.get("is_active_weak"):
                continue
            records.append(
                {
                    "id": swing.get("id"),
                    "index": swing.get("index"),
                    "timestamp": swing.get("timestamp"),
                    "level": float(swing.get("price")),
                    "kind": swing.get("kind"),
                    "label": swing.get("label"),
                    "timeframe": swing.get("timeframe"),
                    "structure_scope": swing.get("structure_scope"),
                    "strength_score": swing.get("strength_score"),
                    "strength_class": swing.get("strength_class"),
                    "role": swing.get("structure_role"),
                }
            )
        return sorted(records, key=lambda x: (str(x["timeframe"]), int(x["index"])))

    def _phase1_strong_levels(self, kind: Literal["high", "low"]) -> list[dict]:
        target_role = f"STRONG_{kind.upper()}"
        records: list[dict] = []
        for swing in self.external_swings + self.internal_swings:
            if swing.get("kind") != kind or swing.get("structure_role") != target_role:
                continue
            records.append(
                {
                    "id": swing.get("id"),
                    "index": swing.get("index"),
                    "timestamp": swing.get("timestamp"),
                    "level": float(swing.get("price")),
                    "kind": swing.get("kind"),
                    "label": swing.get("label"),
                    "timeframe": swing.get("timeframe"),
                    "structure_scope": swing.get("structure_scope"),
                    "strength_score": swing.get("strength_score"),
                    "strength_class": swing.get("strength_class"),
                    "role": swing.get("structure_role"),
                    "bos_produced": bool(swing.get("bos_produced")),
                }
            )
        return sorted(records, key=lambda x: (str(x["timeframe"]), int(x["index"])))

    def _protected_phase1_levels(self, structure_bias: Optional[Direction], current_external_swing: dict) -> dict:
        high = current_external_swing.get("high")
        low = current_external_swing.get("low")
        return {
            "protected_high": high if structure_bias == "bearish" else None,
            "protected_low": low if structure_bias == "bullish" else None,
            "protected_high_level": float(high["price"]) if structure_bias == "bearish" and high else None,
            "protected_low_level": float(low["price"]) if structure_bias == "bullish" and low else None,
        }

    def _current_external_swing(self) -> dict:
        strong_high = next(
            (
                swing
                for swing in reversed(self.external_swings)
                if swing.get("kind") == "high" and swing.get("bos_produced")
            ),
            None,
        )
        strong_low = next(
            (
                swing
                for swing in reversed(self.external_swings)
                if swing.get("kind") == "low" and swing.get("bos_produced")
            ),
            None,
        )
        return {"high": strong_high, "low": strong_low}

    def _current_leg_type(self) -> tuple[str, Optional[dict], Optional[dict]]:
        events = self.internal_events or self.external_events
        swings = self.internal_swings if self.internal_events else self.external_swings
        if not events:
            return "WEAK", None, None

        latest_event = events[-1]
        responsible_id = ((latest_event.get("responsible_structure") or {}).get("id"))
        responsible = next((s for s in swings if s.get("id") == responsible_id), None)
        if latest_event.get("continuation_confirmed"):
            return "STRUCTURAL", latest_event, responsible
        return "WEAK", latest_event, responsible

    def _phase1_gate_reason(self, current_leg_type: str) -> str:
        if current_leg_type != "STRUCTURAL":
            return "weak_leg_no_permission_for_phase3_poi"
        return "structural_leg_green_light_for_phase3"

    def _phase1_last_bos_output(self, event: Optional[dict]) -> Optional[dict]:
        if not event:
            return None
        return {
            "type": event.get("event"),
            "level": event.get("broken_level"),
            "time": event.get("timestamp"),
            "direction": event.get("direction"),
            "index": event.get("index"),
            "timeframe": event.get("timeframe"),
            "confirmed": True,
        }

    def _phase1_df_for_timeframe(self, timeframe: Optional[str]) -> pd.DataFrame:
        if timeframe == self.external_timeframe_key:
            return self.external_df
        if timeframe == self.internal_timeframe_key:
            return self.internal_df
        return self.chart_df

    def _is_phase1_shift_active(self, latest_shift: Optional[dict], structure_events: list[dict]) -> bool:
        if not latest_shift:
            return False

        shift_confirmed_at = str(latest_shift.get("confirmation_timestamp") or latest_shift.get("timestamp") or "")
        for event in structure_events:
            event_time = str(event.get("confirmation_timestamp") or event.get("timestamp") or "")
            if event_time and shift_confirmed_at and event_time >= shift_confirmed_at:
                return False

        timeframe = latest_shift.get("timeframe")
        df = self._phase1_df_for_timeframe(timeframe if isinstance(timeframe, str) else None)
        if df.empty:
            return False

        valid_bars = int(self.params.get("shift_valid_bars", self.params.get("n_candles", 2)) or 2)
        valid_bars = max(0, valid_bars)
        confirmation_index = int(latest_shift.get("confirmation_index", latest_shift.get("index", len(df) - 1)))
        return 0 <= (len(df) - 1 - confirmation_index) <= valid_bars

    def _build_phase1_state(
        self,
        trend_direction: Optional[Direction],
        structure_bias: Optional[Direction],
        trend_timeframe: str,
        trend_source: str,
        trend_components: Optional[list[dict]] = None,
    ) -> dict:
        current_leg_type, latest_leg_event, responsible_swing = self._current_leg_type()
        poi_allowed = current_leg_type == "STRUCTURAL"
        gate_reason = self._phase1_gate_reason(current_leg_type)
        structure_swings = self.external_swings + self.internal_swings
        structure_events = self.external_events + self.internal_events
        last_bos = next((event for event in reversed(structure_events) if event.get("event") == "BOS"), None)
        shift_events = sorted(self.phase1_shift_events, key=lambda x: (str(x.get("timestamp", "")), int(x.get("index", 0))))
        latest_shift = shift_events[-1] if shift_events else None
        latest_choch = next((event for event in reversed(structure_events) if event.get("event") == "CHoCH"), None)
        last_bos_output = self._phase1_last_bos_output(last_bos)
        current_external_swing = self._current_external_swing()
        active_weak_highs = self._active_phase1_weak_levels("high")
        active_weak_lows = self._active_phase1_weak_levels("low")
        strong_highs = self._phase1_strong_levels("high")
        strong_lows = self._phase1_strong_levels("low")
        protected_levels = self._protected_phase1_levels(structure_bias, current_external_swing)
        shift_detected = self._is_phase1_shift_active(latest_shift, structure_events)
        swing_strength_map = {
            str(s["id"]): {
                "score": s.get("strength_score", 0.0),
                "class": s.get("strength_class", "MINOR"),
                "liquidity_taken": bool(s.get("liquidity_taken")),
                "bos_produced": bool(s.get("bos_produced")),
                "continuation_quality": s.get("continuation_quality", 0.0),
                "structural_impact": s.get("structural_impact_score", 0.0),
            }
            for s in structure_swings
            if s.get("id")
        }

        timeframe_candle_counts = {
            "chart": len(self.chart_df),
            "external": len(self.external_df),
            "internal": len(self.internal_df),
            "weekly": len(self.weekly_df),
            "daily": len(self.daily_df),
        }

        return {
            "phase": 1,
            "phase_name": "Structure Foundation",
            "timeframe_candle_counts": timeframe_candle_counts,
            "swing_settings": self.phase1_swing_settings,
            "trend_direction": (trend_direction or "neutral").upper(),
            "trend_timeframe": trend_timeframe,
            "trend_source": trend_source,
            "trend_components": trend_components or [],
            "structure_bias": (structure_bias or "neutral").upper(),
            "last_bos": last_bos_output,
            "last_bos_event": last_bos,
            "current_external_swing": current_external_swing,
            "protected_high": protected_levels["protected_high"],
            "protected_low": protected_levels["protected_low"],
            "protected_high_level": protected_levels["protected_high_level"],
            "protected_low_level": protected_levels["protected_low_level"],
            "activeWeakHighs": active_weak_highs,
            "activeWeakLows": active_weak_lows,
            "strongHighs": strong_highs,
            "strongLows": strong_lows,
            "currentLegType": current_leg_type,
            "current_leg_type": current_leg_type,
            "poi_allowed": poi_allowed,
            "poi_allowed_meaning": "green/red permission only; Phase 1 never builds or calculates POI zones",
            "gate_status": "open" if poi_allowed else "blocked",
            "gate_reason": gate_reason,
            "shift_detected": shift_detected,
            "latest_shift": latest_shift,
            "latest_choch": latest_choch,
            "shift_events": shift_events,
            "choch_requires_shift": True,
            "mss_detected": False,
            "mss_events": [],
            "mss_note": "MSS depends on Phase 2 Sweep context and is not activated inside Phase 1 alone",
            "updated_at": self.df.iloc[-1]["timestamp"] if not self.df.empty else None,
            "latest_structure_event": latest_leg_event,
            "responsible_swing": responsible_swing,
            "official_outputs": {
                "trend_direction": (trend_direction or "neutral").upper(),
                "structure_bias": (structure_bias or "neutral").upper(),
                "last_bos": last_bos_output,
                "current_external_swing": current_external_swing,
                "protected_high": protected_levels["protected_high"],
                "protected_low": protected_levels["protected_low"],
                "activeWeakHighs": active_weak_highs,
                "activeWeakLows": active_weak_lows,
                "strongHighs": strong_highs,
                "strongLows": strong_lows,
                "currentLegType": current_leg_type,
                "poi_allowed": poi_allowed,
                "shift_detected": shift_detected,
                "latest_shift": latest_shift,
                "swing_strength_map": swing_strength_map,
            },
            "swing_strength_map": swing_strength_map,
            "trend_swings": self.trend_swings,
            "trend_events": self.trend_events,
            "stop_hunts": self.stop_hunts,
            "rules_applied": [
                "Trend Engine combines Weekly and Daily direction when supplied or derived",
                "Swing Detection uses TradingView-style pivot highs/lows with configurable external/internal lengths",
                "BOS/CHoCH require candle-body close plus continuation",
                "A strong displacement candle can confirm BOS immediately without waiting for a delayed redraw",
                "CHoCH is only labeled when a protected high/low created a prior Shift",
                "A break without continuation is classified as Stop Hunt and does not update structure",
                "Weak Leg blocks Phase 3; Structural Leg sets poi_allowed=true",
                "Phase 1 does not build Sweep, Range, IDM, Liquidity, POI, ABC, or Entry outputs",
            ],
        }

    def _detect_swings(self, df: pd.DataFrame, tier: str, timeframe: str, length: int) -> list[dict]:
        swings: list[dict] = []
        if df.empty or len(df) <= length:
            return swings

        os = 0
        last_high_price: Optional[float] = None
        last_low_price: Optional[float] = None

        for i in range(length, len(df)):
            candidate_index = i - length
            candidate_high = float(df.iloc[candidate_index]["high"])
            candidate_low = float(df.iloc[candidate_index]["low"])
            upper = float(df["high"].iloc[i - length + 1 : i + 1].max())
            lower = float(df["low"].iloc[i - length + 1 : i + 1].min())
            prev_os = os

            if candidate_high > upper:
                os = 0
            elif candidate_low < lower:
                os = 1

            if os == 0 and prev_os != 0:
                label = "HH" if last_high_price is None or candidate_high > last_high_price else "LH"
                last_high_price = candidate_high
                swings.append(
                    {
                        "index": int(candidate_index),
                        "timestamp": df.iloc[candidate_index]["timestamp"],
                        "price": candidate_high,
                        "kind": "high",
                        "label": label,
                        "tier": tier,
                        "timeframe": timeframe,
                        "length": length,
                        "confirmed_after": int(i),
                    }
                )

            if os == 1 and prev_os != 1:
                label = "LL" if last_low_price is not None and candidate_low < last_low_price else "HL"
                last_low_price = candidate_low
                swings.append(
                    {
                        "index": int(candidate_index),
                        "timestamp": df.iloc[candidate_index]["timestamp"],
                        "price": candidate_low,
                        "kind": "low",
                        "label": label,
                        "tier": tier,
                        "timeframe": timeframe,
                        "length": length,
                        "confirmed_after": int(i),
                    }
                )

        return sorted(swings, key=lambda x: x["index"])

    def _detect_structure(self, df: pd.DataFrame, swings: list[dict], timeframe: str) -> list[dict]:
        events: list[dict] = []
        last_high: Optional[dict] = None
        last_low: Optional[dict] = None
        bias: Optional[Direction] = None

        for i, row in df.iterrows():
            for swing in [x for x in swings if x["confirmed_after"] == i]:
                if swing["kind"] == "high":
                    last_high = swing
                else:
                    last_low = swing

            if last_high is None or last_low is None:
                continue

            # Client rule: BOS/CHoCH requires a candle-body close. Wick breaks are sweeps, not structure.
            prev_close = float(df.iloc[i - 1]["close"]) if i > 0 else float(row["close"])
            high_broken = bool(prev_close <= last_high["price"] and row["close"] > last_high["price"])
            low_broken = bool(prev_close >= last_low["price"] and row["close"] < last_low["price"])

            if high_broken:
                event_type = "BOS" if bias in (None, "bullish") else "CHoCH"
                responsible = self._responsible_structure_point(df, last_low["index"], i, "low")
                events.append(
                    {
                        "index": int(i),
                        "source_index": int(i),
                        "timestamp": row["timestamp"],
                        "event": event_type,
                        "direction": "bullish",
                        "broken_level": float(last_high["price"]),
                        "swing_index": last_high["index"],
                        "swing_timestamp": last_high["timestamp"],
                        "swing_label": last_high.get("label"),
                        "responsible_structure": responsible,
                        "protected_level": responsible["price"] if responsible else float(last_low["price"]),
                        "timeframe": timeframe,
                        "confirmation": "close",
                        "body_close_required": True,
                    }
                )
                bias = "bullish"
                last_high = None

            if low_broken:
                event_type = "BOS" if bias in (None, "bearish") else "CHoCH"
                responsible = self._responsible_structure_point(df, last_high["index"], i, "high")
                events.append(
                    {
                        "index": int(i),
                        "source_index": int(i),
                        "timestamp": row["timestamp"],
                        "event": event_type,
                        "direction": "bearish",
                        "broken_level": float(last_low["price"]),
                        "swing_index": last_low["index"],
                        "swing_timestamp": last_low["timestamp"],
                        "swing_label": last_low.get("label"),
                        "responsible_structure": responsible,
                        "protected_level": responsible["price"] if responsible else float(last_high["price"]),
                        "timeframe": timeframe,
                        "confirmation": "close",
                        "body_close_required": True,
                    }
                )
                bias = "bearish"
                last_low = None

        return events

    def _responsible_structure_point(self, df: pd.DataFrame, start_index: int, end_index: int, kind: Literal["high", "low"]) -> Optional[dict]:
        if df.empty:
            return None
        start = max(0, min(int(start_index), int(end_index)))
        end = min(len(df) - 1, max(int(start_index), int(end_index)))
        if start > end:
            return None
        window = df.iloc[start : end + 1]
        idx = int(window["low"].idxmin()) if kind == "low" else int(window["high"].idxmax())
        row = df.iloc[idx]
        return {
            "index": idx,
            "timestamp": row["timestamp"],
            "price": float(row["low"] if kind == "low" else row["high"]),
            "kind": kind,
            "definition": "responsible structural low/high before the body-close break",
        }

    def _derive_bias(self, swings: list[dict], events: list[dict]) -> Optional[Direction]:
        if events:
            return events[-1]["direction"]

        highs = [s for s in swings if s["kind"] == "high"]
        lows = [s for s in swings if s["kind"] == "low"]
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1]["price"] > highs[-2]["price"] and lows[-1]["price"] > lows[-2]["price"]:
                return "bullish"
            if highs[-1]["price"] < highs[-2]["price"] and lows[-1]["price"] < lows[-2]["price"]:
                return "bearish"
        return None

    def _detect_sweeps(self, df: pd.DataFrame, swings: list[dict], timeframe: str, liquidity_class: str) -> list[dict]:
        sweeps: list[dict] = []
        for i in range(1, len(df)):
            row = df.iloc[i]
            prior_swings = [s for s in swings if s["confirmed_after"] < i]
            low_levels = [s for s in prior_swings if s["kind"] == "low" and row["low"] < s["price"] and row["close"] > s["price"]]
            high_levels = [s for s in prior_swings if s["kind"] == "high" and row["high"] > s["price"] and row["close"] < s["price"]]
            swept_low = max(low_levels, key=lambda x: x["confirmed_after"]) if low_levels else None
            swept_high = max(high_levels, key=lambda x: x["confirmed_after"]) if high_levels else None

            if swept_low:
                sweeps.append(
                    {
                        "index": int(i),
                        "source_index": int(i),
                        "timestamp": row["timestamp"],
                        "type": "bullish_sweep",
                        "direction": "bullish",
                        "swept_level": float(swept_low["price"]),
                        "sweep_price": float(row["low"]),
                        "liquidity_source_index": swept_low["index"],
                        "liquidity_class": liquidity_class,
                        "timeframe": timeframe,
                        "range_point": "A",
                        "updates_structure": False,
                        "reason": "wick swept a confirmed structural liquidity level and closed back inside; structure remains unchanged",
                    }
                )

            if swept_high:
                sweeps.append(
                    {
                        "index": int(i),
                        "source_index": int(i),
                        "timestamp": row["timestamp"],
                        "type": "bearish_sweep",
                        "direction": "bearish",
                        "swept_level": float(swept_high["price"]),
                        "sweep_price": float(row["high"]),
                        "liquidity_source_index": swept_high["index"],
                        "liquidity_class": liquidity_class,
                        "timeframe": timeframe,
                        "range_point": "A",
                        "updates_structure": False,
                        "reason": "wick swept a confirmed structural liquidity level and closed back inside; structure remains unchanged",
                    }
                )

        return sweeps

    def _detect_idms(self, df: pd.DataFrame, events: list[dict], timeframe: str) -> list[dict]:
        idms: list[dict] = []
        sorted_events = sorted(events, key=lambda x: x["index"])

        for event_idx, ev in enumerate(sorted_events):
            start = ev["index"] + 1
            end = sorted_events[event_idx + 1]["index"] if event_idx + 1 < len(sorted_events) else len(df)
            if end - start < 2:
                continue

            direction = ev["direction"]
            found: Optional[dict] = None
            running_high = float(df.iloc[ev["index"]]["high"])
            running_low = float(df.iloc[ev["index"]]["low"])
            responsible = ev.get("responsible_structure")

            for i in range(start + 1, end):
                prev = df.iloc[i - 1]
                row = df.iloc[i]
                inside_bar = row["high"] <= prev["high"] and row["low"] >= prev["low"]
                if inside_bar:
                    continue

                if direction == "bullish":
                    running_high = max(running_high, float(prev["high"]))
                    if prev["high"] >= running_high and row["low"] < prev["low"]:
                        found = {
                            "index": int(i),
                            "timestamp": row["timestamp"],
                            "direction": "bullish",
                            "level": float(prev["low"]),
                            "anchor_index": int(i - 1),
                            "trigger_event_index": ev["index"],
                            "trigger_event": ev["event"],
                            "responsible_structure": responsible,
                            "timeframe": timeframe,
                            "description": "first valid pullback after structural break; inside bars ignored and later checked between POI and current extreme",
                        }
                        break
                else:
                    running_low = min(running_low, float(prev["low"]))
                    if prev["low"] <= running_low and row["high"] > prev["high"]:
                        found = {
                            "index": int(i),
                            "timestamp": row["timestamp"],
                            "direction": "bearish",
                            "level": float(prev["high"]),
                            "anchor_index": int(i - 1),
                            "trigger_event_index": ev["index"],
                            "trigger_event": ev["event"],
                            "responsible_structure": responsible,
                            "timeframe": timeframe,
                            "description": "first valid pullback after structural break; inside bars ignored and later checked between POI and current extreme",
                        }
                        break

            if found is None:
                continue

            found["swept"] = False
            found["swept_at"] = None
            found["sweep_timestamp"] = None
            found["structural_expiry_index"] = int(end)
            for j in range(found["index"] + 1, end):
                row = df.iloc[j]
                if direction == "bullish" and row["low"] < found["level"] and row["close"] > found["level"]:
                    found["swept"] = True
                    found["swept_at"] = int(j)
                    found["sweep_timestamp"] = row["timestamp"]
                    break
                if direction == "bearish" and row["high"] > found["level"] and row["close"] < found["level"]:
                    found["swept"] = True
                    found["swept_at"] = int(j)
                    found["sweep_timestamp"] = row["timestamp"]
                    break

            idms.append(found)

        return idms

    def _build_ranges_for(
        self,
        df: pd.DataFrame,
        events: list[dict],
        sweeps: list[dict],
        idms: list[dict],
        timeframe: str,
    ) -> list[dict]:
        ranges_by_sweep: dict[tuple[str, int], dict] = {}

        for ev in events:
            sweep = next(
                (
                    s
                    for s in reversed(sweeps)
                    if s["direction"] == ev["direction"]
                    and s["index"] <= ev["index"]
                    and self._is_same_structural_leg_in(events, s["index"], ev["index"], ev["direction"])
                ),
                None,
            )
            if sweep is None:
                continue

            idm = next((x for x in idms if x["trigger_event_index"] == ev["index"] and x["direction"] == ev["direction"]), None)
            validation_index = idm["swept_at"] if idm and idm["swept"] else None
            end_index = validation_index or self._structural_expiry_index_for(events, len(df), ev["index"], ev["direction"])
            window = df.iloc[sweep["index"] : end_index + 1]
            if window.empty:
                continue

            if ev["direction"] == "bullish":
                a_price = float(df.iloc[sweep["index"]]["low"])
                c_idx = int(window["high"].idxmax())
                c_price = float(df.iloc[c_idx]["high"])
            else:
                a_price = float(df.iloc[sweep["index"]]["high"])
                c_idx = int(window["low"].idxmin())
                c_price = float(df.iloc[c_idx]["low"])

            lower = min(a_price, c_price)
            upper = max(a_price, c_price)
            midpoint = (lower + upper) / 2

            candidate = {
                "from_event_index": ev["index"],
                "timestamp": ev["timestamp"],
                "direction": ev["direction"],
                "timeframe": timeframe,
                "status": "validated" if validation_index is not None else "candidate_waiting_for_idm_sweep",
                "range_type": "sweep_to_displacement_abc",
                "lower": float(lower),
                "upper": float(upper),
                "midpoint": float(midpoint),
                "discount_zone": {"low": float(lower), "high": float(midpoint)},
                "premium_zone": {"low": float(midpoint), "high": float(upper)},
                "a": {
                    "index": sweep["index"],
                    "timestamp": sweep["timestamp"],
                    "price": float(a_price),
                    "source": sweep["liquidity_class"],
                    "reason": "sweep that starts the range",
                },
                "b": {
                    "index": idm["index"] if idm else None,
                    "timestamp": idm["timestamp"] if idm else None,
                    "price": idm["level"] if idm else None,
                    "swept": bool(idm and idm["swept"]),
                    "swept_at": idm["swept_at"] if idm else None,
                    "sweep_timestamp": idm["sweep_timestamp"] if idm else None,
                    "reason": "first internal liquidity/IDM used as fuel",
                },
                "c": {
                    "index": c_idx,
                    "timestamp": df.iloc[c_idx]["timestamp"],
                    "price": float(c_price),
                    "reason": "moving external-liquidity target for the range",
                },
                "trigger_event": ev["event"],
                "trigger_event_index": ev["index"],
                "validation_index": validation_index,
            }

            range_key = (ev["direction"], sweep["index"])
            existing = ranges_by_sweep.get(range_key)
            if existing is None:
                ranges_by_sweep[range_key] = candidate
                continue

            existing_validated = existing.get("validation_index") is not None
            candidate_validated = validation_index is not None

            # One sweep should seed one range cycle only. We only upgrade the
            # stored range while it is still a candidate and a later event
            # provides a more mature C or the awaited IDM validation.
            should_replace = False
            if candidate_validated and not existing_validated:
                should_replace = True
            elif not candidate_validated and not existing_validated:
                existing_c = float((existing.get("c") or {}).get("price") or 0.0)
                candidate_c = float((candidate.get("c") or {}).get("price") or 0.0)
                if candidate["direction"] == "bullish":
                    should_replace = candidate_c >= existing_c
                else:
                    should_replace = candidate_c <= existing_c

            if should_replace:
                ranges_by_sweep[range_key] = candidate

        return sorted(ranges_by_sweep.values(), key=lambda x: x["from_event_index"])

    def _build_liquidity_targets(self, external_bias: Optional[Direction]) -> list[dict]:
        targets: list[dict] = []
        if self.df.empty:
            return targets

        current = float(self.df.iloc[-1]["close"])
        external_target = self.params.get("external_erl_target")
        invalidation = self.params.get("external_invalidation_level")

        if external_bias == "bullish":
            erl = external_target if external_target is not None else self._nearest_swing_target(self.external_swings, "high", current)
            protected = invalidation if invalidation is not None else self._last_swing_price(self.external_swings, "low")
            irl = self._nearest_swing_target(self.internal_swings, "high", current)
            if erl is not None:
                targets.append({"type": "ERL", "direction": "bullish", "level": float(erl), "timeframe": self.external_timeframe_key, "label": "external buy-side liquidity target"})
            if irl is not None:
                targets.append({"type": "IRL", "direction": "bullish", "level": float(irl), "timeframe": self.internal_timeframe_key, "label": "internal buy-side liquidity target"})
            if protected is not None:
                targets.append({"type": "INVALIDATION", "direction": "bullish", "level": float(protected), "timeframe": self.external_timeframe_key, "label": "no buys below this protected low"})
        elif external_bias == "bearish":
            erl = external_target if external_target is not None else self._nearest_swing_target(self.external_swings, "low", current)
            protected = invalidation if invalidation is not None else self._last_swing_price(self.external_swings, "high")
            irl = self._nearest_swing_target(self.internal_swings, "low", current)
            if erl is not None:
                targets.append({"type": "ERL", "direction": "bearish", "level": float(erl), "timeframe": self.external_timeframe_key, "label": "external sell-side liquidity target"})
            if irl is not None:
                targets.append({"type": "IRL", "direction": "bearish", "level": float(irl), "timeframe": self.internal_timeframe_key, "label": "internal sell-side liquidity target"})
            if protected is not None:
                targets.append({"type": "INVALIDATION", "direction": "bearish", "level": float(protected), "timeframe": self.external_timeframe_key, "label": "no sells above this protected high"})

        targets.extend(self._equal_high_low_targets())
        targets.extend(self.trendline_liquidity)
        targets.extend(self.session_liquidity)
        return targets

    def _nearest_swing_target(self, swings: list[dict], kind: Literal["high", "low"], current: float) -> Optional[float]:
        levels = [float(s["price"]) for s in swings if s["kind"] == kind]
        if kind == "high":
            above = [x for x in levels if x > current]
            return min(above) if above else (max(levels) if levels else None)
        below = [x for x in levels if x < current]
        return max(below) if below else (min(levels) if levels else None)

    def _last_swing_price(self, swings: list[dict], kind: Literal["high", "low"]) -> Optional[float]:
        for swing in reversed(swings):
            if swing["kind"] == kind:
                return float(swing["price"])
        return None

    def _active_external_range(self, internal_index: int, direction: Direction) -> Optional[dict]:
        if not self.external_ranges or self.df.empty or internal_index >= len(self.df):
            return None

        current_ts = self.df.iloc[internal_index].get("_time")
        candidates: list[dict] = []
        for rng in self.external_ranges:
            if rng["direction"] != direction:
                continue
            range_ts = pd.to_datetime(rng["timestamp"], utc=True, errors="coerce")
            if pd.isna(current_ts) or pd.isna(range_ts) or range_ts <= current_ts:
                candidates.append(rng)

        validated = [r for r in candidates if str(r.get("status", "")).startswith("validated")]
        if validated:
            return validated[-1]
        return candidates[-1] if candidates else None

    def _price_level_in_allowed_zone(self, price: float, rng: dict, direction: Direction) -> bool:
        if direction == "bullish":
            return bool(rng["discount_zone"]["low"] <= price <= rng["discount_zone"]["high"])
        return bool(rng["premium_zone"]["low"] <= price <= rng["premium_zone"]["high"])

    def _price_in_external_allowed_zone(self, index: int, direction: Direction) -> bool:
        rng = self._active_external_range(index, direction)
        if not rng:
            return True
        close = float(self.df.iloc[index]["close"])
        return self._price_level_in_allowed_zone(close, rng, direction)

    def _equal_high_low_targets(self) -> list[dict]:
        targets: list[dict] = []
        if len(self.internal_swings) < 2:
            return targets

        median_range = float((self.df["high"] - self.df["low"]).median()) if not self.df.empty else 0.0
        tolerance = max(median_range * 0.18, 1e-9)
        for kind, label in (("high", "equal highs / retail buy-side liquidity"), ("low", "equal lows / retail sell-side liquidity")):
            levels = [s for s in self.internal_swings if s["kind"] == kind]
            for a, b in zip(levels, levels[1:]):
                if abs(a["price"] - b["price"]) <= tolerance:
                    targets.append(
                        {
                            "type": "RETAIL_LQ",
                            "direction": "bullish" if kind == "high" else "bearish",
                            "level": float((a["price"] + b["price"]) / 2),
                            "timeframe": self.internal_timeframe_key,
                            "label": label,
                        }
                    )
                    break
        return targets

    def _detect_trendline_liquidity(self) -> list[dict]:
        targets: list[dict] = []
        if self.df.empty or len(self.internal_swings) < 4:
            return targets

        current_index = len(self.df) - 1
        current_close = float(self.df.iloc[-1]["close"])
        median_range = float((self.df["high"] - self.df["low"]).median()) if not self.df.empty else 0.0
        tolerance = max(median_range * 0.75, abs(current_close) * 0.001)

        for kind, direction, label in (
            ("high", "bullish", "trendline buy-side liquidity"),
            ("low", "bearish", "trendline sell-side liquidity"),
        ):
            levels = [s for s in self.internal_swings if s["kind"] == kind]
            if len(levels) < 2:
                continue
            recent = levels[-8:]
            candidates: list[dict] = []
            for a_pos in range(len(recent) - 1):
                for b_pos in range(a_pos + 1, len(recent)):
                    a = recent[a_pos]
                    b = recent[b_pos]
                    dx = int(b["index"]) - int(a["index"])
                    if dx <= 0:
                        continue
                    slope = (float(b["price"]) - float(a["price"])) / dx
                    touches = []
                    for point in recent:
                        projected = float(a["price"]) + slope * (int(point["index"]) - int(a["index"]))
                        if abs(float(point["price"]) - projected) <= tolerance:
                            touches.append(point)
                    if len(touches) >= 3 or (a_pos == len(recent) - 2 and b_pos == len(recent) - 1):
                        projected_now = float(a["price"]) + slope * (current_index - int(a["index"]))
                        distance = abs(current_close - projected_now)
                        candidates.append(
                            {
                                "type": "TRENDLINE_LQ",
                                "direction": direction,
                                "level": float(projected_now),
                                "timeframe": self.internal_timeframe_key,
                                "label": label,
                                "source": "trendline",
                                "kind": kind,
                                "touches": len(touches),
                                "distance": float(distance),
                                "anchors": [
                                    {"index": int(a["index"]), "timestamp": a["timestamp"], "price": float(a["price"])},
                                    {"index": int(b["index"]), "timestamp": b["timestamp"], "price": float(b["price"])},
                                ],
                            }
                        )
            if candidates:
                targets.append(sorted(candidates, key=lambda x: (x["distance"], -x["touches"]))[0])
        return targets

    def _detect_session_liquidity(self) -> list[dict]:
        if self.chart_df.empty or "_time" not in self.chart_df:
            return []

        session_defs = {
            "Asia": (0, 7),
            "London": (7, 12),
        }
        frame = self.df.copy()
        frame["_date"] = frame["_time"].dt.date
        frame["_hour"] = frame["_time"].dt.hour
        records: list[dict] = []
        current_close = float(self.df.iloc[-1]["close"])

        for session_name, (start_hour, end_hour) in session_defs.items():
            session_rows = frame[(frame["_hour"] >= start_hour) & (frame["_hour"] < end_hour)]
            if session_rows.empty:
                continue
            grouped = session_rows.groupby("_date", sort=True)
            for date, group in list(grouped)[-8:]:
                high_idx = int(group["high"].idxmax())
                low_idx = int(group["low"].idxmin())
                high = float(self.df.iloc[high_idx]["high"])
                low = float(self.df.iloc[low_idx]["low"])
                after = self.df.iloc[max(high_idx, low_idx) + 1 :]
                high_swept = bool((after["high"] > high).any()) if not after.empty else False
                low_swept = bool((after["low"] < low).any()) if not after.empty else False
                records.append(
                    {
                        "type": "SESSION_LQ",
                        "direction": "bullish",
                        "level": high,
                        "timeframe": "session",
                        "label": f"{session_name} high liquidity",
                        "source": "session",
                        "session": session_name,
                        "date": str(date),
                        "index": high_idx,
                        "timestamp": self.df.iloc[high_idx]["timestamp"],
                        "swept": high_swept,
                        "distance": float(abs(current_close - high)),
                    }
                )
                records.append(
                    {
                        "type": "SESSION_LQ",
                        "direction": "bearish",
                        "level": low,
                        "timeframe": "session",
                        "label": f"{session_name} low liquidity",
                        "source": "session",
                        "session": session_name,
                        "date": str(date),
                        "index": low_idx,
                        "timestamp": self.df.iloc[low_idx]["timestamp"],
                        "swept": low_swept,
                        "distance": float(abs(current_close - low)),
                    }
                )

        unswept = [r for r in records if not r["swept"]]
        selected = unswept if unswept else records
        return sorted(selected, key=lambda x: x["distance"])[:8]

    def _detect_authorized_pois(self) -> list[dict]:
        pois: list[dict] = []
        for ev in self.internal_events:
            cause_index = self._find_cause_candle(ev)
            if cause_index is None:
                continue

            cause = self.df.iloc[cause_index]
            low = float(min(cause["open"], cause["close"], cause["low"]))
            high = float(max(cause["open"], cause["close"], cause["high"]))
            fvg = self._find_fvg(cause_index, min(len(self.df) - 1, ev["index"] + 2), ev["direction"])
            recent_sweep = next(
                (
                    s
                    for s in reversed(self.internal_sweeps)
                    if s["direction"] == ev["direction"]
                    and self._is_same_structural_leg(s["index"], ev["index"], ev["direction"])
                ),
                None,
            )
            swept_by_cause = self._candle_swept_previous(cause_index, ev["direction"])
            displacement = self._has_displacement(ev["index"])
            rng = self._active_range(ev["index"], ev["direction"])
            external_rng = self._active_external_range(ev["index"], ev["direction"])
            idm = self._latest_idm(ev["direction"], ev["index"], allow_future_sweep=True)
            idm_between = self._idm_between_poi_and_current(idm, low, high, ev)

            midpoint = (low + high) / 2
            zone_ok = False
            if rng:
                if ev["direction"] == "bullish":
                    zone_ok = rng["discount_zone"]["low"] <= midpoint <= rng["discount_zone"]["high"]
                else:
                    zone_ok = rng["premium_zone"]["low"] <= midpoint <= rng["premium_zone"]["high"]
            external_zone_ok = True
            if external_rng:
                external_zone_ok = self._price_level_in_allowed_zone(midpoint, external_rng, ev["direction"])

            reasons: list[str] = []
            if not (recent_sweep or swept_by_cause):
                reasons.append("missing_liquidity_sweep")
            if not fvg:
                reasons.append("missing_fvg")
            if not displacement:
                reasons.append("weak_displacement")
            if not (idm and idm["swept"]):
                reasons.append("waiting_for_idm_sweep")
            elif not idm_between:
                reasons.append("idm_not_between_poi_and_current_extreme")
            if not rng:
                reasons.append("missing_abc_range")
            elif not zone_ok:
                reasons.append("outside_discount" if ev["direction"] == "bullish" else "outside_premium")
            if external_rng and not external_zone_ok:
                reasons.append("outside_external_discount" if ev["direction"] == "bullish" else "outside_external_premium")

            range_authorized = bool(rng and zone_ok)
            aligned_range_authorized = bool(range_authorized and external_zone_ok)
            valid = len(reasons) == 0
            hierarchy = self._poi_hierarchy(rng, low, high, idm)
            activation_index = max(ev["index"], idm["swept_at"]) if valid and idm else None

            pois.append(
                {
                    "index": int(cause_index),
                    "timestamp": self.df.iloc[cause_index]["timestamp"],
                    "activation_index": activation_index,
                    "type": "Authorized OB + FVG" if fvg else "Order Block Candidate",
                    "direction": ev["direction"],
                    "low": float(min(low, high)),
                    "high": float(max(low, high)),
                    "range_authorized": range_authorized,
                    "aligned_range_authorized": aligned_range_authorized,
                    "entry_ready": valid,
                    "valid": valid,
                    "active": valid,
                    "reason": "authorized_by_sweep_fvg_displacement_idm" if valid else ",".join(reasons),
                    "criteria": {
                        "liquidity_sweep": bool(recent_sweep or swept_by_cause),
                        "fvg": bool(fvg),
                        "displacement": displacement,
                        "idm_swept": bool(idm and idm["swept"]),
                        "premium_discount_ok": zone_ok,
                        "external_range_ok": external_zone_ok,
                    },
                    "hierarchy": hierarchy,
                    "context_event": ev["event"],
                    "context_event_index": ev["index"],
                    "range_index": self.ranges.index(rng) if rng in self.ranges else None,
                    "external_range_index": self.external_ranges.index(external_rng) if external_rng in self.external_ranges else None,
                    "linked_idm_index": idm["index"] if idm else None,
                    "fvg": fvg,
                    "cause_candle_index": int(cause_index),
                    "cause_candle_timestamp": self.df.iloc[cause_index]["timestamp"],
                    "origin_model": "bos_origin_candle",
                    "idm_between_poi_and_current": bool(idm_between),
                }
            )

        dedup: dict[tuple[int, str, float, float], dict] = {}
        for poi in pois:
            key = (poi["index"], poi["direction"], round(poi["low"], 8), round(poi["high"], 8))
            dedup[key] = poi
        return sorted(dedup.values(), key=lambda x: x["index"])

    def _find_cause_candle(self, event: dict) -> Optional[int]:
        event_index = int(event["index"])
        direction: Direction = event["direction"]
        responsible = event.get("responsible_structure") or {}
        start = max(0, int(responsible.get("index", event_index - 20)))
        candidates: list[tuple[float, int]] = []
        for i in range(event_index - 1, start - 1, -1):
            row = self.df.iloc[i]
            if direction == "bullish" and row["close"] < row["open"]:
                displacement_after = abs(float(self.df.iloc[event_index]["close"] - row["close"]))
                depth = float(row["low"])
                candidates.append((displacement_after - depth * 0.000001, int(i)))
            if direction == "bearish" and row["close"] > row["open"]:
                displacement_after = abs(float(row["close"] - self.df.iloc[event_index]["close"]))
                height = float(row["high"])
                candidates.append((displacement_after + height * 0.000001, int(i)))
        if candidates:
            return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
        return int(event_index - 1) if event_index > 0 else None

    def _idm_between_poi_and_current(self, idm: Optional[dict], low: float, high: float, event: dict) -> bool:
        if not idm:
            return False
        level = idm.get("level")
        if level is None:
            return False
        level = float(level)
        event_index = int(event["index"])
        start = max(0, int((event.get("responsible_structure") or {}).get("index", event_index)))
        window = self.df.iloc[min(start, event_index) : max(start, event_index) + 1]
        if window.empty:
            return False
        if event["direction"] == "bullish":
            current_extreme = float(window["high"].max())
            zone_low, zone_high = min(float(high), current_extreme), max(float(high), current_extreme)
            return zone_low <= level <= zone_high
        current_extreme = float(window["low"].min())
        zone_low, zone_high = min(current_extreme, float(low)), max(current_extreme, float(low))
        return zone_low <= level <= zone_high

    def _find_fvg(self, start: int, end: int, direction: Direction) -> Optional[dict]:
        lo = max(2, start)
        hi = min(len(self.df) - 1, end)
        for i in range(lo, hi + 1):
            a = self.df.iloc[i - 2]
            c = self.df.iloc[i]
            if direction == "bullish" and c["low"] > a["high"] and c["low"] - a["high"] >= self.params["min_fvg_size"]:
                return {"index": int(i), "low": float(a["high"]), "high": float(c["low"])}
            if direction == "bearish" and c["high"] < a["low"] and a["low"] - c["high"] >= self.params["min_fvg_size"]:
                return {"index": int(i), "low": float(c["high"]), "high": float(a["low"])}
        return None

    def _candle_swept_previous(self, index: int, direction: Direction) -> bool:
        if index <= 0:
            return False
        row = self.df.iloc[index]
        prev = self.df.iloc[index - 1]
        if direction == "bullish":
            return bool(row["low"] < prev["low"] and row["close"] > prev["low"])
        return bool(row["high"] > prev["high"] and row["close"] < prev["high"])

    def _has_displacement(self, index: int) -> bool:
        if index <= 0:
            return False
        body = abs(float(self.df.iloc[index]["close"] - self.df.iloc[index]["open"]))
        ranges = (self.df["high"] - self.df["low"]).iloc[max(0, index - 20) : index]
        bodies = (self.df["close"] - self.df["open"]).abs().iloc[max(0, index - 20) : index]
        avg_range = float(ranges.mean()) if not ranges.empty else 0.0
        avg_body = float(bodies.mean()) if not bodies.empty else 0.0
        return body >= max(avg_body * 1.15, avg_range * 0.35)

    def _poi_hierarchy(self, rng: Optional[dict], low: float, high: float, idm: Optional[dict]) -> str:
        if rng is None:
            return "unclassified"
        span = max(rng["upper"] - rng["lower"], 1e-9)
        if rng["direction"] == "bullish" and low <= rng["lower"] + span * 0.2:
            return "Extreme POI"
        if rng["direction"] == "bearish" and high >= rng["upper"] - span * 0.2:
            return "Extreme POI"
        if idm and idm["swept"]:
            return "Decisional POI"
        return "candidate"

    def _detect_setups(self, external_bias: Optional[Direction]) -> list[dict]:
        setups: list[dict] = []
        valid_pois = [p for p in self.pois if p["valid"]]

        for poi in valid_pois:
            direction: Direction = poi["direction"]
            if external_bias and direction != external_bias:
                continue

            activation = poi["activation_index"] or poi["context_event_index"]
            expiry = self._structural_expiry_index(activation, direction)
            for i in range(activation, expiry + 1):
                if not self._touches_poi(i, poi, direction):
                    continue

                rng = self._active_range(i, direction)
                if not rng or not self._price_in_allowed_zone(i, rng, direction):
                    continue
                if not self._price_in_external_allowed_zone(i, direction):
                    continue

                sequence = self._entry_sequence(i, direction)
                if sequence["sweep"] and sequence["choch"] and sequence["bos"]:
                    setups.append(self._build_setup(i, "Type A", direction, rng, poi, "high", None, sequence))
                    break

                idm = self._latest_idm(direction, i)
                if sequence["sweep"] and sequence["choch"] and idm and idm["swept"]:
                    sequence["idm"] = idm
                    setups.append(self._build_setup(i, "Type B", direction, rng, poi, "medium", None, sequence))
                    break

        for rng in self.ranges:
            direction = rng["direction"]
            if external_bias and direction != external_bias:
                continue
            index = rng["validation_index"] or rng["trigger_event_index"]
            if index is None or not self._price_in_allowed_zone(index, rng, direction):
                continue
            if not self._price_in_external_allowed_zone(index, direction):
                continue
            sequence = self._entry_sequence(index, direction)
            if sequence["sweep"] and sequence["choch"] and rng["status"].startswith("validated"):
                synthetic_poi = {
                    "low": rng["discount_zone"]["low"] if direction == "bullish" else rng["premium_zone"]["low"],
                    "high": rng["discount_zone"]["high"] if direction == "bullish" else rng["premium_zone"]["high"],
                    "type": "ABC Continuation Range",
                    "hierarchy": "range",
                    "index": index,
                }
                setups.append(self._build_setup(index, "Type C", direction, rng, synthetic_poi, "medium", None, sequence))

        setups = [s for s in setups if self._passes_external_invalidation(s)]
        dedup: dict[tuple[int, str, str], dict] = {}
        for setup in setups:
            key = (setup["index"], setup["setup_type"], setup["direction"])
            dedup[key] = setup
        return sorted(dedup.values(), key=lambda x: x["index"])

    def _touches_poi(self, index: int, poi: dict, direction: Direction) -> bool:
        row = self.df.iloc[index]
        midpoint = float((poi["low"] + poi["high"]) / 2)
        tolerance = abs(midpoint) * float(self.params.get("retest_tolerance_pct", 0) or 0)
        zone_low = float(poi["low"]) - tolerance
        zone_high = float(poi["high"]) + tolerance
        if direction == "bullish":
            return bool(row["low"] <= zone_high and row["close"] >= zone_low)
        return bool(row["high"] >= zone_low and row["close"] <= zone_high)

    def _price_in_allowed_zone(self, index: int, rng: dict, direction: Direction) -> bool:
        close = float(self.df.iloc[index]["close"])
        if direction == "bullish":
            return rng["discount_zone"]["low"] <= close <= rng["discount_zone"]["high"]
        return rng["premium_zone"]["low"] <= close <= rng["premium_zone"]["high"]

    def _window_touches_zone(self, start_index: int, end_index: int, zone: Optional[dict]) -> bool:
        if not zone or self.df.empty:
            return False
        if zone.get("low") is None or zone.get("high") is None:
            return False
        zone_low = float(min(zone.get("low"), zone.get("high")))
        zone_high = float(max(zone.get("low"), zone.get("high")))
        start = max(0, min(start_index, end_index))
        end = min(len(self.df) - 1, max(start_index, end_index))
        if start > end:
            return False
        window = self.df.iloc[start : end + 1]
        return bool(((window["low"] <= zone_high) & (window["high"] >= zone_low)).any())

    def _clean_leg_swings(self) -> list[dict]:
        cleaned: list[dict] = []
        for swing in sorted(self.internal_swings, key=lambda x: x["index"]):
            if not cleaned:
                cleaned.append(swing)
                continue

            previous = cleaned[-1]
            if swing["kind"] != previous["kind"]:
                cleaned.append(swing)
                continue

            if swing["kind"] == "high" and float(swing["price"]) >= float(previous["price"]):
                cleaned[-1] = swing
            elif swing["kind"] == "low" and float(swing["price"]) <= float(previous["price"]):
                cleaned[-1] = swing

        return cleaned

    def _build_movement_legs(self, external_bias: Optional[Direction]) -> list[dict]:
        legs: list[dict] = []
        swings = self._clean_leg_swings()
        if len(swings) < 2:
            return legs

        for seq, (start, end) in enumerate(zip(swings, swings[1:]), start=1):
            start_index = int(start["index"])
            end_index = int(end["index"])
            if start_index == end_index:
                continue

            from_index = min(start_index, end_index)
            to_index = max(start_index, end_index)
            start_price = float(start["price"])
            end_price = float(end["price"])
            direction: Direction = "bullish" if end_price > start_price else "bearish"
            side: Side = "buy" if direction == "bullish" else "sell"

            window_events = [
                event
                for event in self.internal_events
                if from_index <= int(event.get("index", -1)) <= to_index
            ]
            window_sweeps = [
                sweep
                for sweep in self.internal_sweeps
                if from_index <= int(sweep.get("index", -1)) <= to_index
            ]
            window_idms = [
                idm
                for idm in self.idms
                if (
                    from_index <= int(idm.get("index", -1)) <= to_index
                    or from_index <= int(idm.get("anchor_index", -1)) <= to_index
                    or (idm.get("swept_at") is not None and from_index <= int(idm.get("swept_at")) <= to_index)
                )
            ]
            window_ranges = [
                rng
                for rng in self.ranges
                if rng.get("direction") == direction
                and (
                    from_index <= int(rng.get("from_event_index", -1)) <= to_index
                    or from_index <= int((rng.get("a") or {}).get("index", -1)) <= to_index
                    or from_index <= int((rng.get("c") or {}).get("index", -1)) <= to_index
                    or (rng.get("validation_index") is not None and from_index <= int(rng.get("validation_index")) <= to_index)
                )
            ]
            window_pois = [
                poi
                for poi in self.pois
                if poi.get("direction") == direction
                and (
                    from_index <= int(poi.get("index", -1)) <= to_index
                    or from_index <= int(poi.get("context_event_index", -1)) <= to_index
                    or (poi.get("activation_index") is not None and from_index <= int(poi.get("activation_index")) <= to_index)
                )
            ]
            window_setups = [
                setup
                for setup in self.setups
                if setup.get("strategy_direction") == direction and from_index <= int(setup.get("index", -1)) <= to_index
            ]

            active_range = self._active_range(to_index, direction)
            active_external_range = self._active_external_range(to_index, direction)
            allowed_zone = active_range.get("discount_zone") if direction == "bullish" and active_range else None
            if direction == "bearish" and active_range:
                allowed_zone = active_range.get("premium_zone")
            external_zone = active_external_range.get("discount_zone") if direction == "bullish" and active_external_range else None
            if direction == "bearish" and active_external_range:
                external_zone = active_external_range.get("premium_zone")

            conditions = {
                "sweep": any(x.get("direction") == direction for x in window_sweeps),
                "choch": any(x.get("direction") == direction and x.get("event") == "CHoCH" for x in window_events),
                "bos": any(x.get("direction") == direction and x.get("event") == "BOS" for x in window_events),
                "idm": any(x.get("direction") == direction for x in window_idms),
                "idm_swept": any(x.get("direction") == direction and x.get("swept") for x in window_idms),
                "range": active_range is not None or bool(window_ranges),
                "range_validated": any(str(x.get("status", "")).startswith("validated") for x in window_ranges)
                or bool(active_range and str(active_range.get("status", "")).startswith("validated")),
                "allowed_zone_touch": self._window_touches_zone(from_index, to_index, allowed_zone),
                "external_zone_touch": self._window_touches_zone(from_index, to_index, external_zone) if active_external_range else True,
                "poi_candidate": bool(window_pois),
                "poi_range_authorized": any(x.get("range_authorized") for x in window_pois),
                "poi_aligned_range": any(x.get("aligned_range_authorized") for x in window_pois),
                "poi": any(x.get("valid") for x in window_pois),
                "setup": bool(window_setups),
                "external_aligned": external_bias is None or external_bias == direction,
            }

            if conditions["setup"]:
                stage = "setup_ready"
                next_action = "manage_trade"
            elif conditions["poi"]:
                stage = "poi_active"
                next_action = "wait_entry_trigger"
            elif conditions["range_validated"]:
                stage = "range_validated"
                next_action = "wait_authorized_poi"
            elif conditions["choch"] or conditions["bos"]:
                stage = "structure_confirmed"
                next_action = "wait_idm_or_poi"
            elif conditions["sweep"]:
                stage = "liquidity_sweep"
                next_action = "wait_body_close_break"
            else:
                stage = "price_leg"
                next_action = "wait_sweep_or_structure"

            if external_bias is None:
                phase = "neutral"
            elif external_bias == direction:
                phase = "with_external_bias"
            else:
                phase = "counter_external_bias"

            legs.append(
                {
                    "sequence": seq,
                    "timeframe": self.internal_timeframe_key,
                    "start_index": start_index,
                    "end_index": end_index,
                    "start_timestamp": start["timestamp"],
                    "end_timestamp": end["timestamp"],
                    "start_price": start_price,
                    "end_price": end_price,
                    "start_label": start.get("label"),
                    "end_label": end.get("label"),
                    "direction": direction,
                    "side": side,
                    "bars": abs(end_index - start_index),
                    "change": float(end_price - start_price),
                    "change_pct": float(((end_price - start_price) / start_price) * 100) if start_price else 0.0,
                    "phase": phase,
                    "stage": stage,
                    "next_action": next_action,
                    "conditions": conditions,
                    "counts": {
                        "sweeps": len(window_sweeps),
                        "structure_events": len(window_events),
                        "idms": len(window_idms),
                        "ranges": len(window_ranges),
                        "pois": len(window_pois),
                        "range_authorized_pois": len([x for x in window_pois if x.get("range_authorized")]),
                        "aligned_range_authorized_pois": len([x for x in window_pois if x.get("aligned_range_authorized")]),
                        "active_pois": len([x for x in window_pois if x.get("valid")]),
                        "setups": len(window_setups),
                    },
                    "active_range_status": active_range.get("status") if active_range else "none",
                    "active_external_range_status": active_external_range.get("status") if active_external_range else "none",
                    "linked_range": active_range,
                    "linked_external_range": active_external_range,
                    "linked_pois": window_pois[-3:],
                    "linked_setups": window_setups[-3:],
                }
            )

        return legs

    def _entry_sequence(self, index: int, direction: Direction) -> dict[str, Optional[dict]]:
        sweep = next(
            (
                s
                for s in reversed(self.internal_sweeps)
                if s["direction"] == direction and s["index"] <= index and self._is_same_structural_leg(s["index"], index, direction)
            ),
            None,
        )
        if not sweep:
            return {"sweep": None, "choch": None, "bos": None}

        choch = next(
            (
                e
                for e in self.internal_events
                if e["direction"] == direction and e["event"] == "CHoCH" and sweep["index"] <= e["index"] <= index
            ),
            None,
        )
        bos_start = choch["index"] if choch else sweep["index"]
        bos = next(
            (
                e
                for e in self.internal_events
                if e["direction"] == direction and e["event"] == "BOS" and bos_start <= e["index"] <= index
            ),
            None,
        )
        return {"sweep": sweep, "choch": choch, "bos": bos}

    def _estimated_minutes(self, df: pd.DataFrame) -> Optional[float]:
        if df.empty or "_time" not in df or len(df) < 3:
            return None
        deltas = df["_time"].sort_values().diff().dropna().dt.total_seconds() / 60
        if deltas.empty:
            return None
        return float(deltas.median())

    def _detect_correction_protocols(self, external_bias: Optional[Direction]) -> list[dict]:
        if external_bias is None or self.micro_df.empty:
            return []
        interval_minutes = self._estimated_minutes(self.micro_df)
        if interval_minutes is None or interval_minutes > 15:
            return []

        micro_length = int(self.params.get("micro_length", 6))
        micro_swings = self._detect_swings(self.micro_df, "micro", "micro_15m_5m", max(2, micro_length))
        micro_events = self._detect_structure(self.micro_df, micro_swings, "micro_15m_5m")
        if not micro_events:
            return []

        correction_direction: Direction = "bearish" if external_bias == "bullish" else "bullish"
        corrections: list[dict] = []
        active_external = self.external_ranges[-1] if self.external_ranges else None
        target_zone = None
        if active_external:
            target_zone = active_external["discount_zone"] if external_bias == "bullish" else active_external["premium_zone"]

        matching = [e for e in micro_events if e["direction"] == correction_direction]
        for choch in [e for e in matching if e["event"] == "CHoCH"][-4:]:
            bos = next(
                (
                    e
                    for e in matching
                    if e["event"] == "BOS" and int(e["index"]) > int(choch["index"])
                ),
                None,
            )
            corrections.append(
                {
                    "timeframe": "15m/5m",
                    "source_interval_minutes": interval_minutes,
                    "external_bias": external_bias,
                    "correction_direction": correction_direction,
                    "status": "confirmed" if bos else "waiting_for_micro_bos",
                    "choch": choch,
                    "bos": bos,
                    "target_zone": target_zone,
                    "rule": "correction protocol requires micro CHoCH plus micro BOS before trading a pullback toward the higher-timeframe discount/premium zone",
                }
            )
        return corrections[-6:]

    def _is_same_structural_leg(self, start_index: int, end_index: int, direction: Direction) -> bool:
        return self._is_same_structural_leg_in(self.internal_events, start_index, end_index, direction)

    def _is_same_structural_leg_in(self, events: list[dict], start_index: int, end_index: int, direction: Direction) -> bool:
        if start_index > end_index:
            return False
        return not any(
            e["direction"] != direction and start_index < e["index"] <= end_index
            for e in events
        )

    def _structural_expiry_index(self, start_index: int, direction: Direction) -> int:
        return self._structural_expiry_index_for(self.internal_events, len(self.df), start_index, direction)

    def _structural_expiry_index_for(self, events: list[dict], df_length: int, start_index: int, direction: Direction) -> int:
        opposite = next(
            (
                e["index"]
                for e in events
                if e["index"] > start_index and e["direction"] != direction
            ),
            None,
        )
        return int(opposite) if opposite is not None else df_length - 1

    def _latest_idm(self, direction: Direction, index: int, allow_future_sweep: bool = False) -> Optional[dict]:
        candidates = [x for x in self.idms if x["direction"] == direction and x["index"] <= index]
        if allow_future_sweep:
            candidates = [x for x in self.idms if x["direction"] == direction and x["trigger_event_index"] <= index]
        return candidates[-1] if candidates else None

    def _active_range(self, index: int, direction: Optional[Direction] = None) -> Optional[dict]:
        ranges = [r for r in self.ranges if r["from_event_index"] <= index and (direction is None or r["direction"] == direction)]
        return ranges[-1] if ranges else None

    def _build_setup(
        self,
        index: int,
        setup_type: str,
        direction: Direction,
        rng: dict,
        poi: dict,
        confidence: Literal["low", "medium", "high"],
        invalidation_reason: Optional[str],
        sequence: dict[str, Optional[dict]],
    ) -> dict:
        side: Side = "buy" if direction == "bullish" else "sell"
        entry = float((poi["low"] + poi["high"]) / 2)
        target_1 = self._target_level("IRL", direction) or rng["b"]["price"] or rng["c"]["price"]
        target_final = self._target_level("ERL", direction) or rng["c"]["price"]
        stop_source = "poi_or_range_a"

        if side == "buy":
            stop = min(float(poi["low"]), float(rng["a"]["price"]))
            if setup_type == "Type B":
                wick_stop = self._type_b_wick_stop(sequence, direction)
                if wick_stop is not None and float(wick_stop) < entry:
                    stop = float(wick_stop)
                    stop_source = "idm_sweep_wick"
            if target_1 <= entry:
                target_1 = rng["c"]["price"]
            if target_final <= entry:
                target_final = entry + abs(entry - stop) * 2
        else:
            stop = max(float(poi["high"]), float(rng["a"]["price"]))
            if setup_type == "Type B":
                wick_stop = self._type_b_wick_stop(sequence, direction)
                if wick_stop is not None and float(wick_stop) > entry:
                    stop = float(wick_stop)
                    stop_source = "idm_sweep_wick"
            if target_1 >= entry:
                target_1 = rng["c"]["price"]
            if target_final >= entry:
                target_final = entry - abs(entry - stop) * 2

        valid = invalidation_reason is None
        external_rng = self._active_external_range(index, direction)
        return {
            "index": int(index),
            "timestamp": self.df.iloc[index]["timestamp"],
            "setup_type": setup_type,
            "direction": side,
            "strategy_direction": direction,
            "entry_zone": {"low": float(poi["low"]), "high": float(poi["high"]), "mid": float(entry)},
            "stop_loss": float(stop),
            "target_1": float(target_1),
            "target": float(target_final),
            "target_final": float(target_final),
            "reason": self._setup_reason(setup_type, poi),
            "valid": valid,
            "invalidation_reason": invalidation_reason,
            "confidence": confidence,
            "model_rules": {
                "sweep": bool(sequence["sweep"]),
                "choch": bool(sequence["choch"]),
                "bos": bool(sequence["bos"]),
                "poi_authorized": poi.get("type") != "ABC Continuation Range",
                "discount_or_premium": True,
                "external_range_filter": self._price_in_external_allowed_zone(index, direction),
            },
            "abc": {"a": rng["a"], "b": rng["b"], "c": rng["c"]},
            "external_abc": (
                {"a": external_rng["a"], "b": external_rng["b"], "c": external_rng["c"]}
                if external_rng
                else None
            ),
            "poi_type": poi.get("type"),
            "poi_hierarchy": poi.get("hierarchy"),
            "stop_source": stop_source,
        }

    def _type_b_wick_stop(self, sequence: dict[str, Optional[dict]], direction: Direction) -> Optional[float]:
        idm = sequence.get("idm") if isinstance(sequence, dict) else None
        sweep_index = idm.get("swept_at") if isinstance(idm, dict) else None
        if sweep_index is None and isinstance(sequence.get("sweep"), dict):
            sweep_index = sequence["sweep"].get("index")
        if sweep_index is None:
            return None
        sweep_index = int(sweep_index)
        if sweep_index < 0 or sweep_index >= len(self.df):
            return None
        row = self.df.iloc[sweep_index]
        return float(row["low"] if direction == "bullish" else row["high"])

    def _setup_reason(self, setup_type: str, poi: dict) -> str:
        if setup_type == "Type A":
            return "Sweep + CHoCH + BOS after an authorized POI in the valid range zone"
        if setup_type == "Type B":
            return "IDM trap: internal liquidity was swept before entry activation"
        return "Continuation model: new validated ABC range aligned with external structure"

    def _target_level(self, target_type: str, direction: Direction) -> Optional[float]:
        for target in self.liquidity_targets:
            if target["type"] == target_type and target["direction"] == direction:
                return float(target["level"])
        return None

    def _passes_external_invalidation(self, setup: dict) -> bool:
        invalidation = next((x for x in self.liquidity_targets if x["type"] == "INVALIDATION"), None)
        if not invalidation:
            return True
        close = float(self.df.iloc[setup["index"]]["close"])
        if setup["direction"] == "buy" and close < invalidation["level"]:
            setup["valid"] = False
            setup["invalidation_reason"] = "price_below_external_protected_low"
            return False
        if setup["direction"] == "sell" and close > invalidation["level"]:
            setup["valid"] = False
            setup["invalidation_reason"] = "price_above_external_protected_high"
            return False
        return True

    def _build_strategy_state(self, external_bias: Optional[Direction]) -> dict:
        active_range = self.ranges[-1] if self.ranges else None
        active_external_range = self.external_ranges[-1] if self.external_ranges else None
        active_pois = [p for p in self.pois if p["valid"]]
        range_authorized_pois = [p for p in self.pois if p.get("range_authorized")]
        aligned_range_authorized_pois = [p for p in self.pois if p.get("aligned_range_authorized")]
        candidate_pois = [p for p in self.pois if not p["valid"]]
        invalidation = next((x for x in self.liquidity_targets if x["type"] == "INVALIDATION"), None)
        erl = next((x for x in self.liquidity_targets if x["type"] == "ERL"), None)
        return {
            "external_bias": external_bias or "neutral",
            "external_timeframe": self.external_timeframe_label,
            "internal_timeframe": self.internal_timeframe_label,
            "active_external_range_status": active_external_range["status"] if active_external_range else "none",
            "active_external_range_direction": active_external_range["direction"] if active_external_range else None,
            "active_external_range_midpoint": active_external_range["midpoint"] if active_external_range else None,
            "active_range_status": active_range["status"] if active_range else "none",
            "active_range_direction": active_range["direction"] if active_range else None,
            "active_range_midpoint": active_range["midpoint"] if active_range else None,
            "authorized_pois": len(active_pois),
            "range_authorized_pois": len(range_authorized_pois),
            "aligned_range_authorized_pois": len(aligned_range_authorized_pois),
            "candidate_pois": len(candidate_pois),
            "total_pois": len(self.pois),
            "entry_model_priority": "A ثم B ثم C",
            "swing_model": "pine_swings_state_flip",
            "external_swing_length": int(self.phase1_swing_settings.get("external_used_length", PHASE1_DEFAULT_SWING_LENGTH)),
            "internal_swing_length": int(self.phase1_swing_settings.get("internal_used_length", PHASE1_DEFAULT_SWING_LENGTH)),
            "micro_timeframe_status": "active" if self.correction_protocols else ("no_micro_feed" if self.micro_df.empty else "waiting_for_micro_choch_bos"),
            "latest_external_swing": self.external_swings[-1] if self.external_swings else None,
            "latest_internal_swing": self.internal_swings[-1] if self.internal_swings else None,
            "latest_structure_event": self.internal_events[-1] if self.internal_events else None,
            "erl_target": erl["level"] if erl else None,
            "external_invalidation": invalidation["level"] if invalidation else None,
            "trendline_liquidity_count": len(self.trendline_liquidity),
            "session_liquidity_count": len(self.session_liquidity),
            "correction_protocol_count": len(self.correction_protocols),
            "rule_notes": [
                "BOS/CHoCH are confirmed by candle-body close only",
                "Wick breaks are liquidity sweeps and do not update structure",
                "Liquidity is read from confirmed structural levels on the selected timeframe, not from a fixed candle count",
                "POIs activate only after IDM sweep + FVG + displacement",
                "Bullish buys require discount; bearish sells require premium",
                "Trendline and Asia/London highs/lows are mapped as liquidity pools",
                "15m/5m correction entries require micro CHoCH plus BOS when micro data is supplied",
            ],
        }

    def _is_chart_timeframe(self, timeframe: str) -> bool:
        return (
            (timeframe == self.external_timeframe_key and self.chart_timeframe_label == self.external_timeframe_label)
            or (timeframe == self.internal_timeframe_key and self.chart_timeframe_label == self.internal_timeframe_label)
        )

    def _map_for_chart(self, records: list[dict], timeframe: str) -> list[dict]:
        if self._is_chart_timeframe(timeframe):
            return [dict(x) for x in records]

        mapped: list[dict] = []
        for record in records:
            item = dict(record)
            item["source_index"] = record["index"]
            item["index"] = self._nearest_chart_index(record["timestamp"])
            if "confirmed_after" in record:
                item["source_confirmed_after"] = record.get("confirmed_after")
                confirmation_timestamp = record.get("confirmation_timestamp")
                if confirmation_timestamp:
                    item["confirmed_after"] = self._nearest_chart_index(confirmation_timestamp)
            if "confirmation_index" in record:
                item["source_confirmation_index"] = record.get("confirmation_index")
                confirmation_timestamp = record.get("confirmation_timestamp")
                if confirmation_timestamp:
                    item["confirmation_index"] = self._nearest_chart_index(confirmation_timestamp)
            if record.get("shift_index") is not None:
                item["source_shift_index"] = record.get("shift_index")
                shift_timestamp = record.get("shift_timestamp")
                if shift_timestamp:
                    item["shift_index"] = self._nearest_chart_index(shift_timestamp)
            if "swing_timestamp" in record:
                item["swing_source_index"] = record.get("swing_index")
                item["swing_index"] = self._nearest_chart_index(record["swing_timestamp"])
            mapped.append(item)
        return mapped

    def _map_ranges_for_chart(self, ranges: list[dict], timeframe: str) -> list[dict]:
        if self._is_chart_timeframe(timeframe):
            return [deepcopy(x) for x in ranges]

        mapped: list[dict] = []
        for record in ranges:
            item = deepcopy(record)
            item["source_from_event_index"] = record.get("from_event_index")
            item["from_event_index"] = self._nearest_chart_index(record.get("timestamp", ""))
            item["source_trigger_event_index"] = record.get("trigger_event_index")
            item["trigger_event_index"] = self._nearest_chart_index(record.get("timestamp", ""))

            for key in ("a", "b", "c"):
                point = item.get(key)
                if not isinstance(point, dict):
                    continue
                point["source_index"] = point.get("index")
                timestamp = point.get("timestamp")
                if timestamp:
                    point["index"] = self._nearest_chart_index(timestamp)
                if point.get("swept_at") is not None:
                    point["source_swept_at"] = point.get("swept_at")
                    sweep_timestamp = point.get("sweep_timestamp")
                    if sweep_timestamp:
                        point["swept_at"] = self._nearest_chart_index(sweep_timestamp)

            if item.get("validation_index") is not None and isinstance(item.get("b"), dict):
                item["source_validation_index"] = item.get("validation_index")
                sweep_timestamp = item["b"].get("sweep_timestamp")
                if sweep_timestamp:
                    item["validation_index"] = self._nearest_chart_index(sweep_timestamp)
            mapped.append(item)

        return sorted(mapped, key=lambda x: x["from_event_index"])

    def _nearest_chart_index(self, timestamp: str) -> int:
        if self.chart_df.empty or "_time" not in self.chart_df:
            return 0
        ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
        if pd.isna(ts):
            return 0
        exact = self.chart_df.index[self.chart_df["_time"] == ts].tolist()
        if exact:
            return int(exact[0])
        prior = self.chart_df.index[self.chart_df["_time"] <= ts].tolist()
        if prior:
            return int(prior[-1])
        return 0


def run_analysis(
    candles: list[dict],
    params: dict,
    external_candles: Optional[list[dict]] = None,
    internal_candles: Optional[list[dict]] = None,
    micro_candles: Optional[list[dict]] = None,
    weekly_candles: Optional[list[dict]] = None,
    daily_candles: Optional[list[dict]] = None,
) -> dict:
    engine = StrategyEngine(
        candles=candles,
        params=params,
        external_candles=external_candles,
        internal_candles=internal_candles,
        micro_candles=micro_candles,
        weekly_candles=weekly_candles,
        daily_candles=daily_candles,
    )
    return engine.run()
