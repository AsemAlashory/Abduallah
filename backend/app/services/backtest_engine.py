from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.strategy_engine import run_analysis


def _setup_model_summary(analysis: Dict[str, Any]) -> tuple[Dict[str, int], List[Dict[str, Any]]]:
    models: List[Dict[str, Any]] = []

    for setup in analysis.get("setups", []):
        models.append(
            {
                "setup_type": setup.get("setup_type"),
                "direction": setup.get("direction"),
                "index": setup.get("index"),
                "timestamp": setup.get("timestamp"),
                "status": "trade_ready" if setup.get("valid") else "invalidated",
                "reason": setup.get("reason"),
                "entry_zone": setup.get("entry_zone"),
                "stop_loss": setup.get("stop_loss"),
                "target_1": setup.get("target_1"),
                "target": setup.get("target"),
                "target_final": setup.get("target_final"),
                "poi_type": setup.get("poi_type"),
                "confidence": setup.get("confidence"),
                "stop_source": setup.get("stop_source"),
            }
        )

    for poi in analysis.get("pois", []):
        if not poi.get("valid"):
            continue

        side = "buy" if poi.get("direction") == "bullish" else "sell"
        models.append(
            {
                "setup_type": "Type A",
                "direction": side,
                "index": poi.get("activation_index") or poi.get("index"),
                "timestamp": poi.get("timestamp"),
                "status": "candidate_waiting_for_entry_trigger",
                "reason": "Authorized POI is active; waiting for final Sweep + CHoCH + BOS entry trigger",
                "entry_zone": {"low": poi.get("low"), "high": poi.get("high")},
                "poi_type": poi.get("type"),
                "confidence": "watchlist",
                "stop_source": "pending_entry_confirmation",
            }
        )

        if poi.get("criteria", {}).get("idm_swept"):
            models.append(
                {
                    "setup_type": "Type B",
                    "direction": side,
                    "index": poi.get("activation_index") or poi.get("index"),
                    "timestamp": poi.get("timestamp"),
                    "status": "candidate_idm_trap",
                    "reason": "IDM has been swept near an authorized POI; waiting for execution trigger",
                    "entry_zone": {"low": poi.get("low"), "high": poi.get("high")},
                    "poi_type": poi.get("type"),
                    "confidence": "watchlist",
                    "stop_source": "idm_sweep_wick",
                }
            )

    for rng in analysis.get("ranges", []):
        if not str(rng.get("status", "")).startswith("validated"):
            continue
        direction = "buy" if rng.get("direction") == "bullish" else "sell"
        zone = rng.get("discount_zone") if direction == "buy" else rng.get("premium_zone")
        models.append(
            {
                "setup_type": "Type C",
                "direction": direction,
                "index": rng.get("validation_index") or rng.get("trigger_event_index"),
                "timestamp": rng.get("timestamp"),
                "status": "candidate_continuation_range",
                "reason": "Validated ABC range; continuation model available if internal and external structure stay aligned",
                "entry_zone": zone,
                "poi_type": "ABC Range",
                "confidence": "watchlist",
                "stop_source": "range_boundary",
            }
        )

    dedup: Dict[tuple, Dict[str, Any]] = {}
    for model in models:
        key = (model.get("setup_type"), model.get("direction"), model.get("index"), model.get("status"))
        dedup[key] = model

    ordered = sorted(dedup.values(), key=lambda x: (x.get("index") is None, x.get("index") or 0, x.get("setup_type") or ""))
    counts = {
        "Type A": len([m for m in ordered if m.get("setup_type") == "Type A"]),
        "Type B": len([m for m in ordered if m.get("setup_type") == "Type B"]),
        "Type C": len([m for m in ordered if m.get("setup_type") == "Type C"]),
        "trade_ready": len([m for m in ordered if m.get("status") == "trade_ready"]),
        "watchlist": len([m for m in ordered if m.get("status") != "trade_ready"]),
    }
    return counts, ordered


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _range_for_model(analysis: Dict[str, Any], model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    index = model.get("index")
    direction = "bullish" if model.get("direction") == "buy" else "bearish"
    if index is None:
        return None

    candidates = [
        r
        for r in analysis.get("ranges", [])
        if r.get("direction") == direction
        and (r.get("from_event_index") or 0) <= int(index)
        and str(r.get("status", "")).startswith("validated")
    ]
    if candidates:
        return candidates[-1]

    candidates = [
        r
        for r in analysis.get("ranges", [])
        if r.get("direction") == direction and (r.get("from_event_index") or 0) <= int(index)
    ]
    return candidates[-1] if candidates else None


def _target_level(analysis: Dict[str, Any], target_type: str, direction: str) -> Optional[float]:
    for target in analysis.get("liquidity_targets", []):
        if target.get("type") == target_type and target.get("direction") == direction:
            return _as_float(target.get("level"))
    return None


def _target_level_at(
    analysis: Dict[str, Any],
    target_type: str,
    direction: str,
    signal_index: int,
    current: float,
) -> Optional[float]:
    timeframe = "external_4h" if target_type == "ERL" else "internal_1h"
    kind = "high" if direction == "bullish" else "low"
    levels = [
        _as_float(swing.get("price"))
        for swing in analysis.get("swings", [])
        if swing.get("timeframe") == timeframe
        and swing.get("kind") == kind
        and swing.get("index") is not None
        and int(swing.get("index")) <= signal_index
    ]
    levels = [level for level in levels if level is not None]
    if not levels:
        return _target_level(analysis, target_type, direction)

    if direction == "bullish":
        above = [level for level in levels if level > current]
        return min(above) if above else max(levels)

    below = [level for level in levels if level < current]
    return max(below) if below else min(levels)


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        number = _as_float(value)
        if number is not None:
            return number
    return None


def _prefix_by_timestamp(candles: Optional[List[Dict[str, Any]]], timestamp: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not candles or not timestamp:
        return candles
    return [c for c in candles if str(c.get("timestamp", "")) <= str(timestamp)]


def _find_matching_model(models: List[Dict[str, Any]], target: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target_index = target.get("index")
    target_timestamp = target.get("timestamp")
    candidates = [
        m
        for m in models
        if m.get("setup_type") == target.get("setup_type")
        and m.get("direction") == target.get("direction")
        and m.get("status") != "invalidated"
    ]
    if target_timestamp:
        by_time = [m for m in candidates if m.get("timestamp") == target_timestamp]
        if by_time:
            return by_time[-1]
    if target_index is None:
        return candidates[-1] if candidates else None
    close = [m for m in candidates if m.get("index") is not None and abs(int(m["index"]) - int(target_index)) <= 2]
    if close:
        return sorted(close, key=lambda m: abs(int(m["index"]) - int(target_index)))[0]
    earlier = [m for m in candidates if m.get("index") is not None and int(m["index"]) <= int(target_index)]
    return earlier[-1] if earlier else None


def _timestamp_to_index(candles: List[Dict[str, Any]], timestamp: Optional[str], fallback: Optional[int]) -> Optional[int]:
    if not timestamp:
        return fallback
    for index, candle in enumerate(candles):
        if str(candle.get("timestamp")) == str(timestamp):
            return index
    return fallback


def _index_is_known(value: Any, signal_index: int) -> bool:
    if value is None:
        return True
    try:
        return int(value) <= signal_index
    except (TypeError, ValueError):
        return False


def _nested_index_is_known(record: Optional[Dict[str, Any]], key: str, signal_index: int) -> bool:
    if not isinstance(record, dict):
        return True
    return _index_is_known(record.get(key), signal_index)


def _range_is_walk_forward_safe(rng: Optional[Dict[str, Any]], signal_index: int) -> bool:
    if not isinstance(rng, dict):
        return True
    return all(
        [
            _index_is_known(rng.get("from_event_index"), signal_index),
            _index_is_known(rng.get("trigger_event_index"), signal_index),
            _index_is_known(rng.get("validation_index"), signal_index),
            _nested_index_is_known(rng.get("a"), "index", signal_index),
            _nested_index_is_known(rng.get("b"), "index", signal_index),
            _nested_index_is_known(rng.get("b"), "swept_at", signal_index),
            _nested_index_is_known(rng.get("c"), "index", signal_index),
        ]
    )


def _find_poi_for_model(analysis: Dict[str, Any], model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    zone = model.get("entry_zone") or {}
    low = _as_float(zone.get("low"))
    high = _as_float(zone.get("high"))
    direction = "bullish" if model.get("direction") == "buy" else "bearish"
    if low is None or high is None:
        return None

    candidates = [
        poi
        for poi in analysis.get("pois", [])
        if poi.get("direction") == direction
        and _as_float(poi.get("low")) is not None
        and _as_float(poi.get("high")) is not None
        and abs(float(poi.get("low")) - low) <= 1e-8
        and abs(float(poi.get("high")) - high) <= 1e-8
    ]
    timestamp = model.get("timestamp")
    if timestamp:
        by_time = [
            poi
            for poi in candidates
            if poi.get("timestamp") == timestamp or poi.get("activation_index") == model.get("index")
        ]
        if by_time:
            return by_time[-1]
    return candidates[-1] if candidates else None


def _find_setup_for_model(analysis: Dict[str, Any], model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    direction = "bullish" if model.get("direction") == "buy" else "bearish"
    matches = [
        setup
        for setup in analysis.get("setups", [])
        if setup.get("setup_type") == model.get("setup_type")
        and setup.get("strategy_direction") == direction
        and setup.get("index") == model.get("index")
    ]
    return matches[-1] if matches else None


def _walk_forward_model(analysis: Dict[str, Any], model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    signal_index = model.get("index")
    if signal_index is None:
        return None
    signal_index = int(signal_index)

    if model.get("status") == "trade_ready":
        setup = _find_setup_for_model(analysis, model)
        if not setup:
            return None
        abc = setup.get("abc") or {}
        external_abc = setup.get("external_abc") or {}
        if not all(
            [
                _nested_index_is_known(abc.get("a"), "index", signal_index),
                _nested_index_is_known(abc.get("b"), "index", signal_index),
                _nested_index_is_known(abc.get("b"), "swept_at", signal_index),
                _nested_index_is_known(abc.get("c"), "index", signal_index),
                _nested_index_is_known(external_abc.get("a"), "index", signal_index),
                _nested_index_is_known(external_abc.get("b"), "index", signal_index),
                _nested_index_is_known(external_abc.get("b"), "swept_at", signal_index),
                _nested_index_is_known(external_abc.get("c"), "index", signal_index),
            ]
        ):
            return None
        return dict(model)

    setup_type = model.get("setup_type")
    if setup_type in {"Type A", "Type B"}:
        poi = _find_poi_for_model(analysis, model)
        if not poi:
            return None
        if not all(
            [
                _index_is_known(poi.get("index"), signal_index),
                _index_is_known(poi.get("activation_index"), signal_index),
                _index_is_known(poi.get("context_event_index"), signal_index),
                _index_is_known(poi.get("cause_candle_index"), signal_index),
            ]
        ):
            return None

        linked_idm = next((x for x in analysis.get("idms", []) if x.get("index") == poi.get("linked_idm_index")), None)
        if linked_idm and not all(
            [
                _index_is_known(linked_idm.get("index"), signal_index),
                _index_is_known(linked_idm.get("swept_at"), signal_index),
            ]
        ):
            return None

        ranges = analysis.get("ranges", [])
        range_index = poi.get("range_index")
        if isinstance(range_index, int) and range_index < len(ranges) and not _range_is_walk_forward_safe(ranges[range_index], signal_index):
            return None

        external_ranges = analysis.get("external_ranges", [])
        external_range_index = poi.get("external_range_index")
        if (
            isinstance(external_range_index, int)
            and external_range_index < len(external_ranges)
            and not _range_is_walk_forward_safe(external_ranges[external_range_index], signal_index)
        ):
            return None

        return dict(model)

    if setup_type == "Type C":
        direction = "bullish" if model.get("direction") == "buy" else "bearish"
        candidates = [
            rng
            for rng in analysis.get("ranges", [])
            if rng.get("direction") == direction
            and str(rng.get("status", "")).startswith("validated")
            and (rng.get("validation_index") == signal_index or rng.get("trigger_event_index") == signal_index or rng.get("from_event_index") == signal_index)
        ]
        if not candidates:
            candidates = [
                rng
                for rng in analysis.get("ranges", [])
                if rng.get("direction") == direction
                and str(rng.get("status", "")).startswith("validated")
                and _index_is_known(rng.get("validation_index"), signal_index)
                and _index_is_known(rng.get("from_event_index"), signal_index)
            ]
        if not candidates:
            return None
        return dict(model) if _range_is_walk_forward_safe(candidates[-1], signal_index) else None

    return dict(model)


def _build_trade_from_model(
    model: Dict[str, Any],
    analysis: Dict[str, Any],
    candles: List[Dict[str, Any]],
    hold_bars: int,
    params: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    tested_model = dict(model)
    idx = model.get("index")
    zone = model.get("entry_zone") or {}
    zone_low = _as_float(zone.get("low"))
    zone_high = _as_float(zone.get("high"))
    if idx is None or zone_low is None or zone_high is None or int(idx) >= len(candles) - 1:
        tested_model["status"] = "not_testable"
        tested_model["outcome_reason"] = "missing index or entry zone"
        return None, tested_model

    idx = int(idx)
    zone_low, zone_high = min(zone_low, zone_high), max(zone_low, zone_high)
    side = model.get("direction")
    strategy_direction = "bullish" if side == "buy" else "bearish"
    rng = _range_for_model(analysis, model)
    range_a = (rng or {}).get("a") or {}
    range_b = (rng or {}).get("b") or {}
    range_c = (rng or {}).get("c") or {}

    entry = _first_number(zone.get("mid"), (zone_low + zone_high) / 2)
    if entry is None:
        tested_model["status"] = "not_testable"
        tested_model["outcome_reason"] = "missing entry"
        return None, tested_model

    a_price = _as_float(range_a.get("price"))
    b_price = _as_float(range_b.get("price"))
    c_price = _as_float(range_c.get("price"))
    stop = _first_number(model.get("stop_loss"), a_price)
    target_1 = _first_number(
        model.get("target_1"),
        b_price,
        c_price,
        _target_level_at(analysis, "IRL", strategy_direction, idx, entry),
    )
    target_final = _first_number(
        model.get("target_final"),
        model.get("target"),
        c_price,
        _target_level_at(analysis, "ERL", strategy_direction, idx, entry),
    )

    risk_floor = max(abs(zone_high - zone_low), abs(entry) * 0.003, 1e-9)
    if side == "buy":
        if stop is None or stop >= entry:
            stop = min(zone_low, entry - risk_floor)
        risk = abs(entry - stop)
        if target_1 is None or target_1 <= entry:
            target_1 = entry + risk
        if target_final is None or target_final <= entry:
            target_final = entry + risk * 2
    else:
        if stop is None or stop <= entry:
            stop = max(zone_high, entry + risk_floor)
        risk = abs(stop - entry)
        if target_1 is None or target_1 >= entry:
            target_1 = entry - risk
        if target_final is None or target_final >= entry:
            target_final = entry - risk * 2

    if risk <= 0:
        tested_model["status"] = "not_testable"
        tested_model["outcome_reason"] = "zero risk"
        return None, tested_model

    end = min(len(candles) - 1, idx + hold_bars)
    trigger_index: Optional[int] = None
    retest_tolerance = abs(entry) * float(params.get("retest_tolerance_pct", 0) or 0)
    trigger_low = zone_low - retest_tolerance
    trigger_high = zone_high + retest_tolerance
    for j in range(idx, end + 1):
        high = float(candles[j]["high"])
        low = float(candles[j]["low"])
        if low <= trigger_high and high >= trigger_low:
            trigger_index = j
            break

    if trigger_index is None:
        tested_model["status"] = "not_triggered"
        tested_model["outcome_reason"] = "price did not revisit the entry zone inside the follow-up window"
        tested_model["entry_zone"] = {"low": zone_low, "high": zone_high, "mid": entry}
        return None, tested_model

    exit_index = end
    exit_price = float(candles[end]["close"])
    exit_reason = "time_exit"

    for j in range(trigger_index + 1, end + 1):
        high = float(candles[j]["high"])
        low = float(candles[j]["low"])
        if side == "buy":
            if low <= stop:
                exit_index = j
                exit_price = stop
                exit_reason = "stop_loss"
                break
            if high >= target_final:
                exit_index = j
                exit_price = target_final
                exit_reason = "target"
                break
        else:
            if high >= stop:
                exit_index = j
                exit_price = stop
                exit_reason = "stop_loss"
                break
            if low <= target_final:
                exit_index = j
                exit_price = target_final
                exit_reason = "target"
                break

    r_multiple = (exit_price - entry) / risk if side == "buy" else (entry - exit_price) / risk
    status = "tested_target" if exit_reason == "target" else "tested_stop_loss" if exit_reason == "stop_loss" else "tested_time_exit"
    tested_model.update(
        {
            "status": status,
            "entry_zone": {"low": zone_low, "high": zone_high, "mid": entry},
            "stop_loss": stop,
            "target_1": target_1,
            "target": target_final,
            "target_final": target_final,
            "outcome_r": round(r_multiple, 3),
            "outcome_reason": exit_reason,
        }
    )
    trade = {
        "setup_type": model.get("setup_type"),
        "direction": side,
        "entry_index": trigger_index,
        "signal_index": idx,
        "exit_index": exit_index,
        "entry": float(entry),
        "stop_loss": float(stop),
        "target": float(target_final),
        "target_1": float(target_1),
        "target_final": float(target_final),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "r_multiple": round(r_multiple, 3),
        "win": r_multiple > 0,
    }
    return trade, tested_model


def _performance_summary(trades: List[Dict[str, Any]], opportunities: int, not_triggered: int) -> Dict[str, Any]:
    total = len(trades)
    wins = len([t for t in trades if t["win"]])
    losses = total - wins
    target_hits = len([t for t in trades if t["exit_reason"] == "target"])
    stop_hits = len([t for t in trades if t["exit_reason"] == "stop_loss"])
    time_exits = len([t for t in trades if t["exit_reason"] == "time_exit"])
    gross_profit_r = round(sum(t["r_multiple"] for t in trades if t["r_multiple"] > 0), 3)
    gross_loss_r = round(abs(sum(t["r_multiple"] for t in trades if t["r_multiple"] < 0)), 3)
    net_r = round(gross_profit_r - gross_loss_r, 3)
    profit_factor = round(gross_profit_r / gross_loss_r, 3) if gross_loss_r else (gross_profit_r if gross_profit_r else 0.0)
    profitability_pct = (
        round((gross_profit_r / (gross_profit_r + gross_loss_r)) * 100, 2)
        if gross_profit_r + gross_loss_r
        else 0.0
    )
    avg_r = round(sum(t["r_multiple"] for t in trades) / total, 3) if total else 0.0
    win_rate = round((wins / total) * 100, 2) if total else 0.0
    target_rate = round((target_hits / total) * 100, 2) if total else 0.0

    return {
        "total_trades": total,
        "total_opportunities": opportunities,
        "triggered_signals": total,
        "not_triggered": not_triggered,
        "wins": wins,
        "losses": losses,
        "target_hits": target_hits,
        "stop_hits": stop_hits,
        "time_exits": time_exits,
        "win_rate_pct": win_rate,
        "target_hit_rate_pct": target_rate,
        "gross_profit_r": gross_profit_r,
        "gross_loss_r": gross_loss_r,
        "net_r": net_r,
        "profit_factor": profit_factor,
        "profitability_pct": profitability_pct,
        "avg_r": avg_r,
    }


def _type_breakdown(tested_models: List[Dict[str, Any]], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    breakdown: Dict[str, Any] = {}
    setup_types = sorted({m.get("setup_type") for m in tested_models if m.get("setup_type")})
    for setup_type in setup_types:
        type_models = [m for m in tested_models if m.get("setup_type") == setup_type]
        type_trades = [t for t in trades if t.get("setup_type") == setup_type]
        not_triggered = len([m for m in type_models if m.get("status") == "not_triggered"])
        breakdown[setup_type] = _performance_summary(type_trades, len(type_models), not_triggered)
    return breakdown


def run_backtest(
    candles: List[Dict[str, Any]],
    params: Dict[str, Any],
    hold_bars: int,
    external_candles: Optional[List[Dict[str, Any]]] = None,
    internal_candles: Optional[List[Dict[str, Any]]] = None,
    micro_candles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    analysis = run_analysis(
        candles=candles,
        params=params,
        external_candles=external_candles,
        internal_candles=internal_candles,
        micro_candles=micro_candles,
    )
    model_counts, setup_models = _setup_model_summary(analysis)
    trades: List[Dict[str, Any]] = []
    tested_models: List[Dict[str, Any]] = []
    strict_walk_forward = True

    for model in setup_models:
        if model.get("status") == "invalidated":
            tested_models.append(model)
            continue

        signal_index = model.get("index")
        signal_timestamp = model.get("timestamp")
        if signal_index is None:
            tested = dict(model)
            tested["status"] = "walk_forward_rejected"
            tested["outcome_reason"] = "missing signal index for walk-forward validation"
            tested_models.append(tested)
            continue

        signal_index = int(signal_index)
        walk_model = _walk_forward_model(analysis, model)
        if not walk_model:
            tested = dict(model)
            tested["status"] = "walk_forward_rejected"
            tested["outcome_reason"] = "signal references data that was not available at the signal candle"
            tested_models.append(tested)
            continue

        walk_model["index"] = _timestamp_to_index(candles, walk_model.get("timestamp"), signal_index)
        walk_validation = "strict_reference_audit"

        trade, tested_model = _build_trade_from_model(walk_model, analysis, candles, hold_bars, params)
        tested_model["walk_forward_validated"] = True
        tested_model["walk_forward_validation"] = walk_validation
        tested_models.append(tested_model)
        if trade:
            trades.append(trade)

    not_triggered = len([m for m in tested_models if m.get("status") == "not_triggered"])
    walk_rejected = len([m for m in tested_models if m.get("status") == "walk_forward_rejected"])
    metrics = _performance_summary(trades, len(tested_models), not_triggered)
    type_breakdown = _type_breakdown(tested_models, trades)
    model_counts.update(
        {
            "tested": metrics["total_trades"],
            "wins": metrics["wins"],
            "losses": metrics["losses"],
            "time_exits": metrics["time_exits"],
            "not_triggered": not_triggered,
            "walk_forward_rejected": walk_rejected,
        }
    )
    metrics["walk_forward_rejected"] = walk_rejected
    metrics["walk_forward_mode"] = True
    metrics["strict_walk_forward"] = strict_walk_forward

    return {
        "metrics": metrics,
        "trades": trades,
        "model_counts": model_counts,
        "type_breakdown": type_breakdown,
        "setup_models": tested_models,
        "analysis_summary": analysis.get("summary", {}),
        "strategy_state": analysis.get("strategy_state", {}),
        "analysis": analysis,
    }
