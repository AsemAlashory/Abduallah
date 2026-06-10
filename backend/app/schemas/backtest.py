from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.analyze import Candle, StrategyParameters


class BacktestRequest(BaseModel):
    candles: List[Candle]
    external_candles: List[Candle] | None = None
    internal_candles: List[Candle] | None = None
    micro_candles: List[Candle] | None = None
    parameters: StrategyParameters = StrategyParameters()
    hold_bars: int = Field(default=48, ge=1, le=500)


class BacktestResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    metrics: Dict[str, Any]
    trades: List[Dict[str, Any]]
    model_counts: Dict[str, Any] = {}
    type_breakdown: Dict[str, Any] = {}
    setup_models: List[Dict[str, Any]] = []
    analysis_summary: Dict[str, Any]
    strategy_state: Dict[str, Any] = {}
    analysis: Dict[str, Any] = {}
