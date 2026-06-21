from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Candle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class StrategyParameters(BaseModel):
    major_length: int = Field(default=45, ge=1, le=500)
    internal_length: int = Field(default=45, ge=1, le=500)
    n_candles: int = Field(default=2, ge=1, le=20)
    break_confirmation: Literal["close"] = "close"
    min_fvg_size: float = Field(default=0.0, ge=0.0)
    retest_tolerance_pct: float = Field(default=0.0015, ge=0.0, le=0.02)
    external_erl_target: Optional[float] = None
    external_invalidation_level: Optional[float] = None
    micro_length: int = Field(default=6, ge=2, le=80)
    external_timeframe: str = "4h"
    internal_timeframe: str = "1h"
    chart_timeframe: str = "4h"


class AnalyzeRequest(BaseModel):
    candles: List[Candle]
    weekly_candles: Optional[List[Candle]] = None
    daily_candles: Optional[List[Candle]] = None
    external_candles: Optional[List[Candle]] = None
    internal_candles: Optional[List[Candle]] = None
    micro_candles: Optional[List[Candle]] = None
    parameters: StrategyParameters = StrategyParameters()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class AnalyzeResponse(BaseModel):
    summary: Dict
    swings: List[Dict]
    structure_events: List[Dict]
    sweeps: List[Dict]
    idms: List[Dict]
    external_ranges: List[Dict] = []
    ranges: List[Dict]
    pois: List[Dict]
    liquidity_targets: List[Dict]
    phase_1: Dict = {}
    stop_hunts: List[Dict] = []
    strategy_state: Dict
    setups: List[Dict]
    movement_legs: List[Dict] = []
    trendline_liquidity: List[Dict] = []
    session_liquidity: List[Dict] = []
    correction_protocols: List[Dict] = []
