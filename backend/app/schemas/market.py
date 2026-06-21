from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field

from app.schemas.analyze import Candle


class MarketDataRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=30)
    interval: Literal["1m", "5m", "15m", "30m", "60m", "1h", "2h", "4h", "1d", "1wk", "1mo"] = "15m"
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"] = "3mo"


class MarketDataResponse(BaseModel):
    symbol: str
    interval: str
    period: str
    candles: List[Candle]


class LocalNqDatasetResponse(BaseModel):
    symbol: str
    source: str
    period: str
    internal_interval: str
    external_interval: str
    internal_candles: List[Candle]
    external_candles: List[Candle]
