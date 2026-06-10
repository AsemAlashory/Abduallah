from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class AlertConfig(BaseModel):
    enabled: bool = True
    webhook_token: Optional[str] = None


class AlertEvent(BaseModel):
    symbol: str
    timeframe: str
    signal_type: str
    direction: str
    price: float
    reason: str


class AlertLogResponse(BaseModel):
    config: AlertConfig
    logs: List[Dict[str, Any]]
