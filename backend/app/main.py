from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.schemas.alerts import AlertConfig, AlertEvent, AlertLogResponse
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse, HealthResponse
from app.schemas.backtest import BacktestRequest, BacktestResponse
from app.schemas.market import LocalNqDatasetResponse, MarketDataRequest, MarketDataResponse
from app.services.alert_store import alert_store
from app.services.backtest_engine import run_backtest
from app.services.market_data import fetch_local_nq_dataset, fetch_market_data
from app.services.strategy_engine import run_analysis


app = FastAPI(title="SMC Strategy Platform", version="0.2.0")

app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "smc-strategy-backend", "health": "/api/health"}


@app.head("/")
def root_head() -> dict:
    return {}


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="smc-strategy-backend", version="0.2.0")


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    if len(payload.candles) < 40:
        raise HTTPException(status_code=400, detail="At least 40 candles are required for analysis")
    if not payload.external_candles:
        raise HTTPException(status_code=400, detail="Phase 1 requires a separate external_candles feed")
    if not payload.internal_candles:
        raise HTTPException(status_code=400, detail="Phase 1 requires a separate auto-paired internal_candles feed")

    result = run_analysis(
        candles=[c.model_dump() for c in payload.candles],
        params=payload.parameters.model_dump(),
        weekly_candles=[c.model_dump() for c in payload.weekly_candles] if payload.weekly_candles else None,
        daily_candles=[c.model_dump() for c in payload.daily_candles] if payload.daily_candles else None,
        external_candles=[c.model_dump() for c in payload.external_candles] if payload.external_candles else None,
        internal_candles=[c.model_dump() for c in payload.internal_candles] if payload.internal_candles else None,
        micro_candles=[c.model_dump() for c in payload.micro_candles] if payload.micro_candles else None,
    )
    return AnalyzeResponse(**result)


@app.post("/api/backtest", response_model=BacktestResponse)
def backtest(payload: BacktestRequest) -> BacktestResponse:
    if len(payload.candles) < 60:
        raise HTTPException(status_code=400, detail="At least 60 candles are required for backtest")

    result = run_backtest(
        candles=[c.model_dump() for c in payload.candles],
        params=payload.parameters.model_dump(),
        hold_bars=payload.hold_bars,
        external_candles=[c.model_dump() for c in payload.external_candles] if payload.external_candles else None,
        internal_candles=[c.model_dump() for c in payload.internal_candles] if payload.internal_candles else None,
        micro_candles=[c.model_dump() for c in payload.micro_candles] if payload.micro_candles else None,
    )
    return BacktestResponse(**result)


@app.post("/api/market/history", response_model=MarketDataResponse)
def market_history(payload: MarketDataRequest) -> MarketDataResponse:
    candles = fetch_market_data(symbol=payload.symbol, interval=payload.interval, period=payload.period)
    if not candles:
        raise HTTPException(status_code=404, detail="No market data found for this symbol/interval/period")
    return MarketDataResponse(symbol=payload.symbol, interval=payload.interval, period=payload.period, candles=candles)


@app.get("/api/market/local-nq", response_model=LocalNqDatasetResponse)
def local_nq_dataset() -> LocalNqDatasetResponse:
    dataset = fetch_local_nq_dataset()
    if not dataset["internal_candles"]:
        raise HTTPException(status_code=404, detail="NQ_1H.csv was not found or did not contain valid candles")
    return LocalNqDatasetResponse(**dataset)


@app.get("/api/alerts/config", response_model=AlertConfig)
def get_alert_config() -> AlertConfig:
    return AlertConfig(**alert_store.get_config())


@app.post("/api/alerts/config", response_model=AlertConfig)
def set_alert_config(payload: AlertConfig) -> AlertConfig:
    cfg = alert_store.set_config(enabled=payload.enabled, webhook_token=payload.webhook_token)
    return AlertConfig(**cfg)


@app.post("/api/alerts/webhook")
def alert_webhook(payload: AlertEvent, token: Optional[str] = Query(default=None)) -> dict:
    if not alert_store.can_accept(token=token):
        raise HTTPException(status_code=403, detail="Webhook rejected: invalid token or disabled config")

    alert_store.add_log(payload.model_dump())
    return {"status": "received"}


@app.get("/api/alerts/webhook")
def alert_webhook_info() -> dict:
    return {
        "status": "ready",
        "message": "Webhook endpoint is active. Use POST with JSON body, not GET.",
        "required_fields": ["symbol", "timeframe", "signal_type", "direction", "price", "reason"],
        "example_post_url": "/api/alerts/webhook?token=YOUR_TOKEN",
    }


@app.get("/api/alerts/logs", response_model=AlertLogResponse)
def get_alert_logs(limit: int = Query(default=100, ge=1, le=500)) -> AlertLogResponse:
    return AlertLogResponse(config=AlertConfig(**alert_store.get_config()), logs=alert_store.get_logs(limit=limit))
