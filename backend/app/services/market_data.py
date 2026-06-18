from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import yfinance as yf


_BINANCE_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}


def _normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper().replace("-", "")
    if s.endswith("=X"):
        s = s[:-2]
    return s


def _period_to_limit(period: str) -> int:
    mapping = {
        "1d": 96,
        "5d": 300,
        "1mo": 720,
        "3mo": 1000,
        "6mo": 1000,
        "1y": 1000,
        "2y": 1000,
    }
    return mapping.get(period, 500)


def _to_binance_interval(interval: str) -> str:
    if interval == "1wk":
        return "1w"
    return "1h" if interval == "60m" else interval


def _aggregate_candles(candles: List[Dict[str, Any]], group_size: int) -> List[Dict[str, Any]]:
    aggregated: List[Dict[str, Any]] = []
    for i in range(0, len(candles), group_size):
        group = candles[i : i + group_size]
        if not group:
            continue
        aggregated.append(
            {
                "timestamp": group[0]["timestamp"],
                "open": float(group[0]["open"]),
                "high": max(float(c["high"]) for c in group),
                "low": min(float(c["low"]) for c in group),
                "close": float(group[-1]["close"]),
                "volume": sum(float(c.get("volume") or 0) for c in group),
            }
        )
    return aggregated


def _resample_candles(candles: List[Dict[str, Any]], rule: str) -> List[Dict[str, Any]]:
    if not candles:
        return []
    df = pd.DataFrame(candles)
    df["_time"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["_time", "open", "high", "low", "close"]).sort_values("_time")
    if df.empty:
        return []
    df = df.set_index("_time")
    resampled = df.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    output: List[Dict[str, Any]] = []
    for ts, row in resampled.iterrows():
        output.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume") or 0),
            }
        )
    return output


def _csv_rows_to_candles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rename_map = {
        "Price": "timestamp",
        "Datetime": "timestamp",
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    required = ["timestamp", "open", "high", "low", "close"]
    if any(col not in df.columns for col in required):
        return []

    candles: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        ts = pd.to_datetime(row["timestamp"], utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        candles.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume") or 0),
            }
        )
    return candles


def _fetch_yfinance(symbol: str, interval: str, period: str) -> List[Dict[str, Any]]:
    yf_interval = "1h" if interval in {"4h", "60m"} else interval
    data = yf.download(symbol.strip(), period=period, interval=yf_interval, progress=False)
    if data.empty:
        return []

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(col[0]) for col in data.columns]

    df = data.reset_index()
    timestamp_col = "Datetime" if "Datetime" in df.columns else "Date"
    candles: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        ts = pd.to_datetime(row[timestamp_col], utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        candles.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume") or 0),
            }
        )

    if interval == "4h":
        return _resample_candles(candles, "4h")
    return candles


def fetch_local_nq_dataset() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    csv_path = root / "NQ_1H.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, skiprows=[1, 2])
        internal = _csv_rows_to_candles(df)
        source = str(csv_path)
    else:
        internal = _fetch_yfinance("NQ=F", "1h", "1y")
        source = "yfinance:NQ=F"

    return {
        "symbol": "NQ=F",
        "source": source,
        "period": "1y",
        "internal_interval": "1h",
        "external_interval": "4h",
        "internal_candles": internal,
        "external_candles": _resample_candles(internal, "4h"),
    }


def fetch_market_data(symbol: str, interval: str, period: str) -> List[Dict[str, Any]]:
    sym = _normalize_symbol(symbol)
    bi = _to_binance_interval(interval)

    if bi not in _BINANCE_INTERVALS:
        raise ValueError("Unsupported interval for live market source")

    if not (sym.endswith("USDT") or sym.endswith("USDC") or sym.endswith("BUSD")):
        return _fetch_yfinance(symbol, interval, period)

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": sym,
        "interval": bi,
        "limit": _period_to_limit(period),
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list):
        return []

    candles: List[Dict[str, Any]] = []
    for row in rows:
        ts = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc).isoformat()
        candles.append(
            {
                "timestamp": ts,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )

    if interval == "4h" and bi == "1h":
        return _resample_candles(candles, "4h")
    return candles
