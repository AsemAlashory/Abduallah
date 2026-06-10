from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AlertStore:
    def __init__(self) -> None:
        self.enabled: bool = True
        self.webhook_token: Optional[str] = None
        self.logs: List[Dict[str, Any]] = []

    def set_config(self, enabled: bool, webhook_token: Optional[str]) -> Dict[str, Any]:
        self.enabled = enabled
        self.webhook_token = webhook_token if webhook_token else None
        return self.get_config()

    def get_config(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "webhook_token": self.webhook_token,
        }

    def can_accept(self, token: Optional[str]) -> bool:
        if not self.enabled:
            return False
        if self.webhook_token and self.webhook_token != token:
            return False
        return True

    def add_log(self, payload: Dict[str, Any]) -> None:
        self.logs.insert(
            0,
            {
                "received_at": datetime.now(timezone.utc).isoformat(),
                **payload,
            },
        )
        self.logs = self.logs[:500]

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.logs[:limit]


alert_store = AlertStore()
