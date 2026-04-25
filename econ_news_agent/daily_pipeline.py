from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from econ_news_agent.cnfin_fetcher import CnfinFetcher
from econ_news_agent.daily_sentiment import DailyHistoryStore, DailySentimentAnalyzer
from econ_news_agent.newsapi_fetcher import NewsAPIFetcher


class DailySentimentPipeline:
    def __init__(self, history_path: str | Path):
        self.cnfin_fetcher = CnfinFetcher()
        self.newsapi_fetcher = NewsAPIFetcher()
        self.analyzer = DailySentimentAnalyzer()
        self.history_store = DailyHistoryStore(history_path)

    def refresh(
        self,
        *,
        manual_url: str | None = None,
        source_type: str = "xinhua",
    ) -> dict[str, Any]:
        if source_type == "newsapi":
            snapshot = self.newsapi_fetcher.build_daily_source_snapshot()
        else:
            snapshot = self.cnfin_fetcher.build_daily_source_snapshot(manual_url=manual_url)

        if snapshot.get("briefing_status") == "missing":
            return {
                **snapshot,
                "article_count": 0,
                "daily_score": "--",
                "sentiment_label": "未生成",
                "engine_mode": "briefing-unavailable",
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        if snapshot.get("newsapi_status") in {"missing-key", "no-results", "request-failed"}:
            return {
                **snapshot,
                "article_count": 0,
                "daily_score": "--",
                "sentiment_label": "未生成",
                "engine_mode": "briefing-unavailable",
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        sentiment = self.analyzer.analyze_snapshot(snapshot)
        result = {
            **snapshot,
            **sentiment,
            "article_count": len(snapshot.get("items", [])),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.history_store.upsert(result)
        return result

    def load_history(self) -> list[dict[str, Any]]:
        return self.history_store.load()

    def latest(self) -> dict[str, Any] | None:
        history = self.load_history()
        return history[0] if history else None
