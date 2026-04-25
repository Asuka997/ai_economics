from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any


NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

ECON_QUERIES = [
    '"宏观经济" OR GDP OR CPI OR PPI OR 通胀 OR 经济增长',
    '央行 OR 降准 OR 降息 OR LPR OR 财政政策 OR 专项债',
    '房地产 OR 消费 OR 出口 OR 外贸 OR 制造业 OR 新能源',
]


def clean_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


class NewsAPIFetcher:
    def __init__(self) -> None:
        self.api_key = os.getenv("NEWSAPI_API_KEY", "").strip()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _request(self, query: str, *, from_date: str, to_date: str, page_size: int = 20) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("NEWSAPI_API_KEY is not configured.")

        params = {
            "q": query,
            "language": "zh",
            "sortBy": "publishedAt",
            "pageSize": str(page_size),
            "from": from_date,
            "to": to_date,
            "apiKey": self.api_key,
        }
        url = f"{NEWSAPI_BASE_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if payload.get("status") != "ok":
            raise RuntimeError(payload.get("message", "NewsAPI request failed."))
        return payload

    def build_daily_source_snapshot(self) -> dict[str, Any]:
        logs: list[str] = []
        if not self.available:
            return {
                "snapshot_date": datetime.now().strftime("%Y-%m-%d"),
                "source_mode": "newsapi-missing-key",
                "source_label": "NewsAPI 未配置",
                "source_url": "https://newsapi.org/",
                "items": [],
                "newsapi_status": "missing-key",
                "fetch_log": ["未检测到 NEWSAPI_API_KEY，无法抓取多源新闻。"],
            }

        now = datetime.now()
        # 免费版有延迟，这里默认拉近 3 天窗口，避免当天完全没有样本。
        to_dt = now - timedelta(days=1)
        from_dt = to_dt - timedelta(days=2)
        from_date = from_dt.strftime("%Y-%m-%d")
        to_date = to_dt.strftime("%Y-%m-%d")
        logs.append(f"使用 NewsAPI 抓取多源经济新闻，时间窗口：{from_date} 至 {to_date}。")

        collected: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        try:
            for query in ECON_QUERIES:
                payload = self._request(query, from_date=from_date, to_date=to_date, page_size=20)
                articles = payload.get("articles", [])
                logs.append(f"关键词组“{query}”返回 {len(articles)} 条结果。")
                for article in articles:
                    title = clean_text(article.get("title", ""))
                    source_name = clean_text(article.get("source", {}).get("name", "NewsAPI"))
                    url = clean_text(article.get("url", ""))
                    if not title or not url:
                        continue
                    fingerprint = (title, source_name)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)

                    description = clean_text(article.get("description", ""))
                    content = clean_text(article.get("content", ""))
                    text = "。".join(part for part in [title, description, content] if part)
                    if len(text) < 20:
                        continue

                    collected.append(
                        {
                            "title": title,
                            "url": url,
                            "published_at": clean_text(article.get("publishedAt", "")),
                            "source": source_name,
                            "text": text[:600],
                        }
                    )
        except Exception as exc:
            return {
                "snapshot_date": now.strftime("%Y-%m-%d"),
                "source_mode": "newsapi-request-failed",
                "source_label": "NewsAPI 请求失败",
                "source_url": "https://newsapi.org/",
                "items": [],
                "newsapi_status": "request-failed",
                "fetch_log": logs + [f"NewsAPI 请求失败：{exc}"],
            }

        collected.sort(key=lambda item: item.get("published_at", ""), reverse=True)
        items = collected[:15]
        if not items:
            return {
                "snapshot_date": now.strftime("%Y-%m-%d"),
                "source_mode": "newsapi-no-results",
                "source_label": "NewsAPI 暂未返回可用经济新闻",
                "source_url": "https://newsapi.org/",
                "items": [],
                "newsapi_status": "no-results",
                "fetch_log": logs + ["当前时间窗口内未获得可用新闻样本。"],
            }

        return {
            "snapshot_date": now.strftime("%Y-%m-%d"),
            "source_mode": "newsapi-economy",
            "source_label": f"NewsAPI 多源经济新闻（样本 {len(items)} 条）",
            "source_url": "https://newsapi.org/",
            "items": items,
            "newsapi_status": "ok",
            "fetch_log": logs,
        }
