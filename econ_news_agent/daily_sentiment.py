from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from econ_news_agent.analyzer import (
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    detect_topic,
    parse_llm_json,
    short_summary,
)
from econ_news_agent.llm_client import OpenAICompatibleClient


SCORE_HINTS = {
    "偏正面": 68,
    "中性偏积极": 60,
    "中性偏观察": 50,
    "中性偏谨慎": 42,
    "偏谨慎": 35,
}


def numeric_sentiment_score(text: str, topic: str) -> int:
    positive = sum(text.count(word) for word in POSITIVE_WORDS)
    negative = sum(text.count(word) for word in NEGATIVE_WORDS)
    raw = 50 + (positive - negative) * 8

    if topic in {"货币政策", "财政政策"} and any(word in text for word in ["支持", "提振", "回升"]):
        raw += 6
    if topic in {"房地产", "国际贸易"} and any(word in text for word in ["压力", "下滑", "走弱"]):
        raw -= 6
    return max(0, min(100, int(raw)))


def label_from_score(score: int) -> str:
    if score >= 65:
        return "偏积极"
    if score >= 55:
        return "中性偏积极"
    if score >= 45:
        return "中性偏观察"
    if score >= 35:
        return "中性偏谨慎"
    return "偏谨慎"


class DailySentimentAnalyzer:
    def __init__(self) -> None:
        self.client = OpenAICompatibleClient()

    def analyze_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        items = snapshot.get("items", [])
        if self.client.available:
            try:
                return self._llm_daily_analysis(snapshot)
            except Exception as exc:
                fallback = self._heuristic_daily_analysis(snapshot, items)
                fallback["llm_error"] = str(exc)
                fallback["fallback_reason"] = "在线模型分析失败，已自动回退到本地引擎。"
                return fallback
        fallback = self._heuristic_daily_analysis(snapshot, items)
        fallback["fallback_reason"] = "未检测到在线模型 API Key，当前使用本地引擎。"
        return fallback

    def _heuristic_daily_analysis(
        self,
        snapshot: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        item_summaries: list[dict[str, Any]] = []
        topic_distribution: dict[str, int] = {}

        for item in items:
            text = item.get("text", "")
            topic = detect_topic(text)
            score = numeric_sentiment_score(text, topic)
            label = label_from_score(score)
            topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
            item_summaries.append(
                {
                    "title": item.get("title", "未命名新闻"),
                    "url": item.get("url", ""),
                    "topic": topic,
                    "score": score,
                    "sentiment": label,
                    "summary": short_summary(text, limit=1),
                }
            )

        scores = [item["score"] for item in item_summaries] or [50]
        daily_score = int(round(mean(scores)))
        sentiment_label = label_from_score(daily_score)
        dominant_topics = sorted(topic_distribution.items(), key=lambda pair: pair[1], reverse=True)

        positive_drivers = [
            f"{item['title']}：{item['summary']}"
            for item in sorted(item_summaries, key=lambda row: row["score"], reverse=True)
            if item["score"] >= 55
        ][:3]
        negative_drivers = [
            f"{item['title']}：{item['summary']}"
            for item in sorted(item_summaries, key=lambda row: row["score"])
            if item["score"] <= 45
        ][:3]

        if not positive_drivers:
            positive_drivers = ["当天新闻中暂未出现特别强的系统性利好信号。"]
        if not negative_drivers:
            negative_drivers = ["当天新闻中暂未出现特别强的系统性风险冲击。"]

        dominant_topic_text = "、".join(topic for topic, _ in dominant_topics[:3]) or "其他"
        summary = (
            f"基于 {len(item_summaries)} 条来自新华财经的新闻样本，当日市场情绪指数为 {daily_score}/100，"
            f"整体判断为“{sentiment_label}”。主要信息集中在 {dominant_topic_text}。"
        )
        watchpoints = [
            "关注后续是否有更多政策落地或数据验证，避免把单日新闻热度直接理解为趋势反转。",
            "如果后续负面主题持续增加，短期情绪指数可能回落。",
        ]

        return {
            "daily_score": daily_score,
            "sentiment_label": sentiment_label,
            "summary": summary,
            "positive_drivers": positive_drivers,
            "negative_drivers": negative_drivers,
            "watchpoints": watchpoints,
            "topic_distribution": topic_distribution,
            "item_summaries": item_summaries,
            "engine_mode": "local-daily-sentiment",
        }

    def _llm_daily_analysis(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        prompt_items = []
        for item in snapshot.get("items", [])[:5]:
            text = item.get("text", "")
            prompt_items.append(
                {
                    "title": item.get("title", "未命名新闻"),
                    "url": item.get("url", ""),
                    "published_at": item.get("published_at", ""),
                    "summary_hint": short_summary(text, limit=1),
                    "text_excerpt": text[:180],
                }
            )
        prompt_payload = {
            "snapshot_date": snapshot.get("snapshot_date"),
            "source_mode": snapshot.get("source_mode"),
            "source_label": snapshot.get("source_label"),
            "items": prompt_items,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名经济新闻情绪指数分析助手。请基于输入新闻集合输出 JSON，字段必须包含 "
                    "daily_score, sentiment_label, summary, positive_drivers, negative_drivers, "
                    "watchpoints, topic_distribution, item_summaries, engine_mode。"
                    "daily_score 取值 0 到 100。item_summaries 中每项包含 title, topic, score, sentiment, summary, url。"
                    "请只输出 JSON 对象，不要输出 markdown，不要额外解释。"
                    "请尽量简洁：positive_drivers, negative_drivers, watchpoints 各最多 3 条，item_summaries 最多 5 条。"
                ),
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
        ]
        parsed = parse_llm_json(self.client.chat(messages, temperature=0.1))
        parsed = self._normalize_llm_output(parsed)
        parsed["engine_mode"] = "openai-compatible-daily-llm"
        return parsed

    def _normalize_llm_output(self, parsed: dict[str, Any]) -> dict[str, Any]:
        score = int(parsed.get("daily_score", 50))
        parsed["daily_score"] = max(0, min(100, score))
        parsed["sentiment_label"] = label_from_score(parsed["daily_score"])
        parsed["summary"] = str(parsed.get("summary", "暂无综合结论。"))
        parsed["positive_drivers"] = list(parsed.get("positive_drivers", []))[:3]
        parsed["negative_drivers"] = list(parsed.get("negative_drivers", []))[:3]
        parsed["watchpoints"] = list(parsed.get("watchpoints", []))[:3]
        parsed["topic_distribution"] = dict(parsed.get("topic_distribution", {}))

        item_summaries = parsed.get("item_summaries", [])
        normalized_items: list[dict[str, Any]] = []
        for item in item_summaries:
            normalized = dict(item)
            try:
                item_score = int(normalized.get("score", 50))
            except (TypeError, ValueError):
                item_score = 50
            item_score = max(0, min(100, item_score))
            normalized["score"] = item_score
            normalized["sentiment"] = label_from_score(item_score)
            normalized["title"] = str(normalized.get("title", "未命名新闻"))
            normalized["topic"] = str(normalized.get("topic", "其他"))
            normalized["summary"] = str(normalized.get("summary", "暂无摘要。"))
            normalized["url"] = str(normalized.get("url", ""))
            normalized_items.append(normalized)
        parsed["item_summaries"] = normalized_items[:5]
        if not parsed["positive_drivers"]:
            parsed["positive_drivers"] = ["当天新闻中暂未出现特别强的系统性利好信号。"]
        if not parsed["negative_drivers"]:
            parsed["negative_drivers"] = ["当天新闻中暂未出现特别强的系统性风险冲击。"]
        if not parsed["watchpoints"]:
            parsed["watchpoints"] = ["后续仍需结合更多经济数据验证单日新闻情绪。"]
        return parsed


class DailyHistoryStore:
    def __init__(self, history_path: str | Path):
        self.history_path = Path(history_path)
        if not self.history_path.exists():
            self.history_path.write_text("[]", encoding="utf-8")

    def load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def save(self, history: list[dict[str, Any]]) -> None:
        self.history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        history = self.load()
        filtered = [item for item in history if item.get("snapshot_date") != snapshot.get("snapshot_date")]
        filtered.append(snapshot)
        filtered.sort(key=lambda row: row.get("snapshot_date", ""), reverse=True)
        self.save(filtered[:30])
        return filtered[:30]
