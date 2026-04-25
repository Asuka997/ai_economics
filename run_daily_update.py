from __future__ import annotations

from pathlib import Path

from econ_news_agent.daily_pipeline import DailySentimentPipeline


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def main() -> None:
    pipeline = DailySentimentPipeline(DATA_DIR / "daily_sentiment_history.json")
    result = pipeline.refresh()
    print("date:", result["snapshot_date"])
    print("source_mode:", result["source_mode"])
    print("score:", result["daily_score"])
    print("label:", result["sentiment_label"])
    print("articles:", result["article_count"])


if __name__ == "__main__":
    main()
