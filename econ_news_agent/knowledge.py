from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KnowledgeItem:
    item_id: str
    title: str
    content: str
    category: str
    tags: list[str]
    impact_targets: list[str]
    date: str
    source: str
    kind: str

    @property
    def searchable_text(self) -> str:
        tags_blob = " ".join(self.tags)
        targets_blob = " ".join(self.impact_targets)
        return f"{self.title}\n{self.content}\n{self.category}\n{tags_blob}\n{targets_blob}"


class KnowledgeRetriever:
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        try:
            raw_items = json.loads(self.data_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise RuntimeError(f"知识库文件不存在：{self.data_path}，请确保 data/knowledge_base.json 已上传。")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"知识库文件格式错误：{exc}") from exc
        self.items = [
            KnowledgeItem(
                item_id=item["id"],
                title=item["title"],
                content=item["content"],
                category=item["category"],
                tags=item.get("tags", []),
                impact_targets=item.get("impact_targets", []),
                date=item.get("date", ""),
                source=item.get("source", ""),
                kind=item.get("kind", "concept"),
            )
            for item in raw_items
        ]
        corpus = [item.searchable_text for item in self.items]
        # Use character n-grams so retrieval remains robust for Chinese text
        # without requiring an external tokenizer dependency.
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        kinds: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda pair: pair[1], reverse=True)

        results: list[dict[str, Any]] = []
        for index, score in indexed:
            item = self.items[index]
            if kinds and item.kind not in kinds:
                continue
            if score <= 0:
                continue
            results.append(
                {
                    "id": item.item_id,
                    "title": item.title,
                    "content": item.content,
                    "category": item.category,
                    "tags": item.tags,
                    "impact_targets": item.impact_targets,
                    "date": item.date,
                    "source": item.source,
                    "kind": item.kind,
                    "score": float(score),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def sample_cases(self, *, limit: int = 3) -> list[dict[str, Any]]:
        cases = [item for item in self.items if item.kind == "case"]
        return [
            {
                "title": item.title,
                "content": item.content,
                "category": item.category,
                "date": item.date,
                "tags": item.tags,
            }
            for item in cases[:limit]
        ]
