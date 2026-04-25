from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, profile_path: str | Path):
        self.profile_path = Path(profile_path)
        if not self.profile_path.exists():
            self.profile_path.write_text("{}", encoding="utf-8")

    def load_profiles(self) -> dict[str, Any]:
        try:
            return json.loads(self.profile_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def load_profile(self, user_id: str) -> dict[str, Any]:
        profiles = self.load_profiles()
        return profiles.get(
            user_id,
            {
                "interested_topics": {},
                "preferred_style": "课程答辩版",
                "analysis_count": 0,
                "recent_topics": [],
            },
        )

    def update_profile(self, user_id: str, analysis: dict[str, Any]) -> dict[str, Any]:
        profiles = self.load_profiles()
        profile = self.load_profile(user_id)

        topic = analysis.get("topic", "其他")
        topic_counter = Counter(profile.get("interested_topics", {}))
        topic_counter[topic] += 1

        recent_topics = profile.get("recent_topics", [])
        recent_topics.append(topic)
        recent_topics = recent_topics[-5:]

        updated = {
            "interested_topics": dict(topic_counter),
            "preferred_style": profile.get("preferred_style", "课程答辩版"),
            "analysis_count": int(profile.get("analysis_count", 0)) + 1,
            "recent_topics": recent_topics,
        }
        profiles[user_id] = updated
        self.profile_path.write_text(
            json.dumps(profiles, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return updated
