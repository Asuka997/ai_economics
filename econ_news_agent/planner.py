from __future__ import annotations


def build_plan(question: str) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = [
        {
            "tool": "conversation_memory",
            "purpose": "读取当前会话中的新闻主题与用户最近关注点",
        }
    ]

    background_markers = ("为什么", "原因", "解释", "背景", "机制")
    compare_markers = ("比较", "对比", "类似", "去年", "以前", "历史")
    impact_markers = ("影响", "利好", "利空", "行业", "板块", "资产")

    if any(marker in question for marker in background_markers):
        plan.append(
            {
                "tool": "knowledge_retrieval",
                "purpose": "检索经济学概念与政策术语，为追问提供背景证据",
            }
        )
    if any(marker in question for marker in compare_markers):
        plan.append(
            {
                "tool": "historical_case_search",
                "purpose": "查找相似历史新闻案例，辅助做横向比较",
            }
        )
    if any(marker in question for marker in impact_markers):
        plan.append(
            {
                "tool": "impact_mapping",
                "purpose": "把新闻主题映射到行业、资产和宏观变量的可能影响",
            }
        )

    plan.append(
        {
            "tool": "response_synthesis",
            "purpose": "融合新闻、检索证据和会话记忆，生成最终回答",
        }
    )
    return plan
