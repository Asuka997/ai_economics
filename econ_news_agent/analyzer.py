from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from econ_news_agent.llm_client import OpenAICompatibleClient
from econ_news_agent.planner import build_plan


TOPIC_KEYWORDS = {
    "货币政策": ["降息", "降准", "lpr", "mlf", "央行", "利率", "流动性", "货币政策"],
    "财政政策": ["财政", "赤字", "专项债", "国债", "补贴", "税收", "财政政策"],
    "房地产": ["房地产", "楼市", "房价", "购房", "按揭", "地产", "去库存"],
    "国际贸易": ["出口", "进口", "关税", "贸易", "外需", "跨境", "汇率"],
    "就业与消费": ["就业", "失业", "收入", "消费", "居民支出", "零售"],
    "通胀与物价": ["cpi", "ppi", "通胀", "物价", "猪价", "价格"],
    "产业政策": ["新能源", "芯片", "制造业", "人工智能", "产业", "补链", "科技"],
}

POSITIVE_WORDS = ["增长", "回升", "改善", "提振", "利好", "扩张", "增加", "支持", "修复"]
NEGATIVE_WORDS = ["下滑", "收缩", "压力", "走弱", "疲软", "利空", "风险", "下降", "波动"]

IMPACT_TEMPLATES = {
    "货币政策": {
        "targets": ["房地产", "银行", "券商", "成长股"],
        "chain": "政策利率或流动性环境变化 -> 融资成本和风险偏好调整 -> 信贷投放与资产估值变化 -> 行业表现分化",
        "risk": "若实体需求没有同步修复，宽松政策可能更多停留在预期层面。",
    },
    "财政政策": {
        "targets": ["基建", "建筑材料", "地方城投", "工业链"],
        "chain": "财政支出或税费安排变化 -> 基建和公共投资节奏调整 -> 上游原材料与中游设备需求变化 -> 相关行业盈利预期变化",
        "risk": "若地方财政约束较强，政策传导可能慢于市场预期。",
    },
    "房地产": {
        "targets": ["房地产", "家居建材", "银行", "消费"],
        "chain": "地产政策松紧变化 -> 购房需求与开发商资金面变化 -> 地产投资与产业链订单变化 -> 消费和金融预期联动变化",
        "risk": "居民收入预期与库存压力仍可能抑制政策效果。",
    },
    "国际贸易": {
        "targets": ["出口链", "航运", "制造业", "汇率敏感资产"],
        "chain": "外需或贸易规则变化 -> 出口订单与企业成本变化 -> 制造业利润和就业变化 -> 汇率与市场预期调整",
        "risk": "海外需求和地缘政治变化会放大不确定性。",
    },
    "就业与消费": {
        "targets": ["可选消费", "服务业", "电商", "旅游"],
        "chain": "就业和收入预期变化 -> 居民消费能力与消费意愿变化 -> 终端需求修复程度变化 -> 服务和零售行业表现变化",
        "risk": "居民部门若偏向储蓄，消费恢复速度会偏慢。",
    },
    "通胀与物价": {
        "targets": ["消费", "食品链", "债券市场", "货币政策预期"],
        "chain": "物价走势变化 -> 真实利率与政策空间变化 -> 居民购买力和市场预期变化 -> 资产定价与消费表现变化",
        "risk": "单月价格数据可能受季节性影响，不能直接等同于趋势反转。",
    },
    "产业政策": {
        "targets": ["科技", "制造业", "新能源", "高端装备"],
        "chain": "产业支持政策变化 -> 企业投资和研发预期变化 -> 产业链订单与竞争格局变化 -> 板块估值与资本开支节奏调整",
        "risk": "政策支持若缺少需求落地，行业盈利兑现可能弱于预期。",
    },
    "其他": {
        "targets": ["宏观经济", "权益市场", "企业预期"],
        "chain": "新闻事件变化 -> 市场预期调整 -> 不同行业风险收益再定价",
        "risk": "信息有限时，结论更适合视为方向性提示而非确定判断。",
    },
}


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"[。！？!?]\s*", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def detect_topic(text: str) -> str:
    lowered = text.lower()
    best_topic = "其他"
    best_score = 0
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def detect_sentiment(text: str) -> str:
    positive = sum(text.count(word) for word in POSITIVE_WORDS)
    negative = sum(text.count(word) for word in NEGATIVE_WORDS)
    if positive > negative:
        return "偏正面"
    if negative > positive:
        return "偏谨慎"
    return "中性偏观察"


def short_summary(text: str, *, limit: int = 2) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return "暂无可提取摘要。"
    return "；".join(sentences[:limit]) + "。"


def extract_event(text: str) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return "未识别到明确核心事件。"
    return sentences[0]


def build_followups(topic: str) -> list[str]:
    templates = {
        "货币政策": [
            "这次政策变化更可能先影响银行、地产还是成长股？",
            "此次措施与去年同类宽松政策相比有哪些差异？",
        ],
        "财政政策": [
            "财政发力更可能先传导到哪些行业？",
            "该政策对地方投资和基建链意味着什么？",
        ],
        "房地产": [
            "这条新闻对地产销售和地产链条的传导路径是什么？",
            "与此前去库存政策相比，这次力度如何？",
        ],
        "国际贸易": [
            "外需变化最先影响哪些出口行业？",
            "类似贸易新闻在历史上通常如何影响制造业？",
        ],
    }
    return templates.get(
        topic,
        [
            "这条新闻背后的经济学逻辑是什么？",
            "有没有历史上相似的新闻案例可以参考？",
        ],
    )


def load_few_shots(path: str | Path) -> list[dict[str, str]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def parse_llm_json(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain JSON.")
    return json.loads(match.group(0))


class EconNewsAnalyzer:
    def __init__(self, few_shot_path: str | Path):
        self.client = OpenAICompatibleClient()
        self.few_shots = load_few_shots(few_shot_path)

    def analyze(self, news_text: str, retrieved_docs: list[dict[str, Any]]) -> dict[str, Any]:
        if self.client.available:
            try:
                return self._llm_analyze(news_text, retrieved_docs)
            except Exception as exc:
                result = self._heuristic_analyze(news_text, retrieved_docs)
                result["llm_error"] = str(exc)
                result["fallback_reason"] = "在线模型分析失败，已自动回退到本地引擎。"
                return result
        return self._heuristic_analyze(news_text, retrieved_docs)

    def answer_followup(
        self,
        question: str,
        news_text: str,
        analysis: dict[str, Any],
        retrieved_docs: list[dict[str, Any]],
        session_context: dict[str, Any],
    ) -> dict[str, Any]:
        plan = build_plan(question)
        if self.client.available:
            try:
                answer = self._llm_followup(
                    question=question,
                    news_text=news_text,
                    analysis=analysis,
                    retrieved_docs=retrieved_docs,
                    session_context=session_context,
                )
                return {"answer": answer, "plan": plan}
            except Exception:
                pass  # fall through to heuristic answer

        evidence = [doc["title"] for doc in retrieved_docs] or ["当前主要依据为新闻原文与会话上下文"]
        topic = analysis.get("topic", "其他")
        answer = (
            f"从当前新闻与检索证据看，这个问题主要围绕“{topic}”展开。"
            f"结合新闻原文，比较稳妥的解释是：{analysis.get('impact_chain', '')}"
            f" 当前可引用的背景材料包括：{'、'.join(evidence)}。"
            f" 如果要继续细化，可以进一步追问行业、资产类别或历史对比。"
        )
        return {"answer": answer, "plan": plan}

    def _heuristic_analyze(
        self,
        news_text: str,
        retrieved_docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        topic = detect_topic(news_text)
        template = IMPACT_TEMPLATES.get(topic, IMPACT_TEMPLATES["其他"])
        evidence = [doc["title"] for doc in retrieved_docs]
        return {
            "summary": short_summary(news_text),
            "topic": topic,
            "core_event": extract_event(news_text),
            "sentiment": detect_sentiment(news_text),
            "impact_targets": template["targets"],
            "impact_chain": template["chain"],
            "risk_points": [template["risk"], "新闻可能只反映短期政策信号，后续还需结合数据验证。"],
            "follow_up_questions": build_followups(topic),
            "evidence": evidence,
            "engine_mode": "local-rule-engine",
        }

    def _llm_analyze(
        self,
        news_text: str,
        retrieved_docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        examples = json.dumps(self.few_shots, ensure_ascii=False, indent=2)
        context = json.dumps(retrieved_docs, ensure_ascii=False, indent=2)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名经济新闻智能分析助手。请严格输出 JSON，字段必须包含 "
                    "summary, topic, core_event, sentiment, impact_targets, impact_chain, "
                    "risk_points, follow_up_questions, evidence, engine_mode。"
                    "不要输出 markdown。不要做无依据的资产涨跌预测。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"以下是 few-shot 示例：\n{examples}\n\n"
                    f"以下是检索到的背景材料：\n{context}\n\n"
                    f"现在请分析这条经济新闻：\n{news_text}"
                ),
            },
        ]
        parsed = parse_llm_json(self.client.chat(messages))
        parsed["engine_mode"] = "openai-compatible-llm"
        return parsed

    def _llm_followup(
        self,
        *,
        question: str,
        news_text: str,
        analysis: dict[str, Any],
        retrieved_docs: list[dict[str, Any]],
        session_context: dict[str, Any],
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名经济新闻问答助手。请结合新闻原文、结构化分析结果、检索证据和会话记忆回答。"
                    "回答要简洁、专业、可用于课堂展示。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "news_text": news_text,
                        "analysis": analysis,
                        "retrieved_docs": retrieved_docs,
                        "session_context": session_context,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        return self.client.chat(messages)
