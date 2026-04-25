from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

# 加载 .env 文件中的环境变量（本地开发用，生产环境直接设系统环境变量）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from econ_news_agent.analyzer import EconNewsAnalyzer
from econ_news_agent.daily_pipeline import DailySentimentPipeline
from econ_news_agent.knowledge import KnowledgeRetriever
from econ_news_agent.llm_client import PROVIDER_PRESETS
from econ_news_agent.memory import MemoryStore


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
USER_ID = "course_team"
COURSE_LOGO_PATH = ROOT / "logo.png"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 25%),
                radial-gradient(circle at top right, rgba(180, 83, 9, 0.16), transparent 30%),
                linear-gradient(180deg, #F3EFE5 0%, #FAF8F2 65%, #EFE4D0 100%);
        }
        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(18, 77, 72, 0.94), rgba(146, 64, 14, 0.88));
            color: #FFF7ED;
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 22px 60px rgba(68, 64, 60, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
            letter-spacing: 0.01em;
        }
        .hero p {
            margin: 0.6rem 0 0;
            font-size: 1rem;
            line-height: 1.6;
        }
        .stat-card {
            background: rgba(255, 251, 235, 0.82);
            border: 1px solid rgba(146, 64, 14, 0.16);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            min-height: 120px;
            box-shadow: 0 10px 30px rgba(120, 113, 108, 0.08);
        }
        .stat-label {
            color: #9A3412;
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .stat-value {
            color: #134E4A;
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 0.35rem;
            line-height: 1.5;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(15, 118, 110, 0.12);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
            backdrop-filter: blur(8px);
        }
        .chip {
            display: inline-block;
            background: #E7F6F2;
            color: #115E59;
            border: 1px solid rgba(15, 118, 110, 0.18);
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            margin: 0.15rem 0.35rem 0.15rem 0;
            font-size: 0.85rem;
        }
        .note {
            color: #57534E;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .signal-positive {
            background: rgba(220, 252, 231, 0.65);
            border: 1px solid rgba(34, 197, 94, 0.22);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.65rem;
        }
        .signal-negative {
            background: rgba(254, 226, 226, 0.65);
            border: 1px solid rgba(239, 68, 68, 0.18);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(max_entries=1)
def get_services() -> tuple[KnowledgeRetriever, EconNewsAnalyzer, MemoryStore, DailySentimentPipeline]:
    retriever = KnowledgeRetriever(DATA_DIR / "knowledge_base.json")
    analyzer = EconNewsAnalyzer(DATA_DIR / "few_shot_examples.json")
    memory = MemoryStore(DATA_DIR / "user_profiles.json")
    daily_pipeline = DailySentimentPipeline(DATA_DIR / "daily_sentiment_history.json")
    return retriever, analyzer, memory, daily_pipeline


def init_session(retriever: KnowledgeRetriever, daily_pipeline: DailySentimentPipeline) -> None:
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("news_text", "")
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("agent_plan", [])
    st.session_state.setdefault("retrieved_docs", [])
    st.session_state.setdefault("current_profile", {})
    st.session_state.setdefault("sample_cases", retriever.sample_cases(limit=3))
    st.session_state.setdefault("daily_snapshot", daily_pipeline.latest())
    st.session_state.setdefault("analysis_history", [])


def render_stat_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_engine_name(engine_mode: str) -> str:
    if engine_mode == "briefing-unavailable":
        return "未运行"
    if engine_mode.startswith("local"):
        return "本地引擎"
    model_name = st.session_state.get(
        "user_model",
        os.getenv("ARK_MODEL", "doubao-seed-2-0-mini-260215"),
    ).strip()
    return model_name


def render_chip_row(values: list[str]) -> None:
    chips = "".join(f'<span class="chip">{value}</span>' for value in values)
    st.markdown(chips or '<span class="note">暂无</span>', unsafe_allow_html=True)


def build_cot_steps(analysis: dict, retrieved_docs: list[dict]) -> list[dict[str, str]]:
    evidence_titles = [doc["title"] for doc in retrieved_docs[:3]]
    evidence_text = "、".join(evidence_titles) if evidence_titles else "当前主要依据为新闻原文"
    targets = "、".join(analysis.get("impact_targets", [])[:4]) or "宏观经济主体"
    risk = analysis.get("risk_points", ["后续仍需结合更多数据验证。"])[0]
    model_engine = display_engine_name(analysis.get("engine_mode", "local-rule-engine"))
    topic_method = (
        f"当前使用的是 {model_engine}。系统先读取新闻里的政策词、行业词和宏观指标词，"
        "再结合结构化提示模板判断这条新闻最接近哪个经济主题。"
    )
    sentiment_method = (
        f"{model_engine} 会结合新闻中的方向性表达、政策措辞和结果描述来判断情绪。"
        "如果是在线模型，会按 few-shot 示例的输出风格做归纳；如果是本地引擎，则主要依据正负向关键词和规则。"
    )
    chain_method = (
        "系统会先根据主题找到对应的经济传导模板，再结合新闻原文补足“政策/事件 -> 影响对象 -> 结果变化”的链条。"
        "如果启用了在线模型，这一步会额外使用结构化提示进行推理生成。"
    )
    evidence_method = (
        "系统会从本地经济知识库里检索相关概念和历史案例，作为检索增强（RAG）证据；"
        "然后再把这些证据与新闻原文一起用于风险和结论校验。"
    )
    return [
        {
            "title": "步骤 1：识别主题",
            "technique": "提示工程 + 结构化分析",
            "method": topic_method,
            "result": f"根据新闻关键信息，系统先将事件归类为“{analysis.get('topic', '其他')}”，识别到的核心事件是：{analysis.get('core_event', '未识别到核心事件。')}",
        },
        {
            "title": "步骤 2：判断情绪",
            "technique": "COT 风格推理 + few-shot 输出约束",
            "method": sentiment_method,
            "result": f"结合新闻措辞和政策/数据方向，系统判断当前情绪为“{analysis.get('sentiment', '中性偏观察')}”。对应摘要为：{analysis.get('summary', '暂无摘要。')}",
        },
        {
            "title": "步骤 3：推导影响链",
            "technique": "经济学规则模板 + 链式推理",
            "method": chain_method,
            "result": f"系统判断这条新闻更可能先影响 {targets}。当前生成的传导逻辑为：{analysis.get('impact_chain', '新闻事件变化 -> 市场预期调整 -> 经济主体反应变化')}",
        },
        {
            "title": "步骤 4：证据与风险校验",
            "technique": "RAG 检索增强 + 风险提示生成",
            "method": evidence_method,
            "result": f"本次参考的知识证据包括：{evidence_text}。结合这些证据，当前需要重点注意：{risk}",
        },
    ]


def render_analysis(analysis: dict) -> None:
    stat_cols = st.columns(3)
    with stat_cols[0]:
        render_stat_card("经济主题", analysis["topic"])
    with stat_cols[1]:
        render_stat_card("情绪倾向", analysis["sentiment"])
    with stat_cols[2]:
        render_stat_card("分析引擎", display_engine_name(analysis["engine_mode"]))

    st.markdown(
        f"""
        <div class="section-card">
            <h3>新闻摘要</h3>
            <div class="note">{analysis["summary"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="section-card">
            <h3>核心事件</h3>
            <div class="note">{analysis["core_event"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="section-card"><h3>影响对象</h3></div>', unsafe_allow_html=True)
        render_chip_row(analysis["impact_targets"])
        st.markdown('<div class="section-card"><h3>影响链条</h3></div>', unsafe_allow_html=True)
        st.write(analysis["impact_chain"])
    with right:
        st.markdown('<div class="section-card"><h3>风险提示</h3></div>', unsafe_allow_html=True)
        for risk in analysis["risk_points"]:
            st.write(f"- {risk}")
        st.markdown('<div class="section-card"><h3>建议追问</h3></div>', unsafe_allow_html=True)
        for question in analysis["follow_up_questions"]:
            st.write(f"- {question}")

def run_analysis(
    news_text: str,
    retriever: KnowledgeRetriever,
    analyzer: EconNewsAnalyzer,
    memory: MemoryStore,
) -> None:
    topic_hint = news_text[:120]
    docs = retriever.search(topic_hint, top_k=4)
    analysis = analyzer.analyze(news_text, docs, force_local=st.session_state.get("force_local", False))
    profile = memory.update_profile(USER_ID, analysis)

    st.session_state.news_text = news_text
    st.session_state.retrieved_docs = docs
    st.session_state.analysis = analysis
    st.session_state.current_profile = profile
    st.session_state.agent_plan = [
        {"tool": "news_analysis", "purpose": "读取新闻并生成结构化经济分析"},
        {"tool": "knowledge_retrieval", "purpose": "检索经济概念和历史案例，为分析结果补充证据"},
    ]
    st.session_state.chat_history = []

    history_item = {
        "title": analysis["core_event"],
        "topic": analysis["topic"],
        "sentiment": analysis["sentiment"],
        "text": news_text,
    }
    filtered_history = [
        item for item in st.session_state.analysis_history if item["text"].strip() != news_text.strip()
    ]
    st.session_state.analysis_history = [history_item, *filtered_history][:6]


def render_history_card(
    retriever: KnowledgeRetriever,
    analyzer: EconNewsAnalyzer,
    memory: MemoryStore,
) -> None:
    st.markdown(
        """
        <div class="section-card">
            <h3>历史记录</h3>
            <div class="note">这里会保留当前会话中已经分析过的单条经济新闻，方便快速回看。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not st.session_state.analysis_history:
        st.markdown(
            """
            <div class="section-card">
                <div class="note">还没有历史记录，先分析一条新闻后这里会自动保存。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    latest = st.session_state.analysis_history[0]
    preview = latest["text"][:88] + ("..." if len(latest["text"]) > 88 else "")
    st.markdown(
        f"""
        <div class="section-card">
            <strong>最近一次分析：</strong>{latest['title']}<br>
            <span class="note">主题：{latest['topic']} | 情绪：{latest['sentiment']}</span><br>
            <span class="note">{preview}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"展开全部历史记录（{len(st.session_state.analysis_history)}）", expanded=False):
        for index, item in enumerate(st.session_state.analysis_history, start=1):
            content_col, action_col = st.columns([4.3, 1.1])
            with content_col:
                item_preview = item["text"][:90] + ("..." if len(item["text"]) > 90 else "")
                st.markdown(
                    f"""
                    <div class="section-card">
                        <strong>{item['title']}</strong><br>
                        主题：{item['topic']} | 情绪：{item['sentiment']}<br>
                        {item_preview}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with action_col:
                if st.button("重新载入", key=f"history_reload_{index}", use_container_width=True):
                    run_analysis(item["text"], retriever, analyzer, memory)
                    st.rerun()


def render_single_news_tab(
    retriever: KnowledgeRetriever,
    analyzer: EconNewsAnalyzer,
    memory: MemoryStore,
) -> None:
    input_col, result_col = st.columns([1.05, 1.35])
    with input_col:
        st.subheader("新闻输入区")
        default_text = st.session_state.news_text or (
            "央行宣布将通过结构性货币政策工具支持科技创新和消费扩大，"
            "同时市场关注后续利率与信贷投放节奏对经济修复的影响。"
        )
        news_text = st.text_area(
            "请输入或粘贴经济新闻正文",
            value=default_text,
            height=260,
            label_visibility="collapsed",
            placeholder="例如：央行、房地产、出口、消费、财政政策相关的新闻内容",
        )
        if st.button("开始分析", type="primary", use_container_width=True):
            with st.spinner("正在分析中，请稍候…"):
                run_analysis(news_text, retriever, analyzer, memory)

        render_history_card(retriever, analyzer, memory)

    with result_col:
        st.subheader("分析结果区")
        if st.session_state.analysis:
            render_analysis(st.session_state.analysis)
        else:
            st.info("输入一条经济新闻后，这里会显示结构化分析结果。")

    if st.session_state.analysis:
        st.subheader("连续追问区")
        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

        followup = st.chat_input("继续追问，例如：为什么这会影响房地产？")
        if followup:
            st.session_state.chat_history.append({"role": "user", "content": followup})
            with st.spinner("正在思考…"):
                query = f"{followup}\n主题：{st.session_state.analysis['topic']}"
                docs = retriever.search(query, top_k=4, kinds=["concept", "case"])
                followup_result = analyzer.answer_followup(
                    question=followup,
                    news_text=st.session_state.news_text,
                    analysis=st.session_state.analysis,
                    retrieved_docs=docs,
                    session_context={
                        "recent_questions": [item["content"] for item in st.session_state.chat_history[-4:]],
                        "profile": st.session_state.current_profile,
                    },
                    force_local=st.session_state.get("force_local", False),
                )
            st.session_state.retrieved_docs = docs
            st.session_state.agent_plan = followup_result["plan"]
            st.session_state.chat_history.append(
                {"role": "assistant", "content": followup_result["answer"]}
            )
            st.rerun()

        with st.expander("COT 分步推理", expanded=False):
            st.write("下面展示的是这条新闻的结构化推理链，用于说明模型如何从新闻文本走到分析结论。")
            for step in build_cot_steps(st.session_state.analysis, st.session_state.retrieved_docs):
                st.markdown(f"**{step['title']}**")
                st.write(f"使用技术：{step['technique']}")
                st.write(f"分析方式：{step['method']}")
                st.write(f"本步结论：{step['result']}")


def refresh_daily_snapshot(
    daily_pipeline: DailySentimentPipeline,
    *,
    manual_url: str | None = None,
    source_type: str = "xinhua",
) -> None:
    snapshot = daily_pipeline.refresh(manual_url=manual_url or None, source_type=source_type)
    st.session_state.daily_snapshot = snapshot


def render_daily_signals(title: str, items: list[str], positive: bool) -> None:
    box_class = "signal-positive" if positive else "signal-negative"
    st.markdown(f"#### {title}")
    for item in items:
        st.markdown(f'<div class="{box_class}">{item}</div>', unsafe_allow_html=True)


def render_sidebar_course_info() -> None:
    if COURSE_LOGO_PATH.exists():
        logo_col_left, logo_col_center, logo_col_right = st.columns([0.12, 0.76, 0.12])
        with logo_col_center:
            st.image(str(COURSE_LOGO_PATH), width=170)
    st.markdown(
        """
        <div class="section-card">
            <h3 style="font-size: 1.02rem; white-space: nowrap;">人工智能与经济学课程</h3>
            <div class="note">
            小组成员：<br>
            朱相甫、王哲、夏光宇
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_daily_tab(daily_pipeline: DailySentimentPipeline) -> None:
    snapshot = st.session_state.daily_snapshot
    control_col, board_col = st.columns([0.95, 1.45])

    with control_col:
        st.subheader("抓取控制区")
        source_choice = st.radio(
            "新闻来源",
            options=["新华财经早报", "NewsAPI 多源"],
            horizontal=True,
        )
        source_type = "newsapi" if source_choice == "NewsAPI 多源" else "xinhua"
        if source_type == "xinhua":
            st.markdown(
                """
                <div class="section-card">
                    <h3>自动抓取逻辑</h3>
                    <div class="note">
                    系统会在新华财经官网搜索“新华财经早报”，并只匹配当天发布的早报正文。
                    如果当天还没有检索到对应早报，就直接提示“当日财经早报尚未公布”，不再回退到首页聚合新闻。
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="section-card">
                    <h3>自动抓取逻辑</h3>
                    <div class="note">
                    系统会调用 NewsAPI 抓取多源经济新闻，并按宏观、政策、消费、地产、外贸等关键词筛选样本，
                    然后再统一生成每日综合情绪指数。免费 NewsAPI 可能存在一定时间延迟。
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        manual_url = st.text_input(
            "可选：手动指定新华财经详情页链接",
            placeholder="例如：https://www.cnfin.com/hs-lb/detail/20241118/4140146_1.html",
            disabled=source_type != "xinhua",
        )
        run_cols = st.columns(2)
        with run_cols[0]:
            if st.button("抓取今日新闻", type="primary", use_container_width=True):
                with st.spinner("正在抓取新闻并分析，请稍候…"):
                    refresh_daily_snapshot(daily_pipeline, source_type=source_type)
                st.rerun()
        with run_cols[1]:
            if st.button("使用手动链接", use_container_width=True):
                with st.spinner("正在抓取新闻并分析，请稍候…"):
                    refresh_daily_snapshot(
                        daily_pipeline,
                        manual_url=manual_url,
                        source_type=source_type,
                    )
                st.rerun()

        history = daily_pipeline.load_history()
        with st.expander("最近历史记录", expanded=False):
            if history:
                for row in history[:7]:
                    st.write(
                        f"- {row['snapshot_date']} | {row['daily_score']}/100 | "
                        f"{row['sentiment_label']} | {display_engine_name(row['engine_mode'])}"
                    )
            else:
                st.write("还没有历史记录，先点击“抓取今日新闻”。")

    with board_col:
        st.subheader("每日情绪看板")
        if not snapshot:
            st.info("点击左侧“抓取今日新闻”后，这里会显示新华财经日度情绪指数。")
            return

        if snapshot.get("briefing_status") == "missing":
            st.markdown(
                f"""
                <div class="section-card">
                    <h3>当日状态</h3>
                    <div class="note">
                    {snapshot['source_label']}<br>
                    查询时间：{snapshot['updated_at']}<br>
                    查询入口：<a href="{snapshot['source_url']}" target="_blank">{snapshot['source_url']}</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("查询日志", expanded=False):
                for line in snapshot.get("fetch_log", []):
                    st.write(f"- {line}")
            return

        if snapshot.get("newsapi_status") in {"missing-key", "no-results", "request-failed"}:
            st.markdown(
                f"""
                <div class="section-card">
                    <h3>当前状态</h3>
                    <div class="note">
                    {snapshot['source_label']}<br>
                    查询时间：{snapshot['updated_at']}<br>
                    数据入口：<a href="{snapshot['source_url']}" target="_blank">{snapshot['source_url']}</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("查询日志", expanded=False):
                for line in snapshot.get("fetch_log", []):
                    st.write(f"- {line}")
            return

        stat_cols = st.columns(4)
        with stat_cols[0]:
            render_stat_card("情绪指数", f"{snapshot['daily_score']}/100")
        with stat_cols[1]:
            render_stat_card("情绪标签", snapshot["sentiment_label"])
        with stat_cols[2]:
            render_stat_card("样本数量", str(snapshot["article_count"]))
        with stat_cols[3]:
            render_stat_card("分析引擎", display_engine_name(snapshot["engine_mode"]))

        st.markdown(
            f"""
            <div class="section-card">
                <h3>来源说明</h3>
                <div class="note">
                来源模式：{snapshot['source_mode']}<br>
                来源标题：{snapshot['source_label']}<br>
                更新时间：{snapshot['updated_at']}<br>
                来源链接：<a href="{snapshot['source_url']}" target="_blank">{snapshot['source_url']}</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if snapshot.get("fallback_reason"):
            error_detail = ""
            if snapshot.get("llm_error"):
                error_detail = f"<br><br>失败原因：{snapshot['llm_error']}"
            st.markdown(
                f"""
                <div class="section-card">
                    <h3>引擎状态</h3>
                    <div class="note">
                    {snapshot['fallback_reason']}{error_detail}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="section-card">
                <h3>综合结论</h3>
                <div class="note">{snapshot['summary']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        topic_col, watch_col = st.columns([1, 1])
        with topic_col:
            st.markdown('<div class="section-card"><h3>主题分布</h3></div>', unsafe_allow_html=True)
            topic_distribution = snapshot.get("topic_distribution", {})
            topic_rows = sorted(topic_distribution.items(), key=lambda pair: pair[1], reverse=True)
            for topic, count in topic_rows:
                st.write(f"- {topic}: {count}")
        with watch_col:
            st.markdown('<div class="section-card"><h3>关注点</h3></div>', unsafe_allow_html=True)
            for watchpoint in snapshot.get("watchpoints", []):
                st.write(f"- {watchpoint}")

        signal_cols = st.columns(2)
        with signal_cols[0]:
            render_daily_signals("正向驱动", snapshot.get("positive_drivers", []), positive=True)
        with signal_cols[1]:
            render_daily_signals("风险驱动", snapshot.get("negative_drivers", []), positive=False)

        with st.expander("样本新闻明细", expanded=False):
            for item in snapshot.get("item_summaries", []):
                st.markdown(
                    f"""
                    <div class="section-card">
                        <strong>{item['title']}</strong><br>
                        主题：{item['topic']} | 情绪：{item['sentiment']} | 分数：{item['score']}<br>
                        摘要：{item['summary']}<br>
                        <a href="{item['url']}" target="_blank">{item['url']}</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander("抓取日志", expanded=False):
            for line in snapshot.get("fetch_log", []):
                st.write(f"- {line}")


def render_config_tab(analyzer: EconNewsAnalyzer, daily_pipeline: DailySentimentPipeline) -> None:
    st.subheader("模型配置")
    st.markdown(
        '<div class="note">在此填写您自己的 API Key，配置后优先使用您的模型。配置仅在当前会话有效，刷新页面后需重新填写。</div>',
        unsafe_allow_html=True,
    )

    provider_labels = {"doubao": "豆包 (Ark)", "deepseek": "DeepSeek"}
    provider = st.selectbox(
        "模型提供商",
        options=list(provider_labels.keys()),
        format_func=lambda k: provider_labels[k],
    )
    preset = PROVIDER_PRESETS[provider]

    col_left, col_right = st.columns(2)
    with col_left:
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="填写您的 API Key",
            value=st.session_state.get("user_api_key", ""),
        )
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.get("user_base_url", preset["base_url"]),
        )
    with col_right:
        model = st.text_input(
            "模型名称",
            value=st.session_state.get("user_model", preset["model"]),
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("应用配置", type="primary", use_container_width=True):
            if not api_key.strip():
                st.error("API Key 不能为空")
            else:
                analyzer.client.reconfigure(api_key, base_url, model)
                daily_pipeline.reconfigure_client(api_key, base_url, model)
                st.session_state["user_api_key"] = api_key
                st.session_state["user_base_url"] = base_url
                st.session_state["user_model"] = model
                st.success(f"已切换到 {provider_labels[provider]} · {model}")
                st.rerun()

    # 当切换提供商时自动更新预设值
    if st.session_state.get("_last_provider") != provider:
        st.session_state["user_base_url"] = preset["base_url"]
        st.session_state["user_model"] = preset["model"]
        st.session_state["_last_provider"] = provider
        st.rerun()

    st.markdown("---")
    st.markdown("**当前生效配置**")
    current_key = analyzer.client.api_key
    st.markdown(
        f"""
        <div class="section-card">
            <div class="note">
            API Key：{"已配置 ✓" if current_key else "未配置"}<br>
            Base URL：{analyzer.client.base_url}<br>
            模型：{analyzer.client.model}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Econews | 经济新闻智能分析助手",
        page_icon=":bar_chart:",
        layout="wide",
    )
    inject_css()
    retriever, analyzer, memory, daily_pipeline = get_services()
    init_session(retriever, daily_pipeline)

    st.markdown(
        """
        <div class="hero">
            <h1>Econews</h1>
            <p>一个面向个人用户的经济新闻智能体：既能分析单条新闻，也能自动抓取新华财经数据，生成每日综合情绪指数与解释。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        render_sidebar_course_info()
        st.markdown("---")
        llm_available = analyzer.client.available
        engine_options = ["在线模型", "本地规则引擎"]
        default_index = 0 if llm_available else 1
        engine_choice = st.radio(
            "分析引擎",
            options=engine_options,
            index=default_index,
            disabled=not llm_available,
            help="请先在「配置」页面填写 API Key" if not llm_available else None,
        )
        st.session_state["force_local"] = (engine_choice == "本地规则引擎")
        st.markdown("---")
        st.subheader("样例新闻")
        for case in st.session_state.sample_cases:
            if st.button(case["title"], use_container_width=True):
                run_analysis(case["content"], retriever, analyzer, memory)
                st.rerun()

    news_tab, daily_tab, config_tab = st.tabs(["单条新闻分析", "每日情绪看板", "⚙️ 配置"])
    with news_tab:
        render_single_news_tab(retriever, analyzer, memory)
    with daily_tab:
        render_daily_tab(daily_pipeline)
    with config_tab:
        render_config_tab(analyzer, daily_pipeline)


if __name__ == "__main__":
    main()
