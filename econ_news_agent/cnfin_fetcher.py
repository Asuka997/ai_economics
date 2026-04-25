from __future__ import annotations

import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from typing import Any


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

SEARCH_BASE_URL = "https://search.cnfin.com/news"
BRIEFING_QUERY = "新华财经早报"

CANDIDATE_LISTING_URLS = [
    "https://m.cnfin.com/",
    "https://www.cnfin.com/news/index.html",
    "https://www.cnfin.com/macro/index.html",
    "https://www.cnfin.com/policy/index.html",
    "https://www.cnfin.com/industry/index.html",
]


def fetch_html(url: str, *, timeout: int = 20) -> str:
    request = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        header_charset = response.headers.get_content_charset()

    encodings: list[str] = []
    if header_charset:
        encodings.append(header_charset)

    meta_probe = raw[:2048].decode("ascii", "ignore")
    meta_match = re.search(r"charset=['\"]?([A-Za-z0-9._-]+)", meta_probe, re.I)
    if meta_match:
        encodings.append(meta_match.group(1))

    encodings.extend(["utf-8", "gb18030", "gbk"])
    tried: set[str] = set()
    for encoding in encodings:
        normalized = encoding.lower()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", "ignore")


def normalize_cnfin_url(url: str) -> str:
    href = url.strip()
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/"):
        href = "https://www.cnfin.com" + href

    desktop_match = re.search(
        r"https?://(?:www\.)?cnfin\.com/([^/]+)/detail/(\d{8})/(\d+_1\.html)",
        href,
    )
    if desktop_match:
        channel, day, article = desktop_match.groups()
        return f"https://m.cnfin.com/{channel}/zixun/{day}/{article}"

    nested_desktop_match = re.search(
        r"https?://(?:www\.)?cnfin\.com/([^/]+)/[^/]+/detail/(\d{8})/(\d+_1\.html)",
        href,
    )
    if nested_desktop_match:
        channel, day, article = nested_desktop_match.groups()
        return f"https://m.cnfin.com/{channel}/zixun/{day}/{article}"

    mobile_match = re.search(
        r"https?://m\.cnfin\.com/([^/]+)/+zixun/(\d{8})/(\d+_1\.html)",
        href,
    )
    if mobile_match:
        channel, day, article = mobile_match.groups()
        return f"https://m.cnfin.com/{channel}/zixun/{day}/{article}"

    return href


def clean_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_date_from_url(url: str) -> str:
    match = re.search(r"/(20\d{6})/", url)
    if not match:
        return ""
    compact = match.group(1)
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self.current_href: str | None = None
        self.current_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        href = attr_map.get("href", "") or ""
        if "detail/" in href or "/zixun/" in href:
            self.current_href = href
            self.current_parts = []

    def handle_data(self, data: str) -> None:
        if self.current_href is not None:
            self.current_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self.current_href is None:
            return
        title = clean_text("".join(self.current_parts))
        self.links.append({"href": self.current_href, "title": title})
        self.current_href = None
        self.current_parts = []


class SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self.current_attrs: dict[str, str] | None = None
        self.current_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = {key: (value or "") for key, value in attrs}
        onclick = attr_map.get("onclick", "")
        href = attr_map.get("href", "")
        if "checkUrl(" in onclick or href == "javascript:void(0);":
            self.current_attrs = attr_map
            self.current_parts = []

    def handle_data(self, data: str) -> None:
        if self.current_attrs is not None:
            self.current_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self.current_attrs is None:
            return
        title = clean_text("".join(self.current_parts))
        onclick = self.current_attrs.get("onclick", "")
        match = re.search(r'checkUrl\("([^"]+)"\)', onclick)
        raw_url = match.group(1).replace("\\/", "/") if match else ""
        if title and raw_url:
            self.results.append({"title": title, "url": raw_url})
        self.current_attrs = None
        self.current_parts = []


class ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.source = ""
        self.published_at = ""
        self.description = ""
        self.capture_field: str | None = None
        self.detail_depth = 0
        self.capture_block = False
        self.current_block: list[str] = []
        self.blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        class_name = attr_map.get("class", "") or ""

        if tag == "meta" and attr_map.get("name") == "description":
            self.description = clean_text(attr_map.get("content", "") or "")

        if tag == "h3" and "detail-title" in class_name:
            self.capture_field = "title"
        elif tag == "span" and "source" in class_name:
            self.capture_field = "source"
        elif tag == "span" and "time" in class_name:
            self.capture_field = "published_at"

        if tag == "div" and "detail-con" in class_name:
            self.detail_depth = 1
            return

        if self.detail_depth:
            if tag == "div":
                self.detail_depth += 1
            if tag in {"p", "li", "h4"}:
                self.capture_block = True
                self.current_block = []
            if tag == "br" and self.capture_block:
                self.current_block.append("\n")

    def handle_data(self, data: str) -> None:
        text = clean_text(data)
        if not text:
            return

        if self.capture_field == "title":
            self.title += text
        elif self.capture_field == "source":
            self.source += text
        elif self.capture_field == "published_at":
            self.published_at += text
        elif self.capture_block:
            self.current_block.append(text)

    def handle_endtag(self, tag: str) -> None:
        if self.capture_field and tag in {"h3", "span"}:
            self.capture_field = None

        if self.detail_depth:
            if tag in {"p", "li", "h4"} and self.capture_block:
                block = clean_text(" ".join(self.current_block))
                if block and block not in {"暂无", "新华财经"}:
                    self.blocks.append(block)
                self.capture_block = False
                self.current_block = []
            elif tag == "div":
                self.detail_depth -= 1


@dataclass
class FetchedArticle:
    title: str
    url: str
    source: str
    published_at: str
    description: str
    content_blocks: list[str]

    @property
    def body(self) -> str:
        sections = [self.description] + self.content_blocks
        return "\n".join(part for part in sections if part)

    @property
    def published_date(self) -> str:
        match = re.search(r"\d{4}-\d{2}-\d{2}", self.published_at)
        if match:
            return match.group(0)
        return ""


class CnfinFetcher:
    def build_search_url(self, query: str = BRIEFING_QUERY) -> str:
        encoded = urllib.parse.quote(query)
        return f"{SEARCH_BASE_URL}?q={encoded}&t=news&isAdvance=&p=0&timetype="

    def search_briefing_links(self, query: str = BRIEFING_QUERY) -> tuple[list[dict[str, str]], list[str]]:
        logs: list[str] = []
        url = self.build_search_url(query)
        try:
            html = fetch_html(url)
        except Exception as exc:
            logs.append(f"搜索请求失败：{exc}")
            return [], logs
        parser = SearchResultParser()
        parser.feed(html)
        results = [item for item in parser.results if BRIEFING_QUERY in item["title"]]
        logs.append(f"通过新华财经搜索页检索关键词“{query}”。")
        logs.append(f"搜索结果中命中 {len(results)} 条“新华财经早报”候选。")
        return results, logs

    def collect_candidate_links(self, *, max_links: int = 80) -> list[dict[str, str]]:
        candidates: list[dict[str, str]] = []
        seen: set[str] = set()

        for url in CANDIDATE_LISTING_URLS:
            try:
                html = fetch_html(url)
            except Exception:
                continue
            parser = LinkCollector()
            parser.feed(html)
            for link in parser.links:
                normalized = normalize_cnfin_url(link["href"])
                if "cnfin.com" not in normalized or normalized in seen:
                    continue
                seen.add(normalized)
                candidates.append({"url": normalized, "title": clean_text(link["title"])})
                if len(candidates) >= max_links:
                    return candidates
        return candidates

    def fetch_article(self, url: str) -> FetchedArticle:
        normalized = normalize_cnfin_url(url)
        try:
            html = fetch_html(normalized)
        except Exception as exc:
            raise RuntimeError(f"抓取文章失败（{normalized}）：{exc}") from exc
        parser = ArticleParser()
        parser.feed(html)
        title = parser.title
        if not title:
            title_match = re.search(r"<title>(.*?)\s*-\s*中国金融信息网</title>", html, re.S)
            if title_match:
                title = clean_text(title_match.group(1))
        return FetchedArticle(
            title=title or "未识别标题",
            url=normalized,
            source=parser.source or "新华财经",
            published_at=parser.published_at,
            description=parser.description,
            content_blocks=parser.blocks,
        )

    def find_latest_briefing(self, *, max_checks: int = 25) -> tuple[FetchedArticle | None, list[str]]:
        results, logs = self.search_briefing_links()
        today = datetime.now().strftime("%Y-%m-%d")
        for candidate in results[:max_checks]:
            try:
                article = self.fetch_article(candidate["url"])
            except Exception as exc:
                logs.append(f"抓取失败：{candidate['url']} ({exc})")
                continue
            article_date = article.published_date or extract_date_from_url(article.url)
            if "新华财经早报" in article.title and article_date == today:
                logs.append(f"识别到当日早报正文：{article.title}")
                return article, logs
            if "新华财经早报" in article.title:
                logs.append(f"发现非当日早报：{article.title}（日期 {article_date or '未知'}）")

        logs.append("搜索结果中未发现当日《新华财经早报》，视为今日尚未公布。")
        return None, logs

    def fetch_latest_headlines(self, *, max_articles: int = 8, max_checks: int = 18) -> tuple[list[FetchedArticle], list[str]]:
        logs: list[str] = []
        articles: list[FetchedArticle] = []
        candidates = sorted(
            self.collect_candidate_links(),
            key=lambda item: item["url"],
            reverse=True,
        )
        candidate_dates = [extract_date_from_url(item["url"]) for item in candidates]
        recent_dates = [day for day in candidate_dates if day]
        top_dates = sorted(set(recent_dates), reverse=True)[:2]
        if top_dates:
            logs.append(f"聚焦最近日期：{'、'.join(top_dates)}")

        for candidate in candidates[:max_checks]:
            candidate_date = extract_date_from_url(candidate["url"])
            if top_dates and candidate_date and candidate_date not in top_dates:
                continue
            try:
                article = self.fetch_article(candidate["url"])
            except Exception as exc:
                logs.append(f"抓取失败：{candidate['url']} ({exc})")
                continue

            if len(article.body) < 30:
                continue
            if "新华财经早报" in article.title:
                continue
            if article.title == "未识别标题":
                continue

            articles.append(article)
            logs.append(f"收录新闻：{article.title}")
            if len(articles) >= max_articles:
                break
        return articles, logs

    def build_daily_source_snapshot(self, *, manual_url: str | None = None) -> dict[str, Any]:
        fetch_logs: list[str] = []
        if manual_url:
            try:
                article = self.fetch_article(manual_url)
            except RuntimeError as exc:
                fetch_logs.append(str(exc))
                return {
                    "snapshot_date": datetime.now().strftime("%Y-%m-%d"),
                    "source_mode": "manual-briefing-failed",
                    "source_label": "手动链接抓取失败",
                    "source_url": manual_url,
                    "items": [],
                    "briefing_status": "missing",
                    "fetch_log": fetch_logs,
                }
            fetch_logs.append("使用手动指定链接生成日报。")
            snapshot_date = article.published_date or datetime.now().strftime("%Y-%m-%d")
            return {
                "snapshot_date": snapshot_date,
                "source_mode": "manual-briefing",
                "source_label": article.title,
                "source_url": article.url,
                "items": [
                    {
                        "title": article.title,
                        "url": article.url,
                        "published_at": article.published_at,
                        "source": article.source,
                        "text": article.body,
                    }
                ],
                "fetch_log": fetch_logs,
            }

        briefing, briefing_logs = self.find_latest_briefing()
        fetch_logs.extend(briefing_logs)
        if briefing is not None:
            items = []
            for block in briefing.content_blocks:
                normalized = block.lstrip("•").strip()
                if not normalized:
                    continue
                items.append(
                    {
                        "title": normalized[:36],
                        "url": briefing.url,
                        "published_at": briefing.published_at,
                        "source": briefing.source,
                        "text": normalized,
                    }
                )

            return {
                "snapshot_date": briefing.published_date or datetime.now().strftime("%Y-%m-%d"),
                "source_mode": "xinhua-daily-briefing",
                "source_label": briefing.title,
                "source_url": briefing.url,
                "items": items,
                "fetch_log": fetch_logs,
            }
        return {
            "snapshot_date": datetime.now().strftime("%Y-%m-%d"),
            "source_mode": "xinhua-daily-briefing-missing",
            "source_label": "当日财经早报尚未公布",
            "source_url": self.build_search_url(),
            "items": [],
            "briefing_status": "missing",
            "fetch_log": fetch_logs,
        }
