"""종목 관련 최신 뉴스(구글 뉴스)를 받아서, 제목에 든 호재·악재 단어로 분위기를 점수 매김."""
from __future__ import annotations

import math
import re
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

MAX_ADJ = 0.06  # 뉴스로 확률을 바꾸는 최대 폭 (±6%p)

# 긴 표현부터 먼저 찾음 ('적자 축소'는 '적자'보다 먼저 → 호재)
KO_POS = [
    "흑자전환", "흑자 전환", "적자 축소", "적자축소", "손실 축소", "우려 해소", "목표가 상향", "목표주가 상향",
    "실적 개선", "어닝 서프라이즈", "사상 최대", "사상최대", "역대 최대", "역대최대", "최대 실적", "신고가",
    "호실적", "호조", "호재", "급등", "강세", "상승", "반등", "돌파", "껑충", "훨훨", "랠리", "질주",
    "수주", "계약 체결", "공급계약", "승인", "허가", "출시", "흑자", "성장", "증가", "확대", "개선",
    "상향", "순매수", "매수", "↑", "▲", "수혜", "기대", "선정", "1위", "자사주", "배당", "특허", "협력", "투자 유치",
]
KO_NEG = [
    "적자전환", "적자 전환", "흑자 축소", "목표가 하향", "목표주가 하향", "어닝 쇼크", "어닝쇼크", "실적 부진",
    "상장폐지", "거래정지", "관리종목", "유상증자", "압수수색", "신저가", "급락", "폭락", "추락", "약세",
    "하락", "털썩", "뚝", "급감", "감소", "부진", "둔화", "하향", "순매도", "매도", "악재", "우려", "리스크",
    "적자", "손실", "소송", "제재", "횡령", "배임", "감자", "리콜", "철회", "취소", "실패", "쇼크", "경고",
    "위기", "파업", "↓", "▼", "불확실", "과징금", "벌금", "해지", "반락", "고발", "기소",
]
EN_POS = [
    "beats", "beat", "surge", "soar", "jump", "rally", "record high", "upgrade", "raises", "raised", "gain",
    "rises", "rise", "strong", "growth", "profit", "approval", "approved", "wins", "deal", "partnership",
    "buy rating", "outperform", "higher", "boost", "bullish",
]
EN_NEG = [
    "misses", "miss", "plunge", "slump", "tumble", "sink", "falls", "fall", "drop", "downgrade", "cuts",
    "loss", "lawsuit", "probe", "recall", "weak", "decline", "sell rating", "warning", "fraud", "layoffs",
    "lower", "bearish", "underperform", "investigation", "crash",
]


def _score_title(title: str, korean: bool) -> tuple[int, list[str], list[str]]:
    """제목 하나 → (+1 호재 / -1 악재 / 0 중립, 찾은 호재 단어, 찾은 악재 단어)."""
    text = title if korean else title.lower()
    pos_words, neg_words = (KO_POS, KO_NEG) if korean else (EN_POS, EN_NEG)
    found = {"+": [], "-": []}
    # 두 목록을 합쳐 긴 단어부터 찾고, 찾은 부분은 지워서 짧은 단어로 또 세지 않게 함
    for w, sign in sorted([(w, "+") for w in pos_words] + [(w, "-") for w in neg_words], key=lambda x: -len(x[0])):
        pattern = re.escape(w) if korean else r"\b" + re.escape(w) + r"\b"
        if re.search(pattern, text):
            found[sign].append(w)
            text = re.sub(pattern, " ", text)
    p, n = len(found["+"]), len(found["-"])
    return (1 if p > n else -1 if n > p else 0), found["+"], found["-"]


def fetch_news(name: str, ticker: str, limit: int = 10) -> list[dict]:
    """구글 뉴스에서 최근 7일 뉴스를 최신순으로."""
    korean = ticker.endswith((".KS", ".KQ"))
    name = re.sub(r"\s*\(.*\)\s*$", "", name).strip() or ticker
    if korean:
        q, lang = f'"{name}" when:7d', "hl=ko&gl=KR&ceid=KR:ko"
    else:
        q, lang = f"{ticker.split('.')[0]} stock when:7d", "hl=en-US&gl=US&ceid=US:en"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&{lang}"
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    # 증권사가 '다른 회사'를 분석한 리포트 기사는 뺌. 예) 키움증권 "휴젤, 어닝 쇼크" / ... - 키움증권
    report = re.compile(rf'^\s*{re.escape(name)}\s*["“”\'‘’]|[-–]\s*{re.escape(name)}\s*$') if korean else None
    items = []
    for it in ET.fromstring(r.content).iter("item"):
        source = it.findtext("source") or ""
        title = it.findtext("title") or ""
        if source and title.endswith(" - " + source):  # 제목 끝의 ' - 언론사' 떼기
            title = title[: -len(source) - 3]
        if report and report.search(title):
            continue
        # 본문에만 이름이 잠깐 나오는 엉뚱한 기사는 뺌: 제목에 종목 이름이 있어야 함 (띄어쓰기 무시)
        if korean and name.replace(" ", "") not in title.replace(" ", ""):
            continue
        try:
            when = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate"))).tz_convert("Asia/Seoul")
        except Exception:
            continue
        sign, pw, nw = _score_title(title, korean)
        items.append({"title": title, "link": it.findtext("link"), "source": source, "when": when,
                      "sign": sign, "pos": pw, "neg": nw})
    items.sort(key=lambda x: x["when"], reverse=True)
    # 같은 제목(여러 언론사가 그대로 옮긴 기사)은 한 번만
    seen, out = set(), []
    for x in items:
        key = re.sub(r"\W", "", x["title"])[:30]
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out[:limit]


def news_score(items: list[dict]) -> float:
    """-1(악재 일색) ~ +1(호재 일색). 최근 뉴스일수록 더 크게 반영 (3일마다 약 1/e로 줄어듦)."""
    if not items:
        return 0.0
    now = pd.Timestamp.now(tz="Asia/Seoul")
    w = [math.exp(-max((now - x["when"]).total_seconds(), 0) / 86400 / 3) for x in items]
    return sum(wi * x["sign"] for wi, x in zip(w, items)) / sum(w)


def adjust(prob_up: float, score: float) -> float:
    """차트 AI 확률에 뉴스 점수를 더함 (최대 ±6%p)."""
    return min(max(prob_up + MAX_ADJ * score, 0.01), 0.99)
