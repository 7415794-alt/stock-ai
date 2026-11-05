"""시장 분위기(주요 지수)와 종목 기본 정보(시가총액·PER·배당 등)."""
from __future__ import annotations

import pandas as pd
import requests
import yfinance as yf

INDICES = [("^KS11", "코스피"), ("^KQ11", "코스닥"), ("^GSPC", "S&P 500"), ("^IXIC", "나스닥"),
           ("KRW=X", "원/달러 환율")]


def market_snapshot() -> list[dict]:
    """주요 지수의 현재 값·전일 대비, 최근 3개월 흐름."""
    out = []
    for sym, name in INDICES:
        try:
            h = yf.Ticker(sym).history(period="3mo")["Close"].dropna()
            if len(h) < 2:
                continue
            out.append({"name": name, "last": float(h.iloc[-1]), "chg": float(h.iloc[-1] / h.iloc[-2] - 1),
                        "date": h.index[-1], "hist": h})
        except Exception:
            continue
    return out


def _naver_info(code: str) -> dict:
    """네이버 증권 (한국 종목): 이미 '1,175조', '9.02배'처럼 보기 좋게 만들어진 값."""
    r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/integration",
                     timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    vals = {x.get("code"): x.get("value") for x in r.json().get("totalInfos", [])}
    pick = [("marketValue", "시가총액"), ("per", "PER"), ("pbr", "PBR"), ("dividendYieldRatio", "배당수익률"),
            ("highPriceOf52Weeks", "52주 최고"), ("lowPriceOf52Weeks", "52주 최저"),
            ("foreignRate", "외국인 보유"), ("eps", "EPS")]
    info = {}
    for k, label in pick:
        v = vals.get(k)
        if k == "marketValue" and v:
            v += "원"
        elif k in ("highPriceOf52Weeks", "lowPriceOf52Weeks") and v:
            v += "원"
        info[label] = v if v and v != "N/A" else "-"
    return info


def _yahoo_info(ticker: str) -> dict:
    """야후 파이낸스 (해외 종목)."""
    i = yf.Ticker(ticker).info or {}
    cur = i.get("currency") or ""

    def num(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) else "-"

    cap = i.get("marketCap")
    if isinstance(cap, (int, float)):
        cap_txt = f"{cap / 1e12:,.2f}T {cur}" if cap >= 1e12 else f"{cap / 1e9:,.1f}B {cur}"
    else:
        cap_txt = "-"
    return {
        "시가총액": cap_txt,
        "PER": num(i.get("trailingPE"), "{:.2f}배"),
        "PBR": num(i.get("priceToBook"), "{:.2f}배"),
        "배당수익률": num(i.get("dividendYield"), "{:.2f}%"),
        "52주 최고": num(i.get("fiftyTwoWeekHigh"), "{:,.2f} " + cur),
        "52주 최저": num(i.get("fiftyTwoWeekLow"), "{:,.2f} " + cur),
        "업종": i.get("industry") or "-",
        "EPS": num(i.get("trailingEps"), "{:,.2f} " + cur),
    }


def stock_info(ticker: str) -> dict | None:
    try:
        if ticker.endswith((".KS", ".KQ")):
            return _naver_info(ticker[:6])
        return _yahoo_info(ticker)
    except Exception:
        return None


def ai_mood(rank: pd.DataFrame) -> pd.DataFrame:
    """순위 데이터로 '상승 확률 50% 이상인 종목 비율'을 기간·시장별로."""
    g = rank.assign(up=rank["prob_up"] >= 0.5)
    return g.groupby(["horizon", "market"])["up"].mean().unstack()
