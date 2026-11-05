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


# 지표 50가지 표에서 '기본적 지표' 현재 값으로 쓰는 항목
FIN_KEYS = ("salesIncreasingRate", "operatingProfitIncreasingRate", "roe", "per", "pbr", "dividendRate",
            "sales", "operatingProfit", "netIncome", "propertyTotal", "debtTotal", "eps")


def krx_status() -> dict[str, dict]:
    """코스피·코스닥 전 종목의 거래정지·관리종목·투자주의/경고/위험 상태 (네이버 증권, 1초 정도).
    {종목코드: {"halted": bool, "managed": bool, "alert": 0~3}}  (alert 1=투자주의 2=투자경고 3=투자위험)"""
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    out = {}
    for page in range(10):
        d = s.get("https://stock.naver.com/api/domestic/market/stock/default",
                  params={"tradeType": "KRX", "marketType": "ALL", "orderType": "marketSum",
                          "startIdx": page, "pageSize": 1000}, timeout=15).json()
        for r in d:
            out[r["itemcode"]] = {"halted": r.get("tradeStopYn") == "Y",
                                  "managed": str(r.get("manageStatusGb")) in ("1", "2", "3"),  # 9 = 상장폐지 절차 등(거래정지로 따로 표시)
                                  "alert": int(r.get("marketAlertType") or 0),
                                  "fin": {k: r.get(k) for k in FIN_KEYS}}
        if len(d) < 1000:
            break
    return out


def market_clock() -> dict:
    """한국 주식시장이 지금 열렸는지: 장 시작 전 · 장중 · 장 마감 · 휴장."""
    now = pd.Timestamp.now(tz="Asia/Seoul")
    if now.weekday() >= 5:
        return {"state": "closed", "label": "💤 주말 휴장", "note": "순위·확률은 금요일 장 마감 기준이에요"}
    traded_today = True
    try:
        r = requests.get("https://m.stock.naver.com/api/index/KOSPI/basic", timeout=5,
                         headers={"User-Agent": "Mozilla/5.0"}).json()
        traded_today = pd.Timestamp(r["localTradedAt"]).date() == now.date()
    except Exception:
        pass
    hm = now.hour * 100 + now.minute
    if hm < 900:
        return {"state": "pre", "label": "⏰ 장 시작 전", "note": "오전 9시에 열려요"}
    if not traded_today:
        return {"state": "closed", "label": "💤 오늘은 휴장일", "note": "순위·확률은 지난 거래일 기준이에요"}
    if hm < 1530:
        return {"state": "open", "label": "🟢 장 열림", "note": "오후 3시 30분에 마감해요 · 가격은 약 20분 늦어요"}
    return {"state": "closed", "label": "🔴 장 마감", "note": "AI 순위는 매일 오후 6시쯤 새로 계산돼요"}


def halted(code: str) -> bool:
    """한국 종목이 지금 거래정지인지 (네이버 증권). 확인이 안 되면 False."""
    try:
        r = requests.get(f"https://m.stock.naver.com/api/stock/{code[:6]}/basic",
                         timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        return (r.json().get("tradeStopType") or {}).get("name") == "HALTED"
    except Exception:
        return False


def _naver(code: str) -> dict:
    r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/integration",
                     timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


def _naver_info(raw: dict) -> dict:
    """네이버 증권 (한국 종목): 이미 '1,175조', '9.02배'처럼 보기 좋게 만들어진 값."""
    vals = {x.get("code"): x.get("value") for x in raw.get("totalInfos", [])}
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


def _yahoo_info(ticker: str, i: dict) -> dict:
    """야후 파이낸스 (해외 종목)."""
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


def _num(s) -> float:
    try:
        return float(str(s).replace(",", "").replace("+", ""))
    except ValueError:
        return float("nan")


def _flows(raw: dict) -> pd.DataFrame | None:
    """최근 거래일별 외국인·기관·개인 순매수 수량 (한국 종목)."""
    rows = [{"날짜": pd.to_datetime(x["bizdate"]), "외국인": _num(x.get("foreignerPureBuyQuant")),
             "기관": _num(x.get("organPureBuyQuant")), "개인": _num(x.get("individualPureBuyQuant")),
             "종가": _num(x.get("closePrice"))} for x in raw.get("dealTrendInfos") or [] if x.get("bizdate")]
    return pd.DataFrame(rows).sort_values("날짜") if rows else None


def _schedule(ticker: str, raw: dict | None) -> list[tuple[str, pd.Timestamp, str]]:
    """실적 발표·배당 일정: [(이름, 날짜, 덧붙임)]."""
    out = []
    ir = (raw or {}).get("irScheduleInfo")
    if ir and ir.get("irScheduleDate"):
        out.append((f"📢 {ir.get('title') or '실적 발표'}", pd.Timestamp(ir["irScheduleDate"]), "회사 공지"))
    try:
        cal = yf.Ticker(ticker).calendar or {}
    except Exception:
        cal = {}
    dates = cal.get("Earnings Date") or []
    if dates and not out:
        out.append(("📢 실적 발표", pd.Timestamp(dates[0]), "예상일"))
    if cal.get("Ex-Dividend Date"):
        out.append(("💰 배당락일", pd.Timestamp(cal["Ex-Dividend Date"]), "이 날 전에 갖고 있어야 배당"))
    if cal.get("Dividend Date"):
        out.append(("💵 배당 지급일", pd.Timestamp(cal["Dividend Date"]), ""))
    return sorted(out, key=lambda x: x[1])


def _opinion(score: float) -> str:
    """투자의견 점수(5=강력 매수 … 1=매도)를 말로."""
    for cut, word in ((4.5, "강력 매수"), (3.5, "매수"), (2.5, "중립"), (1.5, "매도")):
        if score >= cut:
            return word
    return "강력 매도"


def _consensus(raw: dict | None, yinfo: dict | None) -> dict | None:
    """증권사 평균 목표주가·투자의견. 네이버(한국) 우선, 없으면 야후."""
    c = (raw or {}).get("consensusInfo") or {}
    target = _num(c.get("priceTargetMean"))
    if target == target and target > 0:
        score = _num(c.get("recommMean"))  # 네이버: 5 강력 매수 … 1 강력 매도
        date = c.get("createDate") or ""
        return {"target": target, "opinion": _opinion(score) if score == score else "-",
                "n": None, "date": date.replace("-", ".")[:10], "src": "네이버 증권"}
    i = yinfo or {}
    target = i.get("targetMeanPrice")
    if isinstance(target, (int, float)) and target > 0:
        score = i.get("recommendationMean")  # 야후: 1 강력 매수 … 5 매도 → 뒤집기
        return {"target": float(target), "opinion": _opinion(6 - score) if isinstance(score, (int, float)) else "-",
                "n": i.get("numberOfAnalystOpinions"), "date": "", "src": "야후 파이낸스"}
    return None


def company(ticker: str) -> dict:
    """기본 정보·외국인/기관 매매·일정을 한 번에 (실패한 항목은 None)."""
    korean = ticker.endswith((".KS", ".KQ"))
    raw = None
    if korean:
        try:
            raw = _naver(ticker[:6])
        except Exception:
            raw = None
    yinfo = None
    if not korean or not (raw or {}).get("consensusInfo"):  # 해외 종목 또는 네이버에 목표주가가 없을 때만
        try:
            yinfo = yf.Ticker(ticker).info or {}
        except Exception:
            yinfo = None
    try:
        info = _naver_info(raw) if korean else _yahoo_info(ticker, yinfo or {})
    except Exception:
        info = None
    if korean and raw is None:
        info = None
    try:
        cons = _consensus(raw, yinfo)
    except Exception:
        cons = None
    return {"info": info, "flow": _flows(raw) if raw else None, "schedule": _schedule(ticker, raw),
            "consensus": cons}


def stock_info(ticker: str) -> dict | None:
    return company(ticker)["info"]


def ai_mood(rank: pd.DataFrame) -> pd.DataFrame:
    """순위 데이터로 '상승 확률 50% 이상인 종목 비율'을 기간·시장별로."""
    g = rank.assign(up=rank["prob_up"] >= 0.5)
    return g.groupby(["horizon", "market"])["up"].mean().unstack()
