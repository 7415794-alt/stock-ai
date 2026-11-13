"""새 앱(웹앱)이 읽을 파일 만들기 — 매일 오후 5시 계산(rank_build.py) 끝에 실행.

앱은 서버에 계산을 시키지 않고, 여기서 미리 만든 파일만 읽어서 바로 보여 줌.
  out/app/index.json      전 종목 목록(검색·순위용): 이름, 확률, 적중률, 종가
  out/app/s/{코드}.json    종목 하나: 차트, 확률(1·5·20일), AI 성적·적중 기록, 지표 값, 수급, 테마, 뉴스, 기업 정보
  out/app/themes.json     테마 목록과 대장주, 테마 소식
  out/app/ind.json        지표 50가지 설명 + AI 중요도
  out/app/meta.json       만든 시각·기준일·버전
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 20)
CHART_DAYS = 260
TRACK_N = 60
HORIZON_FADE = {1: 1.0, 5: 0.6, 20: 0.3}  # 테마 소식 효과는 기간이 길수록 약해짐 (앱과 같은 값)
NEWS_MAX = 0.06  # 뉴스로 확률을 바꾸는 최대 폭 (news.MAX_ADJ와 같음)


def _r(x, n=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, n)


def _px(x):
    """가격: 정수면 정수로 (파일 크기 줄이기)."""
    v = _r(x, 2)
    return int(v) if v is not None and v == int(v) else v


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


# ───────────── 조각(part) 단계: 종목마다 차트·뉴스·기업 정보 ─────────────

def part_payload(ticker: str, name: str, market: str, df: pd.DataFrame, stt: dict) -> dict:
    """주가 표(df)와 상태(stt)로 앱용 기본 정보. 뉴스는 실패해도 빈 목록."""
    d = df.iloc[-CHART_DAYS:]
    idx = pd.DatetimeIndex(d.index)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    chart = {
        "d": [int(x.strftime("%y%m%d")) for x in idx],
        "o": [_px(x) for x in d["Open"]], "h": [_px(x) for x in d["High"]],
        "l": [_px(x) for x in d["Low"]], "c": [_px(x) for x in d["Close"]],
        "v": [int(x) if x == x else 0 for x in d["Volume"]],
    }
    news = []
    try:
        from news import fetch_news
        for x in fetch_news(name, ticker, limit=5):
            news.append({"t": x["title"], "u": x["link"], "src": x["source"],
                         "w": x["when"].strftime("%Y-%m-%d %H:%M"), "s": int(x["sign"])})
    except Exception:
        pass
    return {"code": ticker[:6], "ticker": ticker, "name": name, "market": market, "chart": chart,
            "managed": int(bool(stt.get("managed"))), "alert": int(stt.get("alert") or 0),
            "fin": {k: v for k, v in (stt.get("fin") or {}).items() if v not in (None, "")}, "news": news}


# ───────────── 합치기(merge) 단계 ─────────────

def _news_adj(news: list[dict], now: pd.Timestamp) -> float:
    """최근 뉴스 호재·악재 점수 → 확률 보정 (최근일수록 크게, 최대 ±6%p). news.news_score/adjust와 같은 방식."""
    if not news:
        return 0.0
    w, s = [], []
    for x in news:
        age = max((now - pd.Timestamp(x["w"]).tz_localize("Asia/Seoul")).total_seconds(), 0) / 86400
        w.append(math.exp(-age / 3))
        s.append(x["s"])
    return NEWS_MAX * sum(a * b for a, b in zip(w, s)) / sum(w) if sum(w) else 0.0


def export(out: str | Path, info: pd.DataFrame, res: pd.DataFrame, latest: pd.DataFrame,
           parts_dir: str | Path = "parts") -> None:
    """info: 종목 정보(part csv), res: 통합 모델 결과(확률·성적), latest: 종목별 가장 최근 지표 한 줄."""
    from indicators import indicators_for

    out = Path(out) / "app"
    now = pd.Timestamp.now(tz="Asia/Seoul")
    payload = {}
    for f in sorted(Path(parts_dir).rglob("app-*.json")):
        payload.update(json.loads(f.read_text(encoding="utf-8")))
    print(f"앱 파일: 종목 {len(payload)}개", flush=True)

    # 테마: 종목 → [테마번호, 테마이름, 역할], 테마 보정값
    th, tstats, tnews = None, None, None
    try:
        th = pd.read_csv("out/themes.csv", dtype={"code": str})
        tstats = pd.read_csv("out/theme_stats.csv", dtype={"up_mover": str, "down_mover": str}).set_index("theme_no")
        for k in ("up_mover", "down_mover"):
            if k in tstats:
                tstats[k] = tstats[k].map(lambda v: str(v).split(".")[0].zfill(6) if isinstance(v, str) else v)
        tnews = pd.read_csv("out/theme_news.csv")
    except Exception as e:
        print("테마 파일 없음:", e)
    roles_of, theme_adj = {}, {}
    if th is not None:
        for r in th.itertuples():
            roles = "".join(k for k, c in (("L", "leader"), ("M", "mover"), ("F", "flow")) if getattr(r, c, 0) == 1)
            roles_of.setdefault(r.code, []).append([int(r.theme_no), r.theme, roles])
        if tstats is not None and "adj" in tstats:
            for code, lst in roles_of.items():
                a = tstats["adj"].reindex([x[0] for x in lst]).dropna()
                a = a[a != 0]
                theme_adj[code] = float(a.mean()) if len(a) else 0.0

    # 적중 기록: 저장된 과거 예측 + 실제 결과(차트 종가로 계산)
    tracks = {}
    for h in HORIZONS:
        try:
            t = pd.read_parquet(f"out/pool/track_{h}.parquet")
            t["ticker"] = t["ticker"].astype(str)
            tracks[h] = {k: g for k, g in t.groupby("ticker", sort=False)}
        except Exception:
            tracks[h] = {}

    res = res.set_index(["ticker", "horizon"])
    latest = latest.set_index("ticker")
    rows, dates = [], []
    for tk, p in payload.items():
        code = p["code"]
        ch = p["chart"]
        closes = pd.Series(ch["c"], index=pd.to_datetime([str(x) for x in ch["d"]], format="%y%m%d"))
        nadj = _news_adj(p.get("news", []), now)
        tadj = theme_adj.get(code, 0.0)
        prob, stat, track = {}, {}, {}
        for h in HORIZONS:
            if (tk, h) not in res.index:
                continue
            s = res.loc[(tk, h)]
            base = float(s["prob_up"])
            ta = max(min(tadj * HORIZON_FADE[h], 0.02), -0.02)
            final = min(max(base + nadj + ta, 0.01), 0.99) if nadj or ta else base
            prob[h] = {"ai": _r(base), "news": _r(nadj), "theme": _r(ta), "p": _r(final)}
            stat[h] = {"acc": _r(s.get("acc")), "base": _r(s.get("base")), "auc": _r(s.get("auc")),
                       "hits": int(s["hits"]) if s.get("hits") == s.get("hits") else 0,
                       "n": int(s["n_recent"]) if s.get("n_recent") == s.get("n_recent") else 0}
            g = tracks[h].get(tk)
            if g is not None and len(g):
                g = g.sort_values("date").tail(TRACK_N + h)
                fut = closes.shift(-h) / closes - 1
                rr = []
                for d, pr in zip(pd.to_datetime(g["date"]), g["prob"]):
                    r = fut.get(d, np.nan)
                    rr.append([int(d.strftime("%y%m%d")), _r(pr, 3), _r(r, 4)])
                track[h] = rr[-TRACK_N:]
        if not prob:
            continue
        ind = {}
        if tk in latest.index:
            lr = latest.loc[tk]
            ind = {c: _r(v, 5) for c, v in lr.items() if c not in ("date",) and _r(v, 5) is not None}
        c = ch["c"]
        chg = (c[-1] / c[-2] - 1) if len(c) >= 2 and c[-2] else None
        doc = {**{k: p[k] for k in ("code", "ticker", "name", "market", "managed", "alert", "fin", "news")},
               "date": str(closes.index[-1].date()), "close": c[-1], "chg": _r(chg),
               "chart": ch, "prob": prob, "stat": stat, "track": track, "ind": ind,
               "themes": roles_of.get(code, [])}
        _dump(out / "s" / f"{code}.json", doc)
        dates.append(doc["date"])
        rows.append([code, p["name"], p["market"], c[-1], _r(chg),
                     *[(prob.get(h) or {}).get("p") for h in HORIZONS],
                     *[(lambda s: _r(s["hits"] / s["n"], 3) if s and s["n"] else None)(stat.get(h)) for h in HORIZONS],
                     p["managed"], p["alert"]])
    # 거래정지 종목도 검색은 되게
    for r in info[info.get("halted", 0) == 1].drop_duplicates("ticker").itertuples():
        rows.append([r.ticker[:6], r.name, r.market, None, None, None, None, None, None, None, None, 0, 0, 1])
    cols = ["code", "name", "market", "close", "chg", "p1", "p5", "p20", "hr1", "hr5", "hr20", "managed", "alert"]
    _dump(out / "index.json", {"cols": cols + ["halted"], "rows": [r + [0] if len(r) == len(cols) else r for r in rows]})

    # 테마
    if th is not None:
        themes = []
        for no, g in th.groupby("theme_no", sort=False):
            st = tstats.loc[no] if tstats is not None and no in tstats.index else None
            items = [] if tnews is None else [
                {"t": x.title, "u": x.link, "src": x.source, "w": x.when, "s": int(x.sign)}
                for x in tnews[tnews["theme_no"] == no].itertuples()]
            stat = {}
            if st is not None:
                for k in ("up_eff", "up_strength", "up_days", "up_mover", "up_mover_avg",
                          "down_eff", "down_strength", "down_days", "down_mover", "down_mover_avg", "adj"):
                    v = st.get(k)
                    if v == v and v is not None:
                        stat[k] = v if isinstance(v, str) else _r(v)
            themes.append({
                "no": int(no), "name": g["theme"].iloc[0], "chg": _r(g["theme_chg"].iloc[0], 2),
                "desc": (g["desc"].dropna().iloc[0] if g["desc"].notna().any() else ""),
                "stocks": [[r.code, r.name, _r(r.chg, 2), _r(r.mcap, 0),
                            "".join(k for k, c in (("L", "leader"), ("M", "mover"), ("F", "flow")) if getattr(r, c, 0) == 1),
                            _r(getattr(r, "hot_avg", None)), _r(getattr(r, "tv5", None), 2),
                            (r.why if isinstance(r.why, str) else "")[:90]] for r in g.itertuples()],
                "news": items, "stat": stat})
        themes.sort(key=lambda x: -(x["chg"] or -999))
        _dump(out / "themes.json", themes)

    # 지표 설명 + 중요도
    try:
        pm = json.loads(Path("out/pool/meta.json").read_text(encoding="utf-8"))
    except Exception:
        pm = {"horizons": {}}
    ind_doc = {"table": {str(h): [[r, n, t, d, cols] for r, n, t, d, cols in indicators_for(h)] for h in HORIZONS},
               "imp": {h: pm["horizons"].get(h, {}).get("importances", {}) for h in map(str, HORIZONS)},
               "oos": {h: {k: pm["horizons"].get(h, {}).get(k) for k in ("oos_acc", "oos_auc", "rows")}
                       for h in map(str, HORIZONS)}}
    _dump(out / "ind.json", ind_doc)
    _dump(out / "meta.json", {"built_at": now.strftime("%Y-%m-%d %H:%M"),
                              "date": pd.Series(dates).mode().iloc[0] if dates else "", "stocks": len(dates), "v": 1})
    print(f"앱 파일 완료: 종목 {len(rows)}개", flush=True)
