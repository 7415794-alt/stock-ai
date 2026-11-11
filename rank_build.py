"""코스피·코스닥 전 종목을 계산해서 순위용 파일을 만듦 (GitHub Actions에서 매일 자동 실행).

사용법:
  python rank_build.py part 0 20     # 전체를 20조각으로 나눈 것 중 0번 조각: 주가를 받아 지표 계산
                                     #   → parts/part-0.csv (종목 정보), parts/feat-0.parquet (날짜별 지표)
  python rank_build.py merge         # 전 종목 지표를 모아 통합 모델 학습(pool.py) → out/pool/,
                                     #   out/rankings.csv, out/meta.json 생성
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

import pool  # noqa: E402
from market import halted, krx_status  # noqa: E402
from predictor import load_history, load_index, load_krx, market_index  # noqa: E402

HORIZONS = (1, 5, 20)
RECENT = 60  # 적중률은 앱과 똑같이 '최근 60번' 기준


def fetch(ticker: str, tries: int = 3) -> pd.DataFrame | None:
    for i in range(tries):
        try:
            df = load_history(ticker)
            if len(df):
                return df
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None


def build_part(part: int, parts: int) -> None:
    krx = load_krx().sort_values("ticker").reset_index(drop=True)
    mine = krx[krx.index % parts == part]
    rows, feats, start = [], [], time.time()
    try:
        status = krx_status()  # 거래정지·관리종목·투자주의 (한 번에)
    except Exception:
        status = {}
    for n, r in enumerate(mine.itertuples(), 1):
        stt = status.get(r.ticker[:6]) or {"halted": halted(r.ticker), "managed": False, "alert": 0}
        if stt["halted"]:  # 거래정지 종목은 계산하지 않고 '거래정지'로만 표시
            for h in HORIZONS:
                rows.append({"ticker": r.ticker, "name": r.name, "market": r.market, "horizon": h, "halted": 1})
            continue
        df = fetch(r.ticker)
        # 데이터가 1년 남짓 안 되거나, 최근 거래가 없는(거래정지 등) 종목은 뺌
        if df is None or len(df) < 300 or df.index[-1] < pd.Timestamp.now(tz=df.index.tz) - timedelta(days=10):
            continue
        try:
            feats.append(pool.stock_frame(r.ticker, df, load_index(market_index(r.ticker))))
        except Exception as e:
            print(f"skip {r.ticker}: {e}", flush=True)
            continue
        for h in HORIZONS:  # 확률·적중률은 merge에서 통합 모델로 채움
            rows.append({
                "ticker": r.ticker, "name": r.name, "market": r.market, "horizon": h,
                "close": float(df["Close"].iloc[-1]), "date": f"{df.index[-1]:%Y-%m-%d}", "halted": 0,
                "managed": int(stt["managed"]), "alert": stt["alert"],
            })
        if n % 20 == 0:
            print(f"part {part}: {n}/{len(mine)} ({time.time() - start:.0f}s)", flush=True)
    Path("parts").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(f"parts/part-{part}.csv", index=False, encoding="utf-8")
    if feats:
        pd.concat(feats, ignore_index=True).to_parquet(f"parts/feat-{part}.parquet", index=False)
    print(f"part {part} done: {len(rows)} rows, {time.time() - start:.0f}s")


def merge() -> None:
    files = sorted(Path("parts").rglob("part-*.csv"))
    frames = [pd.read_csv(f, dtype={"ticker": str}) for f in files if f.stat().st_size > 1]
    info = pd.concat(frames, ignore_index=True)
    Path("out").mkdir(exist_ok=True)

    # 전 종목 통합 학습 → 오늘 확률·최근 적중률
    feats = pd.concat([pd.read_parquet(f) for f in sorted(Path("parts").rglob("feat-*.parquet"))],
                      ignore_index=True)
    print(f"학습 데이터: {feats['ticker'].nunique()}종목, {len(feats):,}줄", flush=True)
    res = pool.train(feats, "out/pool")
    build_themes(feats)
    del feats
    res = res[["ticker", "horizon", "prob_up", "hit_rate", "hits", "n_recent"]].rename(columns={"n_recent": "n"})
    live = info[info["halted"] != 1].merge(res, on=["ticker", "horizon"], how="inner")
    df = pd.concat([live, info[info["halted"] == 1]], ignore_index=True)
    cols = ["ticker", "name", "market", "horizon", "prob_up", "hit_rate", "hits", "n", "close", "date",
            "halted", "managed", "alert"]
    df = df[[c for c in cols if c in df]]
    df.to_csv("out/rankings.csv", index=False, encoding="utf-8")
    kst = timezone(timedelta(hours=9))
    meta = {
        "built_at": datetime.now(kst).strftime("%Y-%m-%d %H:%M"),
        "date": df["date"].mode().iloc[0] if len(df) else "",
        "stocks": int(df.loc[df["halted"] != 1, "ticker"].nunique()) if len(df) else 0,
        "halted": int(df.loc[df["halted"] == 1, "ticker"].nunique()) if len(df) else 0,
        "model": "pool",
    }
    meta["prev_date"] = keep_prev(df, meta["date"])
    Path("out/meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(meta)


def build_themes(feats: pd.DataFrame) -> None:
    """테마별 종목표(out/themes.csv)와 대장주 3종류.
    - 시총 대장주(leader): 테마 안 시가총액 1위
    - 상승 대장주(mover): 최근 3개월 테마가 많이 오른 날(상위 20%) 테마 평균보다 평균 얼마나 더 올랐나 → 1위
    - 수급 대장주(flow): 최근 5거래일 거래대금 증가(평소 대비) + 매수세(OBV·MFI) → 테마 안에서 가장 강한 종목"""
    import numpy as np
    from market import theme_table
    try:
        th = theme_table()
    except Exception as e:
        print("테마 불러오기 실패:", e)
        return
    if th.empty:
        return
    th = th.drop_duplicates(["theme_no", "code"]).reset_index(drop=True)
    days = np.sort(feats["date"].unique())[-250:]
    r = feats[feats["date"].isin(days)]

    def pivot(col, n=None):
        d = r if n is None else r[r["date"].isin(days[-n:])]
        w = d.pivot_table(index="ticker", columns="date", values=col)
        w.index = w.index.str[:6]
        return w[~w.index.duplicated()]

    wide = pivot("mom_1")  # 날짜별 등락률 (1년)
    recent = wide[wide.columns[-60:]]  # 최근 약 3개월
    tv5 = pivot("tv_ratio", 5).mean(axis=1)  # 최근 5일 거래대금 / 20일 평균
    obv = pivot("obv_slope", 1).iloc[:, -1]  # 최근 20일 매수 쪽 거래량 비중
    mfi = pivot("mfi", 5).mean(axis=1)  # 자금 유입 강도 (0~100)
    th["hot_avg"], th["leader"], th["mover"], th["flow"] = np.nan, 0, 0, 0
    th["tv5"] = th["code"].map(tv5).round(2)
    for _, g in th.groupby("theme_no"):
        codes = [c for c in g["code"] if c in recent.index]
        if len(codes) >= 3:
            R = recent.loc[codes]
            avg = R.mean()
            hot_days = avg[avg >= avg.quantile(0.8)].index
            rel = R[hot_days].sub(avg[hot_days], axis=1)  # 테마 평균보다 얼마나 더 올랐나
            ha = rel.mean(axis=1)[rel.count(axis=1) >= 6]
            th.loc[g.index, "hot_avg"] = g["code"].map(ha).to_numpy()
            if len(ha):
                th.loc[g.index[(g["code"] == ha.idxmax()).to_numpy()], "mover"] = 1
            # 수급: 거래대금 증가 순위 + 매수세 순위 (거래대금이 평소보다 늘어난 종목만)
            f = pd.DataFrame({"tv": tv5.reindex(codes), "obv": obv.reindex(codes), "mfi": mfi.reindex(codes)}).dropna()
            f = f[f["tv"] > 1]
            if len(f):
                score = f["tv"].rank(pct=True) + (f["obv"].rank(pct=True) + f["mfi"].rank(pct=True)) / 2
                th.loc[g.index[(g["code"] == score.idxmax()).to_numpy()], "flow"] = 1
        if g["mcap"].notna().any():
            th.loc[g["mcap"].idxmax(), "leader"] = 1
    th["hot_avg"] = th["hot_avg"].round(4)
    th.loc[th["theme_no"].duplicated(), "desc"] = ""  # 테마 설명은 첫 줄에만 (파일 크기 줄이기)
    th.to_csv("out/themes.csv", index=False, encoding="utf-8")
    print(f"테마 {th['theme_no'].nunique()}개, 종목 {len(th)}줄", flush=True)
    try:
        theme_news_stats(th, wide)
    except Exception as e:
        print("테마 뉴스 분석 실패:", e)


def _strength(eff: float, sigma: float, n: int) -> str:
    if n < 5 or not sigma or eff != eff:
        return "자료 부족"
    z = abs(eff) / sigma
    return "강" if z >= 0.4 else "중" if z >= 0.15 else "약"


def theme_news_stats(th: pd.DataFrame, wide: pd.DataFrame) -> None:
    """테마별 최근 1년 뉴스 → (1) 최근 소식 5개(호재·악재·중립), (2) 호재·악재 소식이 나온 날 테마 종목들이
    얼마나 움직였는지(영향 강도), (3) 그때 가장 크게 움직인 종목, (4) 다음 날까지 이어진 효과로 확률 보정값.
    결과: out/theme_news.csv, out/theme_stats.csv"""
    import time as _t
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from news import news_score, theme_keyword, theme_news_history

    heads = th.drop_duplicates("theme_no")[["theme_no", "theme"]]
    days = pd.DatetimeIndex(sorted(wide.columns))
    wide = wide[days]

    def fetch(row):
        for i in range(3):
            try:
                _t.sleep(0.3)
                return row, theme_news_history(row.theme)
            except Exception:
                _t.sleep(3 * (i + 1))
        return row, []

    def trade_day(when):  # 소식이 반영되는 거래일: 장 마감(15:30) 뒤 소식은 다음 거래일
        d = pd.Timestamp(when.tz_localize(None).normalize())
        after_close = when.hour * 60 + when.minute >= 930
        i = days.searchsorted(d, side="right" if after_close else "left")
        return i if i < len(days) else None

    news_rows, stat_rows = [], []
    now = pd.Timestamp.now(tz="Asia/Seoul")
    with ThreadPoolExecutor(4) as ex:
        for row, items in ex.map(fetch, heads.itertuples()):
            for x in items[:5]:
                news_rows.append({"theme_no": row.theme_no, "title": x["title"], "link": x["link"],
                                  "source": x["source"], "when": x["when"].strftime("%Y-%m-%d %H:%M"),
                                  "sign": x["sign"]})
            codes = [c for c in th.loc[th["theme_no"] == row.theme_no, "code"] if c in wide.index]
            st_row = {"theme_no": row.theme_no, "kw": theme_keyword(row.theme), "n_news": len(items)}
            if len(codes) >= 3 and items:
                R = wide.loc[codes]
                tavg = R.mean()  # 날짜별 테마 평균 등락률
                base, sigma = float(tavg.mean()), float(tavg.std())
                st_row["sigma"] = round(sigma, 5)
                for key, sign in (("up", 1), ("down", -1)):
                    idx = sorted({i for x in items if x["sign"] == sign and (i := trade_day(x["when"])) is not None})
                    same = [days[i] for i in idx]
                    nxt = [days[i + 1] for i in idx if i + 1 < len(days)]
                    eff = float(tavg[same].mean() - base) if same else float("nan")
                    # 다음 날 효과는 표본이 적으면 0 쪽으로 줄여서 씀 (우연히 큰 값 방지)
                    eff_next = float(tavg[nxt].mean() - base) * len(nxt) / (len(nxt) + 20) if len(nxt) >= 5 else 0.0
                    st_row[f"{key}_days"] = len(same)
                    st_row[f"{key}_eff"] = round(eff, 5) if eff == eff else None
                    st_row[f"{key}_next"] = round(eff_next, 5)
                    st_row[f"{key}_strength"] = _strength(eff, sigma, len(same))
                    if len(same) >= 5:  # 그런 소식이 나온 날 가장 크게 움직인 종목
                        m = R[same].mean(axis=1)[R[same].count(axis=1) >= 5]
                        if len(m):
                            c = m.idxmax() if sign > 0 else m.idxmin()
                            st_row[f"{key}_mover"] = c
                            st_row[f"{key}_mover_avg"] = round(float(m[c]), 4)
                # 확률 보정: 최근 2일 소식 분위기 × 과거 '다음 날까지 이어진' 효과 (최대 ±2%p, 1거래일 기준)
                fresh = [x for x in items if now - x["when"] <= pd.Timedelta(days=2)]
                score = news_score(fresh) if fresh else 0.0
                eff_next = st_row["up_next"] if score > 0 else st_row["down_next"] if score < 0 else 0.0
                adj = float(np.clip(0.4 * eff_next / sigma * abs(score), -0.02, 0.02)) if sigma else 0.0
                st_row.update({"fresh_score": round(score, 3), "adj": round(adj, 4)})
            stat_rows.append(st_row)
    pd.DataFrame(news_rows).to_csv("out/theme_news.csv", index=False, encoding="utf-8")
    pd.DataFrame(stat_rows).to_csv("out/theme_stats.csv", index=False, encoding="utf-8")
    print(f"테마 뉴스: {len(news_rows)}건, 분석 {len(stat_rows)}개 테마", flush=True)


def keep_prev(df: pd.DataFrame, today: str) -> str:
    """순위 변동(▲3, 🆕)을 보여 주려고, 지금 data 브랜치에 있는 '어제' 순위를 prev_rankings.csv로 남김.
    같은 날 다시 계산한 경우에는 원래 있던 어제 파일을 그대로 유지. 어제 AI 성적표(scorecard.csv)도 함께 갱신."""
    import os
    base = f"https://raw.githubusercontent.com/{os.environ.get('GITHUB_REPOSITORY', '7415794-alt/stock-ai')}/data/"
    prev, prev_date = None, ""
    for name in ("rankings.csv", "prev_rankings.csv"):
        try:
            old = pd.read_csv(base + name, dtype={"ticker": str})
        except Exception:
            continue
        old_date = old["date"].mode().iloc[0] if len(old) else ""
        if old_date and old_date != today:
            old.to_csv("out/prev_rankings.csv", index=False, encoding="utf-8")
            prev, prev_date = old, str(old_date)
            break
    try:
        card = pd.read_csv(base + "scorecard.csv")
    except Exception:
        card = None
    if prev is not None:
        row = score_row(prev, df, prev_date, today)
        if row:
            new = pd.DataFrame([row])
            card = new if card is None else pd.concat([card[card["date"] != prev_date], new], ignore_index=True)
    if card is not None:
        card.sort_values("date").tail(250).to_csv("out/scorecard.csv", index=False, encoding="utf-8")
    return prev_date


def score_row(prev: pd.DataFrame, df: pd.DataFrame, prev_date: str, today: str) -> dict | None:
    """어제 낸 '1거래일 뒤' 예측 TOP 20이 오늘 실제로 맞았는지."""
    def live(d):  # 거래정지 종목은 채점에서 뺌
        return d[d["halted"] != 1] if "halted" in d else d

    p = live(prev)[lambda d: d["horizon"] == 1][["ticker", "prob_up", "close"]]
    now = live(df)[lambda d: d["horizon"] == 1][["ticker", "close"]].rename(columns={"close": "close_now"})
    m = p.merge(now, on="ticker").dropna(subset=["prob_up", "close", "close_now"])
    if len(m) < 100:
        return None
    m["ret"] = m["close_now"] / m["close"] - 1
    up = m.nlargest(20, "prob_up")
    down = m.nsmallest(20, "prob_up")
    return {"date": prev_date, "eval_date": today,
            "up_hit": int((up["ret"] > 0).sum()), "up_n": len(up), "up_avg": round(float(up["ret"].mean()), 5),
            "down_hit": int((down["ret"] < 0).sum()), "down_n": len(down),
            "down_avg": round(float(down["ret"].mean()), 5),
            "all_up": round(float((m["ret"] > 0).mean()), 4), "all_avg": round(float(m["ret"].mean()), 5)}


if __name__ == "__main__":
    if sys.argv[1] == "part":
        build_part(int(sys.argv[2]), int(sys.argv[3]))
    else:
        merge()
