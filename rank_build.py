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
