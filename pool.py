"""전 종목 통합 학습 모델 (한국 종목).

종목마다 따로 배우는 대신, 코스피·코스닥 전 종목의 과거 데이터를 한 모델이 함께 배움.
시험(종목 509개, 같은 평가 종목·같은 날짜)에서 1·5·20일 뒤 모두 정확도와 AUC가 종목별 학습보다 높았음.

- 매일 GitHub Actions(rank_build.py merge)에서 train()으로 학습 → data 브랜치의 pool/ 폴더에 저장
- 앱은 저장된 모델을 받아 와서, 지금 주가로 지표를 계산해 확률을 냄 (predict_pooled)
- 적중 기록은 '최근 1년을 4구간으로 나눠, 각 구간 직전까지의 데이터로만 학습한 모델'의 예측 (미래 정보 없음)
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from flow import FLOW_COLS
from indicators import D1_COLS, cols_for

HORIZONS = (1, 5, 20)
TRACK_DAYS = 250  # 적중 기록을 남기는 최근 거래일 수 (약 1년)
TRACK_CHUNKS = 4  # 그 1년을 몇 구간으로 나눠 다시 학습할지 (분기마다)
RECENT = 60  # 적중률은 '최근 60번' 기준 (순위표·앱 동일)
POOL_URL = os.environ.get("RANK_URL", "https://raw.githubusercontent.com/7415794-alt/stock-ai/data/") + "pool/"


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=500, l2_regularization=1.0, random_state=42)


def is_kr(ticker: str) -> bool:
    return ticker.endswith((".KS", ".KQ"))


def pool_cols(h: int, available=None) -> list[str]:
    """통합 모델이 배우는 특성 = 기술 지표(cols_for) + 수급 지표(전날까지)."""
    fl = [c for c in FLOW_COLS if available is None or c in available]
    return cols_for(h) + fl


def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return (idx.tz_localize(None) if idx.tz is not None else idx).normalize()


# ───────────────────────── 학습 (GitHub Actions) ─────────────────────────

def stock_frame(ticker: str, df: pd.DataFrame, mkt: pd.Series | None,
                flow_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """한 종목의 날짜별 지표(+ 전날까지 수급) + 1·5·20일 뒤 수익률 (학습용 한 덩어리)."""
    from flow import attach
    from indicators import build_1d
    f = build_1d(df, mkt)[D1_COLS].astype("float32")
    c = df["Close"]
    dates = pd.Series(_naive(df.index), index=df.index)
    f[FLOW_COLS] = attach(pd.DatetimeIndex(dates.to_numpy()), flow_raw).to_numpy()
    for h in HORIZONS:
        f[f"ret{h}"] = (c.shift(-h) / c - 1).astype("float32")
        f[f"end{h}"] = dates.shift(-h)
    f.insert(0, "date", dates)
    f.insert(0, "ticker", ticker)
    return f.reset_index(drop=True)


def train(feats: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    """전 종목 데이터로 1·5·20일 뒤 모델을 학습하고 결과를 out_dir에 저장.
    돌려주는 표: 종목·기간별 오늘 확률과 최근 적중률 (순위표용)."""
    import joblib
    import sklearn
    from sklearn.inspection import permutation_importance

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = feats.sort_values(["ticker", "date"]).reset_index(drop=True)
    last_date = feats["date"].max()
    all_dates = np.sort(feats["date"].unique())
    window = all_dates[-TRACK_DAYS:]
    chunks = np.array_split(window, TRACK_CHUNKS)
    latest = feats.groupby("ticker").tail(1)  # 종목마다 가장 최근 날짜 (오늘 확률을 낼 줄)
    meta = {"trained_at": time.strftime("%Y-%m-%d %H:%M"), "last_date": str(pd.Timestamp(last_date).date()),
            "sklearn": sklearn.__version__, "stocks": int(feats["ticker"].nunique()), "horizons": {}}
    rank_rows, stat_rows = [], []
    for h in HORIZONS:
        t0 = time.time()
        cols = pool_cols(h, feats.columns)
        ret, end = f"ret{h}", f"end{h}"
        lab = feats[feats[ret].notna()]

        # ① 적중 기록: 구간마다 '그 구간 전에 결과가 확정된 데이터'로만 학습 → 그 구간 예측
        oos = []
        for k, ch in enumerate(chunks):
            start = ch[0]
            stop = chunks[k + 1][0] if k + 1 < len(chunks) else None
            tr = lab[lab[end] < start]
            te = feats[(feats["date"] >= start) & ((feats["date"] < stop) if stop is not None else True)]
            m = _model().fit(tr[cols], (tr[ret] > 0).astype(int))
            maj = int((tr[ret] > 0).mean() > 0.5)  # '항상 다수 방향' 기준선
            oos.append(pd.DataFrame({"ticker": te["ticker"].to_numpy(), "date": te["date"].to_numpy(),
                                     "prob": m.predict_proba(te[cols])[:, 1].astype("float32"),
                                     "ret": te[ret].to_numpy(), "end": te[end].to_numpy(), "maj": maj}))
        oos = pd.concat(oos, ignore_index=True)

        # 마지막 구간 모델로 지표 중요도 계산 (학습에 안 쓴 최근 데이터로 섞어 보기)
        done_last = oos[(oos["end"] < last_date) & (oos["date"] >= chunks[-1][0])].dropna(subset=["ret"])
        sample = feats.merge(done_last[["ticker", "date"]], on=["ticker", "date"])
        sample = sample.sample(min(len(sample), 30000), random_state=0)
        try:
            pi = permutation_importance(m, sample[cols], (sample[ret] > 0).astype(int), scoring="roc_auc",
                                        n_repeats=3, random_state=0)
            imp = pd.Series(np.clip(pi.importances_mean, 0, None), index=cols)
        except Exception:
            imp = pd.Series(1.0, index=cols)
        imp = imp / imp.sum() if imp.sum() > 0 else pd.Series(1 / len(cols), index=cols)

        # ② 오늘 확률: 결과가 확정된 모든 데이터로 학습한 최종 모델
        final = _model().fit(lab[cols], (lab[ret] > 0).astype(int))
        joblib.dump(final, out / f"model_{h}.joblib", compress=3)
        prob_now = final.predict_proba(latest[cols])[:, 1]

        # ③ 종목별 과거 검증 성적 (최근 1년)
        done = oos[(oos["end"] < last_date)].dropna(subset=["ret"]).copy()
        done["up"] = (done["ret"] > 0).astype(int)
        done["hit"] = (done["prob"] >= 0.5).astype(int) == done["up"]
        for t, g in done.groupby("ticker", sort=False):
            try:
                auc = roc_auc_score(g["up"], g["prob"]) if g["up"].nunique() > 1 else np.nan
            except ValueError:
                auc = np.nan
            r = g.iloc[-RECENT:]
            stat_rows.append({"ticker": t, "horizon": h, "acc": round(float(g["hit"].mean()), 4),
                              "base": round(float((g["up"] == g["maj"]).mean()), 4),
                              "auc": round(float(auc), 4) if auc == auc else None, "n": len(g),
                              "hits": int(r["hit"].sum()), "recent_n": len(r)})
        for t, p in zip(latest["ticker"], prob_now):
            rank_rows.append({"ticker": t, "horizon": h, "prob_up": round(float(p), 4)})

        # 앱용 적중 기록 (종목·날짜·확률만)
        oos[["ticker", "date", "prob"]].assign(ticker=lambda d: d["ticker"].astype("category")) \
            .to_parquet(out / f"track_{h}.parquet", index=False)
        meta["horizons"][str(h)] = {"rows": int(len(lab)), "cols": cols, "importances": {k: round(float(v), 5) for k, v in imp.items()},
                                    "oos_acc": round(float(done["hit"].mean()), 4),
                                    "oos_auc": round(float(roc_auc_score(done["up"], done["prob"])), 4)}
        print(f"pool h={h}: rows {len(lab):,}, {time.time() - t0:.0f}s, "
              f"acc {meta['horizons'][str(h)]['oos_acc']}, auc {meta['horizons'][str(h)]['oos_auc']}", flush=True)
    pd.DataFrame(stat_rows).to_csv(out / "stats.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    stats = pd.DataFrame(stat_rows)
    res = pd.DataFrame(rank_rows).merge(stats, on=["ticker", "horizon"], how="left")
    res["hit_rate"] = (res["hits"] / res["recent_n"]).round(4)
    return res.rename(columns={"recent_n": "n_recent"})


# ───────────────────────── 앱에서 쓰기 ─────────────────────────

_CACHE: dict[str, tuple[float, object]] = {}


def _get(name: str, loader, ttl: int = 3600):
    hit = _CACHE.get(name)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    import requests
    try:
        r = requests.get(POOL_URL + name, timeout=30)
        r.raise_for_status()
        val = loader(r.content)
    except Exception:
        return hit[1] if hit else None
    _CACHE[name] = (time.time(), val)
    return val


def _load_meta():
    return _get("meta.json", lambda b: json.loads(b.decode("utf-8")))


def _load_model(h):
    import joblib
    return _get(f"model_{h}.joblib", lambda b: joblib.load(io.BytesIO(b)))


def _load_track(h):
    def ld(b):
        t = pd.read_parquet(io.BytesIO(b))
        t["ticker"] = t["ticker"].astype(str)
        return {k: g.set_index("date")["prob"] for k, g in t.groupby("ticker", sort=False)}
    return _get(f"track_{h}.parquet", ld)


def _load_stats():
    return _get("stats.csv", lambda b: pd.read_csv(io.BytesIO(b), dtype={"ticker": str})
                .set_index(["ticker", "horizon"]))


def available(ticker: str, horizon: int) -> bool:
    if not is_kr(ticker) or horizon not in HORIZONS:
        return False
    stats = _load_stats()
    return stats is not None and (ticker, horizon) in stats.index and _load_model(horizon) is not None


def predict_pooled(ticker: str, name: str, horizon: int, df: pd.DataFrame, mkt: pd.Series | None):
    """통합 모델로 예측. 종목별 학습과 같은 모양(Prediction)으로 돌려줌."""
    from indicators import build_1d
    from predictor import Prediction, _track_record

    model, meta, stats = _load_model(horizon), _load_meta(), _load_stats()
    track = (_load_track(horizon) or {}).get(ticker)
    cols = meta["horizons"][str(horizon)].get("cols") or cols_for(horizon)  # 예전 모델(수급 없음)과도 호환
    feats = build_1d(df, mkt)
    if any(c in FLOW_COLS for c in cols):  # 이 종목의 최근 수급(전날까지)을 붙임
        import flow
        fl = flow.attach(_naive(df.index), flow.live(ticker[:6]))
        fl.index = feats.index
        feats = pd.concat([feats, fl], axis=1)
    latest = feats[cols].iloc[-1]
    prob_up = float(model.predict_proba(latest.to_frame().T.astype("float32"))[0, 1])

    # 적중 기록: 저장된 과거 예측 + 오늘(가장 최근 날짜)은 방금 계산한 확률
    day = pd.Series(df.index, index=_naive(df.index))
    day = day[~day.index.duplicated(keep="last")]
    probs = []
    if track is not None and len(track):
        tr = track[track.index.isin(day.index)]
        tr = tr[tr.index < day.index[-1]]
        probs.append(pd.Series(tr.to_numpy(dtype=float), index=pd.DatetimeIndex(day.loc[tr.index].to_numpy())))
    probs.append(pd.Series([prob_up], index=[df.index[-1]]))
    tr_all = pd.concat(probs)
    tr_all = tr_all[~tr_all.index.duplicated(keep="last")].sort_index()

    s = stats.loc[(ticker, horizon)]
    future = df["Close"].shift(-horizon) / df["Close"] - 1
    imp = pd.Series(meta["horizons"][str(horizon)]["importances"]).reindex(cols).fillna(0).sort_values()
    return Prediction(
        ticker=ticker, name=name or ticker, currency="KRW", horizon=horizon, prob_up=prob_up,
        base_rate=float((future.dropna() > 0).mean()),
        cv_accuracy=float(s["acc"]), cv_baseline_accuracy=float(s["base"]),
        cv_auc=float(s["auc"]) if s["auc"] == s["auc"] else float("nan"),
        n_samples=int(meta["horizons"][str(horizon)]["rows"]),
        history=df, latest_features=latest, importances=imp,
        track=_track_record(df["Close"], tr_all, horizon), model="pool",
    )
