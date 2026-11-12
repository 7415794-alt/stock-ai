"""수급(외국인·기관·개인 순매수) 지표.

시험(509종목, 최근 1년)에서 AI 학습에 수급을 넣었더니 1·5·20일 뒤 모두 적중률이 올랐음
(확신 높은 예측 상위 10%에서는 +1~4%p). 그래서 통합 모델(pool.py)의 학습 재료로 씀.

- 하루치 수급은 장 마감 뒤에야 확정되므로, 날짜 d의 지표에는 'd 전날까지'의 수급만 씀
  → 장중에 검색해도, 매일 오후 5시 학습에서도 똑같은 기준 (미래 정보 없음)
- 순매수량은 20일 평균 거래량으로 나눠서, 큰 종목·작은 종목을 같은 잣대로 비교
"""
from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd
import requests

TREND = "https://m.stock.naver.com/api/stock/{code}/trend?pageSize=60&bizdate={biz}"
START = "20210101"  # 처음 받을 때 여기까지 거슬러 올라감
RAW_COLS = ["frg", "org", "ind", "hold", "vol"]
FLOW_COLS = ([f"fl_{k}{w}" for k in ("frg", "org", "ind") for w in (1, 5, 20)]
             + ["fl_hold_d5", "fl_hold_d20", "fl_frg_streak", "fl_both5"])
CACHE_URL = os.environ.get("RANK_URL", "https://raw.githubusercontent.com/7415794-alt/stock-ai/data/") + "flow/"
SHARDS = 10  # 저장 파일을 종목코드 끝자리로 10개로 나눔 (한 파일이 너무 커지지 않게)
_UA = {"User-Agent": "Mozilla/5.0"}


def _num(s) -> float:
    try:
        return float(str(s).replace(",", "").replace("+", "").replace("%", ""))
    except Exception:
        return np.nan


def fetch(code: str, since: str = START, max_pages: int = 30) -> pd.DataFrame:
    """네이버에서 날짜별 수급 받기 (최신부터 60일씩 거슬러 올라감). 열: date, frg, org, ind, hold, vol"""
    rows, biz = [], pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d")
    for _ in range(max_pages):
        d = None
        for i in range(3):
            try:
                r = requests.get(TREND.format(code=code, biz=biz), headers=_UA, timeout=15)
                d = r.json()
                break
            except Exception:
                time.sleep(2 * (i + 1))
        if not d:
            break
        rows += [(x["bizdate"], _num(x.get("foreignerPureBuyQuant")), _num(x.get("organPureBuyQuant")),
                  _num(x.get("individualPureBuyQuant")), _num(x.get("foreignerHoldRatio")),
                  _num(x.get("accumulatedTradingVolume"))) for x in d]
        last = d[-1]["bizdate"]
        if last <= since or len(d) < 60:
            break
        biz = (pd.Timestamp(last) - pd.Timedelta(days=1)).strftime("%Y%m%d")
    df = pd.DataFrame(rows, columns=["date"] + RAW_COLS)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date"]).drop_duplicates("date", keep="first").sort_values("date")
    df[RAW_COLS] = df[RAW_COLS].astype("float32")
    return df.reset_index(drop=True)


def update(code: str, old: pd.DataFrame | None) -> pd.DataFrame:
    """저장해 둔 기록이 있으면 최근 60일만 받아 이어 붙이고, 없거나 오래됐으면 처음부터 받음."""
    if old is not None and len(old) and old["date"].max() >= pd.Timestamp.now() - pd.Timedelta(days=60):
        new = fetch(code, max_pages=1)
        if not len(new):
            return old
        return clean(pd.concat([new, old[old["date"] < new["date"].min()]], ignore_index=True))
    return fetch(code)


def shard_of(code: str) -> int:
    return ord(str(code)[-1]) % SHARDS


def load_cache(url: str = CACHE_URL) -> dict[str, pd.DataFrame]:
    """data 브랜치에 저장된 전 종목 수급 기록 (종목코드 → 표). 없으면 빈 dict."""
    def one(k):
        try:
            r = requests.get(f"{url}flow-{k}.parquet", timeout=120)
            r.raise_for_status()
            return pd.read_parquet(io.BytesIO(r.content))
        except Exception:
            return None
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(5) as ex:
        parts = [t for t in ex.map(one, range(SHARDS)) if t is not None]
    if not parts:
        return {}
    t = pd.concat(parts, ignore_index=True)
    t["code"] = t["code"].astype(str)
    return {c: g.drop(columns="code").reset_index(drop=True) for c, g in t.groupby("code", sort=False)}


def features(raw: pd.DataFrame) -> pd.DataFrame:
    """수급 원자료 → 지표 (그 날짜까지의 수급으로 계산, 날짜가 인덱스)."""
    g = raw.sort_values("date").set_index("date")
    v20 = g["vol"].rolling(20, min_periods=10).mean().replace(0, np.nan)
    o = pd.DataFrame(index=g.index)
    for k in ("frg", "org", "ind"):
        for w in (1, 5, 20):
            o[f"fl_{k}{w}"] = g[k].rolling(w, min_periods=1).sum() / v20 / w
    o["fl_hold_d5"] = g["hold"].diff(5)
    o["fl_hold_d20"] = g["hold"].diff(20)
    sgn = np.sign(g["frg"])
    o["fl_frg_streak"] = (sgn.groupby((sgn != sgn.shift()).cumsum()).cumcount() + 1) * sgn  # 외국인 연속 순매수(+)/순매도(-) 일수
    o["fl_both5"] = (((o["fl_frg5"] > 0) & (o["fl_org5"] > 0)).astype(float)
                     - ((o["fl_frg5"] < 0) & (o["fl_org5"] < 0)).astype(float))  # 외국인·기관 동시 매수(+1)/매도(-1)
    return o.astype("float32")


def attach(dates: pd.DatetimeIndex, raw: pd.DataFrame | None) -> pd.DataFrame:
    """주가 날짜마다 '그 전날까지'의 수급 지표를 붙임. 수급 기록이 없거나 7일 넘게 비면 빈 값(NaN)."""
    out = pd.DataFrame(np.nan, index=range(len(dates)), columns=FLOW_COLS, dtype="float32")
    if raw is None or not len(raw):
        return out
    f = features(raw).reset_index().rename(columns={"date": "fdate"}).sort_values("fdate")
    left = pd.DataFrame({"d": pd.DatetimeIndex(dates).astype("datetime64[ns]"), "i": range(len(dates))}).sort_values("d")
    f["fdate"] = f["fdate"].astype("datetime64[ns]")
    m = pd.merge_asof(left, f, left_on="d", right_on="fdate", allow_exact_matches=False,
                      tolerance=pd.Timedelta(days=7), direction="backward").sort_values("i")
    out[FLOW_COLS] = m[FLOW_COLS].to_numpy(dtype="float32")
    return out


_LIVE: dict[str, tuple[float, pd.DataFrame]] = {}


def live(code: str, ttl: int = 1800) -> pd.DataFrame | None:
    """앱에서 종목 검색할 때: 최근 60일 수급 (30분 동안 재사용)."""
    hit = _LIVE.get(code)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        raw = fetch(code, max_pages=1)
    except Exception:
        raw = None
    if raw is not None and len(raw):
        _LIVE[code] = (time.time(), raw)
        return raw
    return hit[1] if hit else None
