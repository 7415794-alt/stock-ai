"""주가 데이터 수집, 기술적 지표 계산, 머신러닝 상승/하락 확률 예측."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

KRX_CSV = Path(__file__).with_name("krx_stocks.csv")  # build_stock_list.py로 생성
MAX_RESULTS = 30

# 공식 종목명과 다른 별칭, 한글로 부르는 해외 종목 (yfinance 검색은 한글을 못 찾음)
KOREAN_NAMES = {
    "하이닉스": "000660.KS", "현대자동차": "005380.KS", "네이버": "035420.KS",
    "포스코": "005490.KS",
    "애플": "AAPL", "테슬라": "TSLA", "엔비디아": "NVDA", "마이크로소프트": "MSFT",
    "구글": "GOOGL", "알파벳": "GOOGL", "아마존": "AMZN", "메타": "META",
    "넷플릭스": "NFLX", "팔란티어": "PLTR", "브로드컴": "AVGO", "amd": "AMD",
}

FEATURE_LABELS = {
    "ret_1": "1일 수익률", "ret_5": "5일 수익률", "ret_20": "20일 수익률",
    "rsi_14": "RSI(14)", "macd_hist": "MACD 히스토그램",
    "sma5_ratio": "5일선 대비 가격", "sma20_ratio": "20일선 대비 가격",
    "sma60_ratio": "60일선 대비 가격", "bb_pctb": "볼린저 %B",
    "vol_20": "20일 변동성", "volume_ratio": "거래량 비율(20일 평균 대비)",
    "dist_52w_high": "52주 고점 대비",
}


@dataclass
class Prediction:
    ticker: str
    name: str
    currency: str
    horizon: int
    prob_up: float
    base_rate: float
    cv_accuracy: float
    cv_baseline_accuracy: float
    cv_auc: float
    n_samples: int
    history: pd.DataFrame
    latest_features: pd.Series
    importances: pd.Series
    track: pd.DataFrame  # 날짜별 예측과 실제 결과 (적중 기록)


def _norm(s: str) -> str:
    return s.lower().replace(" ", "")


@lru_cache(maxsize=1)
def load_krx() -> pd.DataFrame:
    if not KRX_CSV.exists():
        return pd.DataFrame(columns=["name", "code", "ticker", "market", "key"])
    df = pd.read_csv(KRX_CSV, dtype=str, encoding="utf-8")
    return df.assign(key=df["name"].map(_norm))


def resolve_ticker(query: str) -> list[tuple[str, str]]:
    """검색어를 (티커, 표시이름) 후보 목록으로 변환."""
    q = query.strip()
    if not q:
        return []
    key = _norm(q)
    krx = load_krx()

    if re.fullmatch(r"\d{6}", q):  # 한국 종목코드
        hit = krx[krx["code"] == q]
        if not hit.empty:
            r = hit.iloc[0]
            return [(r["ticker"], f"{r['name']} ({r['market']})")]
        return [(f"{q}.KS", q), (f"{q}.KQ", q)]

    # 코스피·코스닥 전 종목: 정확히 일치 → 앞부분 일치 → 포함 순으로 정렬
    hits = krx[krx["key"].str.contains(key, regex=False)]
    if not hits.empty:
        rank = hits["key"].map(lambda k: 0 if k == key else 1 if k.startswith(key) else 2)
        hits = hits.assign(rank=rank, length=hits["key"].str.len()).sort_values(["rank", "length"])
        results = [(r.ticker, f"{r.name} ({r.market})") for r in hits.head(MAX_RESULTS).itertuples()]
        if key in KOREAN_NAMES and KOREAN_NAMES[key] not in [t for t, _ in results]:
            results.insert(0, (KOREAN_NAMES[key], q))
        return results
    if key in KOREAN_NAMES:
        return [(KOREAN_NAMES[key], q)]

    results = []
    try:
        for item in yf.Search(q, max_results=8).quotes:
            if item.get("quoteType") in ("EQUITY", "ETF"):
                name = item.get("shortname") or item.get("longname") or item["symbol"]
                results.append((item["symbol"], f"{name} ({item.get('exchange', '')})"))
    except Exception:
        pass
    if not results:
        results.append((q.upper(), q.upper()))
    return results


def load_history(ticker: str, years: int = 10) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def live_price(ticker: str) -> tuple[float, pd.Timestamp] | None:
    """가장 최근 거래 가격과 그 시각 (장 마감 후엔 종가, 무료 데이터라 한국 주식은 약 20분 늦음)."""
    try:
        tk = yf.Ticker(ticker)
        h = tk.history(period="5d")["Close"].dropna()
        m = tk.history_metadata or {}
        price = m.get("regularMarketPrice") or (float(h.iloc[-1]) if len(h) else None)
        when = m.get("regularMarketTime")
        if price:
            if isinstance(when, (int, float)):  # 버전에 따라 초 단위 숫자로 올 때
                when = pd.Timestamp(when, unit="s", tz="UTC").tz_convert(m.get("exchangeTimezoneName", "UTC"))
            when = pd.Timestamp(when) if when is not None else h.index[-1]
            return float(price), when
    except Exception:
        pass
    return None


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["Close"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    f["ret_1"] = c.pct_change(1)
    f["ret_5"] = c.pct_change(5)
    f["ret_20"] = c.pct_change(20)

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    for n in (5, 20, 60):
        f[f"sma{n}_ratio"] = c / c.rolling(n).mean() - 1

    mid, std = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (mid - 2 * std)) / (4 * std)
    f["vol_20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    f["volume_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    f["dist_52w_high"] = c / c.rolling(252, min_periods=60).max() - 1
    return f.replace([np.inf, -np.inf], np.nan)


def _model() -> RandomForestClassifier:
    # 과적합을 줄이기 위해 얕은 트리 + 큰 리프 크기
    return RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=40,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )


def _track_record(close: pd.Series, prob_up: pd.Series, horizon: int) -> pd.DataFrame:
    """각 날짜의 예측(50% 기준)이 horizon 거래일 뒤 실제로 맞았는지. 기간이 덜 지났으면 현재가로 중간 판정."""
    t = pd.DataFrame({"prob_up": prob_up})
    pos = close.index.get_indexer(t.index)
    end_pos = np.minimum(pos + horizon, len(close) - 1)
    # 결과일이 최신 날짜면 장중 가격일 수 있으므로 아직 확정이 아닌 것으로 봄
    t["done"] = pos + horizon < len(close) - 1
    t["end_date"] = close.index[end_pos]
    t["start"] = close.iloc[pos].to_numpy()  # 예측한 날 종가
    t["ret"] = close.iloc[end_pos].to_numpy() / t["start"] - 1
    t["hit"] = (t["prob_up"] >= 0.5) == (t["ret"] > 0)
    return t


def predict(ticker: str, name: str = "", horizon: int = 5) -> Prediction:
    df = load_history(ticker)
    if len(df) < 300:
        raise ValueError("과거 데이터가 너무 적어 분석할 수 없습니다 (최소 약 300거래일 필요).")

    feats = build_features(df)
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    data = feats.assign(target=(future_ret > 0).astype(float), future_ret=future_ret)
    data.loc[future_ret.isna(), "target"] = np.nan

    labeled = data.dropna()
    X, y = labeled[list(FEATURE_LABELS)], labeled["target"].astype(int)

    # 워크포워드 검증: 과거로 학습 → 바로 다음 구간을 예측 (미래 정보 누설 방지)
    n_folds, gap = 5, horizon
    fold = len(X) // (n_folds + 1)
    preds, probs, truth, baseline = [], [], [], []
    oos = []  # 각 날짜에 그 이전 데이터로만 학습한 모델이 낸 확률 (적중 기록용)
    for k in range(1, n_folds + 1):
        train_end = fold * k
        test_end = len(X) if k == n_folds else fold * (k + 1)  # 마지막 구간은 최근 날짜까지
        Xtr, ytr = X.iloc[: train_end - gap], y.iloc[: train_end - gap]
        Xte, yte = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        m = _model().fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        oos.append(pd.Series(p, index=Xte.index))
        probs.extend(p)
        preds.extend((p >= 0.5).astype(int))
        truth.extend(yte)
        baseline.extend([int(ytr.mean() > 0.5)] * len(yte))  # "항상 다수 방향" 기준선

    final = _model().fit(X, y)
    all_feats = feats[list(FEATURE_LABELS)].dropna()
    latest = all_feats.iloc[-1]
    prob_up = float(final.predict_proba(latest.to_frame().T)[0, 1])

    # 결과가 아직 안 나온 최근 날짜들: 마지막 검증 모델로 예측, 최신 날짜는 화면의 현재 예측과 동일
    pending = all_feats[all_feats.index > X.index[-1]].iloc[:-1]
    if len(pending):
        oos.append(pd.Series(m.predict_proba(pending)[:, 1], index=pending.index))
    oos.append(pd.Series([prob_up], index=[all_feats.index[-1]]))
    track = _track_record(df["Close"], pd.concat(oos), horizon)

    try:
        auc = roc_auc_score(truth, probs)
    except ValueError:
        auc = float("nan")

    info_currency = "KRW" if ticker.endswith((".KS", ".KQ")) else "USD"
    try:
        info_currency = yf.Ticker(ticker).fast_info.get("currency") or info_currency
    except Exception:
        pass

    return Prediction(
        ticker=ticker,
        name=name or ticker,
        currency=info_currency,
        horizon=horizon,
        prob_up=prob_up,
        base_rate=float(y.mean()),
        cv_accuracy=accuracy_score(truth, preds),
        cv_baseline_accuracy=accuracy_score(truth, baseline),
        cv_auc=auc,
        n_samples=len(X),
        history=df,
        latest_features=latest,
        importances=pd.Series(final.feature_importances_, index=X.columns).sort_values(),
        track=track,
    )
