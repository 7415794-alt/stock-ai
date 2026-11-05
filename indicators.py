"""1일 뒤 예측용 지표 50가지 (기술적 30 + 기본적 20).

기술적 지표는 매일의 주가·거래량으로 계산해서 AI가 학습합니다.
기본적 지표는 과거 날짜별 기록을 무료로 구할 수 없어 학습에는 못 쓰고, 현재 값만 참고로 보여 줍니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (순위, 지표, 유형, 내용, 이 지표로 계산한 특성 이름들 — 비어 있으면 학습 미반영)
INDICATORS = [
    (1, "상대강도(RS)", "기술적", "시장·업종 대비 초과수익률", ["rs_5", "rs_20"]),
    (2, "가격 모멘텀", "기술적", "최근 1·3·5·20일 수익률", ["mom_1", "mom_3", "mom_5", "mom_20"]),
    (3, "이동평균선 추세", "기술적", "5·10·20·60일선 방향", ["ma5_slope", "ma10_slope", "ma20_slope", "ma60_slope"]),
    (4, "실적 서프라이즈", "기본적", "실제 실적과 시장 예상치 차이", []),
    (5, "시장·업종 모멘텀", "기술적", "시장 및 업종 수익률", ["mkt_mom_1", "mkt_mom_5"]),
    (6, "전고점·전저점 돌파", "기술적", "주요 가격대 돌파", ["high60_gap", "low60_gap"]),
    (7, "거래대금 증가율", "기술적", "평균 대비 거래대금 변화", ["tv_ratio"]),
    (8, "상대 거래량(RVOL)", "기술적", "평균 대비 거래량 변화", ["rvol"]),
    (9, "거래량 동반 돌파", "기술적", "가격 돌파 시 거래량 증가", ["vol_breakout"]),
    (10, "고점·저점 구조", "기술적", "HH·HL·LH·LL 추세 구조", ["hhll"]),
    (11, "이익 전망치 상향", "기본적", "향후 EPS 전망 변화", []),
    (12, "매출 성장률", "기본적", "전년 대비 매출 증가", []),
    (13, "이동평균선 이격도", "기술적", "주가와 이동평균선 거리", ["ma5_gap", "ma20_gap", "ma60_gap"]),
    (14, "ADX", "기술적", "추세 강도", ["adx", "di_diff"]),
    (15, "MACD", "기술적", "추세·모멘텀 변화", ["macd_h"]),
    (16, "거래량별 매물대", "기술적", "가격별 거래 집중도", ["poc_gap"]),
    (17, "실현 변동성", "기술적", "수익률 변동 폭", ["rvol_5", "rvol_20"]),
    (18, "ATR", "기술적", "평균 가격 진폭", ["atr_pct"]),
    (19, "OBV", "기술적", "누적 거래량 흐름", ["obv_slope"]),
    (20, "RSI", "기술적", "상승·하락 모멘텀", ["rsi"]),
    (21, "ROC", "기술적", "일정 기간 가격 변화율", ["roc_10"]),
    (22, "영업이익 성장률", "기본적", "전년 대비 영업이익 증가", []),
    (23, "볼린저밴드", "기술적", "밴드 위치와 폭", ["bb_pos", "bb_width"]),
    (24, "VWAP 이격도", "기술적", "현재가와 VWAP 거리", ["vwap_gap"]),
    (25, "EPS 성장률", "기본적", "주당순이익 증가율", []),
    (26, "MFI", "기술적", "가격·거래량 결합 모멘텀", ["mfi"]),
    (27, "캔들 몸통·꼬리", "기술적", "봉 내부 가격 구조", ["body", "upper_wick", "lower_wick"]),
    (28, "영업이익률 변화", "기본적", "수익성 개선 여부", []),
    (29, "골든·데드크로스", "기술적", "이동평균선 교차", ["cross_5_20", "cross_20_60"]),
    (30, "ROE", "기본적", "자기자본이익률", []),
    (31, "스토캐스틱", "기술적", "최근 가격 범위 내 위치", ["stoch_k", "stoch_d"]),
    (32, "PER", "기본적", "주가수익비율", []),
    (33, "일목균형표", "기술적", "복합 추세·지지저항", ["ichi_tk", "ichi_cloud"]),
    (34, "이중바닥·이중천장", "기술적", "가격 반전 패턴", ["double_pat"]),
    (35, "PBR", "기본적", "주가순자산비율", []),
    (36, "FCF 수익률", "기본적", "시가총액 대비 잉여현금흐름", []),
    (37, "영업현금흐름 성장", "기본적", "영업활동 현금 창출 증가", []),
    (38, "ROIC", "기본적", "투하자본이익률", []),
    (39, "순이익률", "기본적", "매출 대비 순이익", []),
    (40, "부채비율", "기본적", "재무 레버리지", []),
    (41, "피벗 포인트", "기술적", "계산된 지지·저항", ["pivot_gap"]),
    (42, "순차입금/EBITDA", "기본적", "차입 부담과 상환 능력", []),
    (43, "피보나치 되돌림", "기술적", "비율 기반 가격대", ["fib_pos"]),
    (44, "헤드앤숄더", "기술적", "추세 반전 패턴", ["hs_pat"]),
    (45, "유동비율", "기본적", "단기 지급 능력", []),
    (46, "배당수익률", "기본적", "주가 대비 배당금", []),
    (47, "자사주 매입·소각", "기본적", "주주환원 규모와 변화", []),
    (48, "EV/EBITDA", "기본적", "기업가치 대비 EBITDA", []),
    (49, "유상증자·주식 희석", "기본적", "발행주식 수 증가와 희석", []),
    (50, "엘리어트 파동", "기술적", "파동 구조 해석", []),
]
D1_COLS = [c for *_, cols in INDICATORS for c in cols]
# 20일 뒤 예측의 지표 순위 (1·5일 뒤는 위 INDICATORS 순서)
ORDER_20 = [
    "상대강도(RS)",
    "이익 전망치 상향",
    "시장·업종 모멘텀",
    "이동평균선 추세",
    "가격 모멘텀",
    "매출 성장률",
    "영업이익 성장률",
    "실적 서프라이즈",
    "EPS 성장률",
    "고점·저점 구조",
    "영업이익률 변화",
    "ROE",
    "PER",
    "전고점·전저점 돌파",
    "ADX",
    "이동평균선 이격도",
    "거래량 동반 돌파",
    "거래대금 증가율",
    "MACD",
    "상대 거래량(RVOL)",
    "PBR",
    "FCF 수익률",
    "영업현금흐름 성장",
    "ROIC",
    "순이익률",
    "실현 변동성",
    "거래량별 매물대",
    "ROC",
    "ATR",
    "OBV",
    "RSI",
    "골든·데드크로스",
    "부채비율",
    "순차입금/EBITDA",
    "볼린저밴드",
    "일목균형표",
    "유동비율",
    "MFI",
    "이중바닥·이중천장",
    "스토캐스틱",
    "배당수익률",
    "VWAP 이격도",
    "캔들 몸통·꼬리",
    "자사주 매입·소각",
    "EV/EBITDA",
    "유상증자·주식 희석",
    "피벗 포인트",
    "피보나치 되돌림",
    "헤드앤숄더",
    "엘리어트 파동",
]


TOP_N_20 = 10  # 20일 뒤는 순위 상위 기술적 지표 10개만 학습 (시험에서 전체를 쓸 때보다 조금 더 정확했음)


def cols_for(horizon: int) -> list[str]:
    """예측 기간별로 AI가 학습하는 특성. 1·5일 뒤 = 전부(AI가 종목마다 중요도를 정함), 20일 뒤 = 순위 상위 10개."""
    if horizon == 20:
        tech = [x for x in indicators_for(20) if x[4]][:TOP_N_20]
        return [c for x in tech for c in x[4]]
    return list(D1_COLS)


def indicators_for(horizon: int) -> list[tuple]:
    """예측 기간에 맞는 순위로 다시 번호를 매긴 지표 표."""
    if horizon != 20:
        return INDICATORS
    by_name = {x[1]: x for x in INDICATORS}
    return [(i, *by_name[n][1:]) for i, n in enumerate(ORDER_20, 1)]


D1_LABELS = {c: f"{name} ({c})" for _, name, _, _, cols in INDICATORS for c in cols}
NOT_USED_REASON = {
    "엘리어트 파동": "파동을 세는 방법이 사람마다 달라 정해진 계산식이 없어요",
    "기본적": "과거 날짜별 재무 기록을 무료로 구할 수 없어 학습에는 못 넣고 현재 값만 보여 드려요",
}


def _rolling_poc(close: np.ndarray, volume: np.ndarray, window: int = 60, bins: int = 12) -> np.ndarray:
    """최근 60일 중 거래가 가장 많이 몰린 가격대(매물대 중심)."""
    out = np.full(len(close), np.nan)
    for i in range(window - 1, len(close)):
        c, v = close[i - window + 1:i + 1], volume[i - window + 1:i + 1]
        lo, hi = c.min(), c.max()
        if hi <= lo or v.sum() <= 0:
            continue
        idx = np.minimum(((c - lo) / (hi - lo) * bins).astype(int), bins - 1)
        hist = np.bincount(idx, weights=v, minlength=bins)
        b = int(hist.argmax())
        out[i] = lo + (b + 0.5) * (hi - lo) / bins
    return out


def build_1d(df: pd.DataFrame, mkt: pd.Series | None = None) -> pd.DataFrame:
    o, h, l, c, v = (df[k].astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    f = pd.DataFrame(index=df.index)
    ret = c.pct_change()

    # 1 가격 모멘텀
    for n in (1, 3, 5, 20):
        f[f"mom_{n}"] = c.pct_change(n)
    # 2·3 거래대금·거래량
    tv = c * v
    f["tv_ratio"] = tv / tv.rolling(20).mean().replace(0, np.nan)
    f["rvol"] = v / v.rolling(20).mean().replace(0, np.nan)
    # 4 VWAP (20일)
    tp = (h + l + c) / 3
    vwap = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    f["vwap_gap"] = c / vwap - 1
    # 5 거래량 동반 돌파: 20일 고점을 넘으면서 거래량이 늘었는지
    prev_high20 = h.rolling(20).max().shift(1)
    f["vol_breakout"] = (c > prev_high20).astype(float) * f["rvol"]
    # 6·7 시장 모멘텀·상대강도
    if mkt is not None:
        days = df.index.tz_localize(None).normalize() if df.index.tz is not None else df.index.normalize()
        m = mkt.reindex(mkt.index.union(days)).ffill().reindex(days)
        m.index = df.index
        f["mkt_mom_1"] = m.pct_change(1)
        f["mkt_mom_5"] = m.pct_change(5)
        f["rs_5"] = c.pct_change(5) - m.pct_change(5)
        f["rs_20"] = c.pct_change(20) - m.pct_change(20)
    else:
        f["mkt_mom_1"] = f["mkt_mom_5"] = f["rs_5"] = f["rs_20"] = 0.0
    # 8 전고점·전저점 (60일)
    f["high60_gap"] = c / h.rolling(60).max().shift(1) - 1
    f["low60_gap"] = c / l.rolling(60).min().shift(1) - 1
    # 9 실현 변동성
    f["rvol_5"] = ret.rolling(5).std() * np.sqrt(252)
    f["rvol_20"] = ret.rolling(20).std() * np.sqrt(252)
    # 10 ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["atr_pct"] = atr / c
    # 11·12 이동평균 이격도·방향
    ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60)}
    for n in (5, 20, 60):
        f[f"ma{n}_gap"] = c / ma[n] - 1
    for n in (5, 10, 20, 60):
        f[f"ma{n}_slope"] = ma[n].pct_change(5)
    # 13 고점·저점 구조: 최근 10일 고가·저가가 그 전 10일보다 높으면 +, 낮으면 -
    h10, l10 = h.rolling(10).max(), l.rolling(10).min()
    f["hhll"] = np.sign(h10 - h10.shift(10)) + np.sign(l10 - l10.shift(10))
    # 14 RSI
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # 15 캔들
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    # 16 볼린저
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - (mid - 2 * sd)) / (4 * sd)
    f["bb_width"] = 4 * sd / mid
    # 17 매물대
    f["poc_gap"] = c / pd.Series(_rolling_poc(c.to_numpy(), v.to_numpy()), index=df.index) - 1
    # 18 MFI
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    f["mfi"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    # 19 OBV 기울기 (20일 동안 늘어난 OBV / 20일 거래량)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_slope"] = (obv - obv.shift(20)) / v.rolling(20).sum().replace(0, np.nan)
    # 20 ROC
    f["roc_10"] = c.pct_change(10)
    # 21 MACD
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_h"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    # 22 ADX
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    f["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    f["di_diff"] = pdi - ndi
    # 23 스토캐스틱
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    f["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()
    # 24 피벗 포인트 (전날 고가·저가·종가)
    pivot = ((h + l + c) / 3).shift(1)
    f["pivot_gap"] = c / pivot - 1
    # 25 이중바닥(+1)·이중천장(-1): 최근 20일 저점(고점)이 그 전 20일 저점(고점)과 3% 안으로 비슷
    low_a, low_b = l.rolling(20).min(), l.rolling(20).min().shift(20)
    high_a, high_b = h.rolling(20).max(), h.rolling(20).max().shift(20)
    f["double_pat"] = ((low_a / low_b - 1).abs() < 0.03).astype(float) - ((high_a / high_b - 1).abs() < 0.03).astype(float)
    # 26 골든(+1)·데드(-1) 크로스: 최근 5일 안에 교차했는지
    for a, b in ((5, 20), (20, 60)):
        s = np.sign(ma[a] - c.rolling(b).mean() if b not in ma else ma[a] - ma[b])
        f[f"cross_{a}_{b}"] = (s.diff() / 2).rolling(5).sum()
    # 27 피보나치: 60일 범위 안에서 현재가 위치 (0 = 저점, 1 = 고점)
    lo60, hi60 = l.rolling(60).min(), h.rolling(60).max()
    f["fib_pos"] = (c - lo60) / (hi60 - lo60).replace(0, np.nan)
    # 28 일목균형표
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    f["ichi_tk"] = (tenkan - kijun) / c
    f["ichi_cloud"] = (c - pd.concat([span_a, span_b], axis=1).max(axis=1)) / c
    # 29 헤드앤숄더(간단판): 60일을 3등분해 가운데 고점이 양쪽보다 3% 이상 높고 양쪽이 비슷하면 -1 (역H&S는 +1)
    p1, p2, p3 = h.rolling(20).max().shift(40), h.rolling(20).max().shift(20), h.rolling(20).max()
    q1, q2, q3 = l.rolling(20).min().shift(40), l.rolling(20).min().shift(20), l.rolling(20).min()
    hs = (p2 > p1 * 1.03) & (p2 > p3 * 1.03) & ((p1 / p3 - 1).abs() < 0.05)
    ihs = (q2 < q1 * 0.97) & (q2 < q3 * 0.97) & ((q1 / q3 - 1).abs() < 0.05)
    f["hs_pat"] = ihs.astype(float) - hs.astype(float)
    return f.replace([np.inf, -np.inf], np.nan)


def fundamentals_now(row: dict | None) -> dict:
    """네이버 증권 전 종목 표의 현재 재무 값 → {지표 이름: 보기 좋은 글자}."""
    if not row:
        return {}

    def num(k):
        try:
            return float(str(row.get(k)).replace(",", ""))
        except (TypeError, ValueError):
            return None

    out = {}
    pairs = [("매출 성장률", "salesIncreasingRate", "{:+.1f}%"), ("영업이익 성장률", "operatingProfitIncreasingRate", "{:+.1f}%"),
             ("ROE", "roe", "{:.1f}%"), ("PER", "per", "{:.1f}배"), ("PBR", "pbr", "{:.2f}배"),
             ("배당수익률", "dividendRate", "{:.2f}%")]
    for label, k, fmt in pairs:
        x = num(k)
        if x is not None:
            out[label] = fmt.format(x)
    sales, op, ni = num("sales"), num("operatingProfit"), num("netIncome")
    if sales:
        if ni is not None:
            out["순이익률"] = f"{ni / sales * 100:.1f}%"
    prop, debt = num("propertyTotal"), num("debtTotal")
    if prop and debt is not None and prop - debt > 0:
        out["부채비율"] = f"{debt / (prop - debt) * 100:.0f}%"
    eps = num("eps")
    if eps is not None:
        out["EPS 성장률"] = f"EPS {eps:,.0f}원 (성장률은 자료 없음)"
    return out
