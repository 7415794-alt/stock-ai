"""주식 AI 상승/하락 확률 분석 앱. 실행: streamlit run app.py"""
import plotly.graph_objects as go
import streamlit as st

from predictor import FEATURE_LABELS, predict, resolve_ticker

st.set_page_config(page_title="주식 AI 확률 분석", page_icon="📈", layout="centered")
st.title("📈 주식 AI 상승/하락 확률 분석")
st.caption("코스피·코스닥 전 종목과 해외 주식을 검색할 수 있어요. "
           "종목명 일부(삼성), 종목코드(005930), 티커(AAPL) 모두 됩니다.")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(q):
    return resolve_ticker(q)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_predict(ticker, name, horizon):
    return predict(ticker, name, horizon)


col_q, col_h = st.columns([3, 1])
query = col_q.text_input("종목 검색", placeholder="예: 삼성전자, NVDA, 005930")
horizon = col_h.selectbox("예측 기간", [1, 5, 20], index=1, format_func=lambda d: f"{d}거래일 후")

if not query:
    st.stop()

with st.spinner("종목 검색 중..."):
    candidates = cached_search(query)
if not candidates:
    st.warning("검색 결과가 없습니다.")
    st.stop()

if len(candidates) > 1:
    choice = st.selectbox("종목 선택", candidates, format_func=lambda c: f"{c[1]} — {c[0]}")
else:
    choice = candidates[0]
ticker, name = choice

try:
    with st.spinner(f"{ticker} 10년치 데이터 학습 및 검증 중..."):
        p = cached_predict(ticker, name, horizon)
except Exception as e:
    st.error(f"분석 실패: {e}")
    st.stop()

# ── 결과 ───────────────────────────────────────────
last = p.history["Close"].iloc[-1]
st.subheader(f"{p.name} ({p.ticker})")
st.write(f"현재가 **{last:,.2f} {p.currency}** · 기준일 {p.history.index[-1]:%Y-%m-%d}")

up = p.prob_up * 100
c1, c2 = st.columns(2)
c1.metric(f"{p.horizon}거래일 후 상승 확률", f"{up:.1f}%",
          f"{(p.prob_up - p.base_rate) * 100:+.1f}%p vs 평소")
c2.metric(f"{p.horizon}거래일 후 하락 확률", f"{100 - up:.1f}%")

gauge = go.Figure(go.Indicator(
    mode="gauge+number", value=up, number={"suffix": "%"},
    title={"text": "상승 확률"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#d62728" if up >= 50 else "#1f77b4"},
        "steps": [{"range": [0, 45], "color": "#e3eefa"},
                  {"range": [45, 55], "color": "#f0f0f0"},
                  {"range": [55, 100], "color": "#fbe5e5"}],
        "threshold": {"line": {"color": "black", "width": 2}, "value": p.base_rate * 100},
    },
))
gauge.update_layout(height=260, margin=dict(t=50, b=10, l=30, r=30))
st.plotly_chart(gauge, use_container_width=True)
st.caption(f"검은 선 = 이 종목이 과거 {p.horizon}거래일 동안 오른 비율({p.base_rate * 100:.1f}%). "
           "이 선에서 멀수록 모델이 평소와 다르게 보고 있다는 뜻입니다.")

# ── 신뢰도 ──────────────────────────────────────────
st.markdown("### 🔍 이 예측을 얼마나 믿을 수 있나요?")
edge = (p.cv_accuracy - p.cv_baseline_accuracy) * 100
m1, m2, m3 = st.columns(3)
m1.metric("과거 검증 정확도", f"{p.cv_accuracy * 100:.1f}%")
m2.metric("단순 기준선 정확도", f"{p.cv_baseline_accuracy * 100:.1f}%", f"{edge:+.1f}%p", delta_color="off")
m3.metric("AUC", f"{p.cv_auc:.3f}")
if edge < 2 or p.cv_auc < 0.53:
    st.warning("⚠️ 이 종목에서는 모델이 '항상 같은 방향을 찍는' 단순 기준선보다 뚜렷하게 낫지 않았습니다. "
               "확률 수치를 참고 정도로만 보세요.")
else:
    st.info("과거 워크포워드 검증에서 기준선보다 약간 나은 성과를 보였습니다. 그래도 미래를 보장하지는 않습니다.")
st.caption(f"검증 방식: {p.n_samples:,}개 거래일을 시간순으로 나눠, 과거로 학습하고 그 다음 구간을 맞혀본 결과 "
           "(워크포워드). AUC 0.5 = 동전 던지기 수준.")

# ── 차트 ────────────────────────────────────────────
st.markdown("### 📊 최근 1년 주가")
h = p.history.iloc[-252:]
fig = go.Figure(go.Candlestick(x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
                               increasing_line_color="#d62728", decreasing_line_color="#1f77b4", name="가격"))
for n, color in ((20, "orange"), (60, "green")):
    fig.add_scatter(x=h.index, y=p.history["Close"].rolling(n).mean().iloc[-252:],
                    line=dict(width=1, color=color), name=f"{n}일선")
fig.update_layout(height=380, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# ── 지표 ────────────────────────────────────────────
st.markdown("### 🧮 현재 지표와 모델이 중요하게 본 요인")
f = p.latest_features
pct_keys = {"ret_1", "ret_5", "ret_20", "sma5_ratio", "sma20_ratio", "sma60_ratio", "vol_20", "dist_52w_high"}
rows = []
for k in p.importances.index[::-1]:
    v = f[k]
    val = f"{v * 100:+.2f}%" if k in pct_keys else f"{v:.2f}" if k != "macd_hist" else f"{v * 100:+.3f}%"
    rows.append({"지표": FEATURE_LABELS[k], "현재 값": val, "중요도": f"{p.importances[k] * 100:.1f}%"})
st.dataframe(rows, hide_index=True, use_container_width=True)

st.divider()
st.caption("⚠️ 본 앱은 과거 가격 패턴을 학습한 통계 모델의 참고용 결과이며 투자 권유가 아닙니다. "
           "주가는 뉴스·실적·거시경제 등 모델이 보지 못하는 요인에 크게 좌우되며, "
           "투자 판단과 그 결과의 책임은 본인에게 있습니다.")
