"""주식 AI 상승/하락 확률 분석 앱. 실행: streamlit run app.py"""
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from predictor import FEATURE_LABELS, live_price, predict, resolve_ticker

st.set_page_config(page_title="스톡시그널", page_icon="📈", layout="centered")
# 앱 껍데기(pwa/index.html)에 "화면이 다 떴어요" 신호를 보냄. 신호가 안 오면 껍데기가 자동으로 다시 불러옴
components.html("<script>window.top.postMessage('stock-ai-ready', '*')</script>", height=0)
st.title("📈 스톡시그널")
st.markdown("**주식 AI 상승/하락 확률 분석**")
st.caption("코스피·코스닥 전 종목과 해외 주식을 검색할 수 있어요. "
           "종목명 일부(삼성), 종목코드(005930), 티커(AAPL) 모두 됩니다.")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(q):
    return resolve_ticker(q)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_predict(ticker, name, horizon, version=2):
    # predictor.py의 결과 모양을 바꾸면 version 숫자를 올려서, 예전에 저장된 결과를 쓰지 않게 함
    return predict(ticker, name, horizon)


@st.cache_data(ttl=60, show_spinner=False)  # 현재가는 1분마다 새로
def cached_live(ticker):
    return live_price(ticker)


def money(v, cur):
    return f"{v:,.0f}원" if cur == "KRW" else f"{v:,.2f} {cur}"


def footer():
    """맨 아래 안내 (어느 화면에서 멈추든 항상 보이게)."""
    st.divider()
    st.caption("⚠️ 본 앱은 과거 가격 패턴을 학습한 통계 모델의 참고용 결과이며 투자 권유가 아닙니다. "
               "주가는 뉴스·실적·거시경제 등 모델이 보지 못하는 요인에 크게 좌우되며, "
               "투자 판단과 그 결과의 책임은 본인에게 있습니다.")
    st.caption("📮 오류나 문의 사항은 [slakeknowledge@gmail.com](mailto:slakeknowledge@gmail.com) 으로 보내 주세요.")
    # 앱(pwa/index.html)이 화면 맨 아래를 잘라내므로, 마지막 내용이 가려지지 않게 여백 추가
    st.markdown('<div style="height:56px"></div>', unsafe_allow_html=True)


col_q, col_h = st.columns([3, 1])
query = col_q.text_input("종목 검색", placeholder="예: 삼성전자, NVDA, 005930")
horizon = col_h.selectbox("예측 기간", [1, 5, 20], index=1, format_func=lambda d: f"{d}거래일 후")

if not query:
    footer()
    st.stop()

with st.spinner("종목 검색 중..."):
    candidates = cached_search(query)
if not candidates:
    st.warning("검색 결과가 없습니다.")
    footer()
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
    footer()
    st.stop()

# ── 결과 ───────────────────────────────────────────
lp = cached_live(ticker)
if lp:
    last, when = lp
    when_txt = f"{when:%Y-%m-%d %H:%M} 기준 (무료 데이터라 약 20분 늦을 수 있어요)"
else:
    last, when_txt = p.history["Close"].iloc[-1], f"기준일 {p.history.index[-1]:%Y-%m-%d}"
st.subheader(f"{p.name} ({p.ticker})")
st.write(f"현재가 **{money(last, p.currency)}** · {when_txt}")

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

# ── 적중 기록 ───────────────────────────────────────
st.markdown("### 🎯 AI 적중 기록")
t = p.track.copy()
# 아직 기간이 안 끝난 예측은 방금 받은 현재가로 다시 판정
t.loc[~t["done"], "ret"] = last / t.loc[~t["done"], "start"] - 1
t["hit"] = (t["prob_up"] >= 0.5) == (t["ret"] > 0)
done = t[t["done"]].iloc[-60:]
live = t[~t["done"]].iloc[:-1]  # 최신 날짜(방금 낸 예측)는 제외
hits = int(done["hit"].sum())
a1, a2, a3 = st.columns(3)
a1.metric(f"최근 {len(done)}번 적중률", f"{done['hit'].mean() * 100:.1f}%",
          f"성공 {hits} · 실패 {len(done) - hits}", delta_color="off")
a2.metric("'항상 상승' 찍기", f"{(done['ret'] > 0).mean() * 100:.1f}%")
if len(live):
    a3.metric("진행 중 예측", f"{int(live['hit'].sum())}/{len(live)} 맞는 중")

latest_date = t.index[-1]
rows = []
for d, r in t.iloc[-20:][::-1].iterrows():
    if d == latest_date:
        result, ret, verdict = "결과 대기", "-", "🆕 최신 예측"
    elif r["done"]:
        result, ret = f"{r['end_date']:%m/%d} 확정", f"{r['ret'] * 100:+.2f}%"
        verdict = "✅ 성공" if r["hit"] else "❌ 실패"
    else:
        result, ret = "진행 중(현재가)", f"{r['ret'] * 100:+.2f}%"
        verdict = "⏳ 맞는 중" if r["hit"] else "⏳ 틀리는 중"
    rows.append({"예측한 날": f"{d:%m/%d}", "AI 상승 확률": f"{r['prob_up'] * 100:.0f}%",
                 "AI 예측": "📈 상승" if r["prob_up"] >= 0.5 else "📉 하락",
                 "결과": result, "실제 수익률": ret, "판정": verdict})
st.dataframe(rows, hide_index=True, use_container_width=True)
st.caption(f"상승 확률 50% 이상이면 '상승', 미만이면 '하락' 예측으로 보고, 예측한 날 종가와 {p.horizon}거래일 뒤 종가를 "
           "비교해 판정합니다. 기간이 덜 지난 예측은 현재가로 중간 판정합니다. "
           "과거 예측은 모두 그 날짜 이전 데이터로만 학습한 모델의 결과입니다. "
           "현재가는 1분마다 새로 받아오지만, 무료 데이터라 약 20분 늦을 수 있습니다.")

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

footer()
