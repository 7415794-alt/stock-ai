"""주식 AI 상승/하락 확률 분석 앱. 실행: streamlit run app.py"""
import os
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

from market import ai_mood, market_snapshot, stock_info
from news import adjust, fetch_news, news_score
from predictor import FEATURE_LABELS, live_price, predict, resolve_ticker

st.set_page_config(page_title="스톡시그널", page_icon="📈", layout="centered")
# 앱 껍데기(pwa/index.html)에 "화면이 다 떴어요" 신호를 보냄. 신호가 안 오면 껍데기가 자동으로 다시 불러옴
components.html("<script>window.top.postMessage('stock-ai-ready', '*')</script>", height=0)
st.markdown("""<style>
/* 위쪽 빈 공간 줄이기 */
[data-testid="stMainBlockContainer"], .block-container { padding-top: 1.2rem; }
.brand { display: flex; align-items: baseline; justify-content: space-between; margin: 0 0 4px; }
.brand b { font-size: 22px; } .brand span { font-size: 13px; color: rgba(128,128,128,1); }
/* 맨 위 메뉴: 한 줄로 두고 손가락으로 옆으로 넘기기 */
.st-key-page [data-testid="stButtonGroup"], .st-key-page div[role="radiogroup"] {
  flex-wrap: nowrap !important; overflow-x: auto; scrollbar-width: none; -webkit-overflow-scrolling: touch; }
.st-key-page [data-testid="stButtonGroup"]::-webkit-scrollbar { display: none; }
.st-key-page button { flex: 0 0 auto; white-space: nowrap; }
/* 순위 버튼: 글자를 왼쪽 정렬하고 높이를 줄여서 표처럼 */
div[class*="st-key-rk_"] button { justify-content: flex-start; text-align: left; min-height: 0;
                                  padding: 6px 12px; margin-bottom: -6px; }
div[class*="st-key-rk_"] button > div { justify-content: flex-start; width: 100%; }
div[class*="st-key-rk_"] button p { font-size: 14px; text-align: left; }
/* 기본 정보·지수 카드 (밝은/어두운 화면 모두 보이게 반투명 회색) */
.kv { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 4px 0 8px; }
.kv > div { background: rgba(128,128,128,.10); border-radius: 10px; padding: 8px 12px; }
.kv small { display: block; font-size: 12px; color: rgba(128,128,128,1); }
.kv b { font-size: 16px; } .kv i { font-style: normal; font-size: 13px; margin-left: 4px; }
.upc { color: #e5484d; } .dnc { color: #3b82f6; }
/* 분석 결과: 종목 줄 · 큰 확률 · 막대 · 요약 */
.sh { display: flex; justify-content: space-between; align-items: flex-end; margin: 6px 0 10px; gap: 8px; }
.sh b { font-size: 20px; } .sh span { display: block; font-size: 12px; color: rgba(128,128,128,1); }
.pr { text-align: right; } .pr b { font-size: 18px; } .pr i { font-style: normal; margin-left: 6px; font-size: 14px; }
.hero { background: rgba(128,128,128,.08); border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; }
.hero small { font-size: 13px; color: rgba(128,128,128,1); }
.hero .big { font-size: 40px; font-weight: 700; line-height: 1.15; margin: 2px 0 10px; }
.bar { position: relative; height: 10px; border-radius: 99px; background: #3b82f6; overflow: visible; }
.bar .fill { height: 100%; border-radius: 99px 0 0 99px; background: #e5484d; }
.bar .mark { position: absolute; top: -4px; width: 2px; height: 18px; background: rgba(128,128,128,1); }
.bl { display: flex; justify-content: space-between; font-size: 12px; margin-top: 6px; color: rgba(128,128,128,1); }
.src { font-size: 12px; color: rgba(128,128,128,1); margin-top: 8px; }
.sum { background: rgba(255,193,7,.12); border-radius: 12px; padding: 10px 14px; font-size: 14px; line-height: 1.6; margin-bottom: 8px; }
.kv3 { grid-template-columns: repeat(3, minmax(0, 1fr)); text-align: center; }
</style>
<div class="brand"><b>📈 스톡시그널</b><span>주식 AI 확률 분석</span></div>""", unsafe_allow_html=True)


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


@st.cache_data(ttl=600, show_spinner=False)  # 뉴스는 10분마다 새로
def cached_news(name, ticker):
    try:
        return fetch_news(name, ticker)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)  # 시가총액·PER 등은 1시간마다
def cached_info(ticker):
    return stock_info(ticker)


@st.cache_data(ttl=300, show_spinner=False)  # 지수는 5분마다
def cached_market():
    return market_snapshot()


def kv_grid(items):
    """[(이름, 값, 작은 덧붙임)] → 2칸 카드."""
    cells = "".join(f"<div><small>{k}</small><b>{v}</b>{x}</div>" for k, v, x in items)
    st.markdown(f'<div class="kv">{cells}</div>', unsafe_allow_html=True)


def md_escape(text):
    """뉴스 제목의 특수 문자가 굵은 글씨·수식 등으로 바뀌지 않게."""
    for ch in "\\`*_[]$~<>#":
        text = text.replace(ch, "\\" + ch)
    return text


def money(v, cur):
    return f"{v:,.0f}원" if cur == "KRW" else f"{v:,.2f} {cur}"


def footer():
    """맨 아래 안내 (어느 화면에서 멈추든 항상 보이게)."""
    st.divider()
    with st.expander("ⓘ 꼭 알아 두세요"):
        st.caption("본 앱은 과거 가격 패턴을 학습한 통계 모델의 참고용 결과이며 투자 권유가 아닙니다. "
                   "뉴스 분석은 기사 제목의 단어만 보고 간단히 판단한 것이며, 주가는 실적·거시경제 등 "
                   "모델이 보지 못하는 요인에 크게 좌우됩니다. "
                   "투자 판단과 그 결과의 책임은 본인에게 있습니다. 주가 정보는 무료 데이터라 약 20분 늦을 수 있습니다.")
    st.caption("📮 오류·문의: [slakeknowledge@gmail.com](mailto:slakeknowledge@gmail.com)")
    # 앱(pwa/index.html)이 화면 맨 아래를 잘라내므로, 마지막 내용이 가려지지 않게 여백 추가
    st.markdown('<div style="height:56px"></div>', unsafe_allow_html=True)


RANK_URL = os.environ.get("RANK_URL", "https://raw.githubusercontent.com/7415794-alt/stock-ai/data/")


@st.cache_data(ttl=600, show_spinner=False)  # 순위 파일은 하루 한 번 바뀌므로 10분마다만 다시 받음
def load_rankings():
    try:
        df = pd.read_csv(RANK_URL + "rankings.csv", dtype={"ticker": str})
        meta = requests.get(RANK_URL + "meta.json", timeout=10).json()
        return df, meta
    except Exception:
        return None, None


PAGES = ["🔍 분석", "🏆 확률순위", "🎯 적중순위", "🌏 시장 분위기"]


def open_stock(code, horizon):
    """순위에서 종목을 누르면: 검색창에 종목코드를 넣고 '종목 분석' 화면으로 이동."""
    st.session_state.query = code
    st.session_state.horizon = horizon
    st.session_state.page = st.session_state.page_last = PAGES[0]
    st.session_state.to_top = True  # 분석 화면 맨 위부터 보이게


# 순위 버튼: 글자를 왼쪽 정렬하고 높이를 줄여서 표처럼 보이게
def go_page(name):
    st.session_state.page = st.session_state.page_last = name


def pick_recent():
    """'최근 본 종목'을 누르면 그 종목을 다시 검색."""
    v = st.session_state.recent_pick
    if v:
        st.session_state.query = v
    st.session_state.recent_pick = None


def remember(code, name):
    r = [x for x in st.session_state.get("recent", []) if x[0] != code]
    st.session_state.recent = [(code, name)] + r[:5]


def rank_table(df, value_col, value_name, key, horizon):
    for i, r in enumerate(df.itertuples(), 1):
        label = (f"{i}. {r.name} ({r.market}) · {value_name} **{getattr(r, value_col) * 100:.1f}%** · "
                 f"{r.close:,.0f}원")
        st.button(label, key=f"rk_{key}_{i}", on_click=open_stock, args=(r.ticker[:6], horizon),
                  use_container_width=True)


def ranking_tab(kind):
    df, meta = load_rankings()
    if df is None or df.empty:
        st.info("순위 데이터를 준비하고 있어요. 매일 장 마감 뒤 오후 6시쯤 새로 계산됩니다.")
        return
    st.caption(f"📅 {meta.get('date', '')} 장 마감 기준 · 코스피·코스닥 {meta.get('stocks', 0):,}개 종목 · "
               f"{meta.get('built_at', '')} 계산")
    st.caption("👆 종목을 누르면 그 종목의 자세한 분석(뉴스 포함) 화면으로 이동해요.")
    for tab, h in zip(st.tabs(["1거래일", "5거래일", "20거래일"]), (1, 5, 20)):
        with tab:
            d = df[df["horizon"] == h]
            if kind == "prob":
                d = d.assign(prob_down=1 - d["prob_up"])
                st.markdown(f"#### 📈 {h}거래일 뒤 상승 확률 TOP 20")
                rank_table(d.nlargest(20, "prob_up"), "prob_up", "상승 확률", f"up{h}", h)
                st.markdown(f"#### 📉 {h}거래일 뒤 하락 확률 TOP 20")
                rank_table(d.nlargest(20, "prob_down"), "prob_down", "하락 확률", f"down{h}", h)
            else:
                d = d[d["n"] >= 60].sort_values(["hit_rate", "hits"], ascending=False)
                st.markdown(f"#### 🥇 적중률 상위 20 ({h}거래일 예측)")
                rank_table(d.head(20), "hit_rate", "적중률", f"best{h}", h)
                st.markdown(f"#### 🥶 적중률 하위 20 ({h}거래일 예측)")
                rank_table(d.tail(20).iloc[::-1], "hit_rate", "적중률", f"worst{h}", h)
    if kind == "prob":
        st.caption("AI가 계산한 확률이 높은 순서일 뿐, 실제로 오르거나 내린다는 보장은 없습니다. "
                   "순위는 뉴스를 반영하지 않은 차트 AI 확률이에요. 종목을 누르면 뉴스까지 반영한 분석을 볼 수 있어요.")
    else:
        st.caption("적중률 = 최근 60번의 예측(상승 확률 50% 이상이면 상승, 미만이면 하락) 중 실제로 맞은 비율. "
                   "과거에 잘 맞았다고 앞으로도 잘 맞는다는 보장은 없습니다.")


def analysis_tab():
    query = st.text_input("종목 검색", placeholder="🔍 삼성전자, 005930, NVDA", key="query",
                          label_visibility="collapsed")
    horizon = st.segmented_control("예측 기간", [1, 5, 20], key="horizon", on_change=keep_horizon,
                                   format_func=lambda d: f"{d}일 뒤", label_visibility="collapsed") or 5
    recent = st.session_state.get("recent", [])
    if recent:
        st.pills("🕘 최근 본 종목", [c for c, _ in recent], key="recent_pick", on_change=pick_recent,
                 format_func=lambda c: dict(recent).get(c, c))

    if not query:
        home_top(horizon)
        return

    with st.spinner("종목 검색 중..."):
        candidates = cached_search(query)
    if not candidates:
        st.warning("검색 결과가 없습니다.")
        return

    if len(candidates) > 1:
        choice = st.selectbox("종목 선택", candidates, format_func=lambda c: f"{c[1]} — {c[0]}")
    else:
        choice = candidates[0]
    ticker, name = choice
    st.session_state.setdefault("recent", [])

    try:
        with st.spinner(f"{ticker} 10년치 데이터 학습 및 검증 중..."):
            p = cached_predict(ticker, name, horizon)
    except Exception as e:
        st.error(f"분석 실패: {e}")
        return

    # ── 결과 ───────────────────────────────────────────
    lp = cached_live(ticker)
    close = p.history["Close"]
    if lp:
        last, when = lp
        before = close[close.index.date < pd.Timestamp(when).date()]
        prev = before.iloc[-1] if len(before) else close.iloc[-1]
        when_txt = f"{when:%m/%d %H:%M} 기준"
    else:
        last, prev, when_txt = close.iloc[-1], close.iloc[-2], f"{close.index[-1]:%m/%d} 종가"
    chg = last / prev - 1
    remember(p.ticker[:6] if p.ticker.endswith((".KS", ".KQ")) else p.ticker,
             p.name.split(" (")[0] if p.name != p.ticker else p.ticker)

    news = cached_news(p.name, p.ticker)
    score = news_score(news) if news else 0.0
    final = adjust(p.prob_up, score)  # 차트 AI 확률 + 뉴스 분위기
    up = final * 100
    edge = (p.cv_accuracy - p.cv_baseline_accuracy) * 100
    trusted = edge >= 2 and p.cv_auc >= 0.53

    t = p.track.copy()
    # 아직 기간이 안 끝난 예측은 방금 받은 현재가로 다시 판정
    t.loc[~t["done"], "ret"] = last / t.loc[~t["done"], "start"] - 1
    t["hit"] = (t["prob_up"] >= 0.5) == (t["ret"] > 0)
    done = t[t["done"]].iloc[-60:]
    hit_rate = done["hit"].mean() * 100 if len(done) else float("nan")

    if not news:
        mood, mood_short = "뉴스 없음", "없음"
    else:
        mood = ("😀 긍정적" if score >= 0.3 else "🙂 조금 긍정적" if score >= 0.1 else
                "😟 부정적" if score <= -0.3 else "😕 조금 부정적" if score <= -0.1 else "😐 중립")
        mood_short = mood.split(" ", 1)[1]

    # ① 종목 이름 + 현재가 (한 줄)
    cls = "upc" if chg > 0 else "dnc" if chg < 0 else ""
    st.markdown(f"""<div class="sh"><div><b>{p.name.split(' (')[0]}</b><span>{p.ticker}</span></div>
<div class="pr"><b>{money(last, p.currency)}</b><i class="{cls}">{chg * 100:+.2f}%</i><span>{when_txt}</span></div></div>""",
                unsafe_allow_html=True)

    # ② 확률 (가장 크게) + 막대
    big_cls = "upc" if up >= 50 else "dnc"
    st.markdown(f"""<div class="hero"><small>{p.horizon}거래일 뒤 오를 확률</small>
<div class="big {big_cls}">{up:.1f}%</div>
<div class="bar"><div class="fill" style="width:{up:.1f}%"></div><div class="mark" style="left:{p.base_rate * 100:.1f}%"></div></div>
<div class="bl"><span class="upc">상승 {up:.1f}%</span><span>│ 평소 {p.base_rate * 100:.0f}%</span><span class="dnc">하락 {100 - up:.1f}%</span></div>
<div class="src">📊 차트 AI {p.prob_up * 100:.1f}% · 📰 뉴스 {(final - p.prob_up) * 100:+.1f}%p</div></div>""",
                unsafe_allow_html=True)

    # ③ 한 줄 요약 + 핵심 3가지
    direction = ("오를 가능성을 더 높게" if up >= 55 else "내릴 가능성을 더 높게" if up <= 45
                 else "오를지 내릴지 비슷하게")
    # 이 종목의 최근 적중률이 코스피·코스닥 전 종목(같은 예측 기간) 중 상위 몇 %인지
    top, pool_n = None, 0
    rank_df, _ = load_rankings()
    if rank_df is not None and p.ticker.endswith((".KS", ".KQ")) and hit_rate == hit_rate:
        pool = rank_df[(rank_df["horizon"] == p.horizon) & (rank_df["n"] >= 60)]["hit_rate"].dropna()
        if len(pool):
            top = max(1, round(((pool > hit_rate / 100).sum() + 1) / len(pool) * 100))
            pool_n = len(pool)
    if top is None:
        hit_txt = f"최근 {len(done)}번 중 {hit_rate:.0f}%를 맞혔어요." if hit_rate == hit_rate else ""
    elif top <= 30:
        hit_txt = f"최근 적중률이 전체 종목 중 <b>상위 {top}%</b>로 좋은 편이에요."
    elif top >= 70:
        hit_txt = f"다만 최근 적중률이 전체 종목 중 <b>상위 {top}%</b>(하위권)라 참고만 하세요."
    else:
        hit_txt = f"최근 적중률은 전체 종목 중 상위 {top}%로 중간 정도예요."
    st.markdown(f'<div class="sum">💡 AI는 이 종목이 {p.horizon}거래일 뒤 <b>{direction}</b> 보고 있어요. {hit_txt}</div>',
                unsafe_allow_html=True)
    kv3 = [("최근 적중률", f"{hit_rate:.0f}%" if hit_rate == hit_rate else "-", ""),
           ("적중률 순위", f"상위 {top}%" if top else "-", ""),
           ("뉴스", mood_short, "")]
    cells = "".join(f"<div><small>{k}</small><b>{v}</b></div>" for k, v, _ in kv3)
    st.markdown(f'<div class="kv kv3">{cells}</div>', unsafe_allow_html=True)

    # ④ 차트
    h = p.history.iloc[-252:]
    fig = go.Figure(go.Candlestick(x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
                                   increasing_line_color="#e5484d", decreasing_line_color="#3b82f6", name="가격"))
    for n, color in ((20, "orange"), (60, "green")):
        fig.add_scatter(x=h.index, y=close.rolling(n).mean().iloc[-252:],
                        line=dict(width=1, color=color), name=f"{n}일선")
    fig.update_layout(height=280, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=1.08, x=0), xaxis=dict(tickformat="%y.%m"))
    st.plotly_chart(fig, use_container_width=True)

    # ⑤ 자세한 내용은 탭으로 나눠서
    tab_n, tab_i, tab_t, tab_d = st.tabs(["📰 뉴스", "🏢 기본 정보", "🎯 적중 기록", "🧮 상세 분석"])

    with tab_n:
        if news is None:
            st.info("지금은 뉴스를 불러오지 못했어요. 잠시 뒤 다시 열어 주세요. (확률에는 뉴스를 반영하지 않았어요)")
        elif not news:
            st.info("최근 7일 동안 이 종목 관련 뉴스가 없어요. 확률에는 뉴스를 반영하지 않았어요.")
        else:
            good, bad = sum(x["sign"] > 0 for x in news), sum(x["sign"] < 0 for x in news)
            st.markdown(f"**뉴스 분위기: {mood}** · 🔴 호재 {good} · 🔵 악재 {bad} · ⚪ 중립 {len(news) - good - bad} "
                        f"→ 확률 **{(final - p.prob_up) * 100:+.1f}%p**")

            def show(items):
                for x in items:
                    tag = "🔴" if x["sign"] > 0 else "🔵" if x["sign"] < 0 else "⚪"
                    why = " · ".join([f"+{w}" for w in x["pos"]] + [f"-{w}" for w in x["neg"]])
                    st.markdown(f"{tag} [{md_escape(x['title'])}]({x['link']})  \n"
                                f":gray[{md_escape(x['source'])} · {x['when']:%m/%d %H:%M}"
                                f"{' · ' + md_escape(why) if why else ''}]")
            show(news[:5])
            if len(news) > 5:
                with st.expander(f"뉴스 {len(news) - 5}개 더 보기"):
                    show(news[5:])
            st.caption("구글 뉴스 최근 7일 기사 제목의 호재·악재 단어로 판단해요. 최근 기사일수록 크게 반영하고, "
                       "확률은 최대 ±6%p까지만 바꿔요. 단어만 보는 간단한 방식이라 틀릴 수 있어요.")

    with tab_i:
        info = cached_info(p.ticker)
        if info:
            kv_grid([(k, v, "") for k, v in info.items()])
            st.caption("한국 종목은 네이버 증권, 해외 종목은 야후 파이낸스 자료예요. 적자 기업 등은 PER이 '-'로 나와요.")
        else:
            st.info("지금은 기본 정보를 불러오지 못했어요.")

    with tab_t:
        live = t[~t["done"]].iloc[:-1]  # 최신 날짜(방금 낸 예측)는 제외
        hits = int(done["hit"].sum())
        kv_grid([(f"최근 {len(done)}번 적중률", f"{hit_rate:.1f}%", f"<small>성공 {hits} · 실패 {len(done) - hits}</small>"),
                 ("'항상 상승' 찍기", f"{(done['ret'] > 0).mean() * 100:.1f}%", "<small>비교용 기준</small>")])
        if top:
            st.caption(f"🏅 적중률 순위: 같은 {p.horizon}거래일 예측 기준 코스피·코스닥 {pool_n:,}개 종목 중 상위 {top}% "
                       "(매일 장 마감 후 계산한 순위 데이터와 비교)")
        rows = []
        for d, r in t.iloc[-11:-1][::-1].iterrows() if len(t) > 1 else []:
            if r["done"]:
                verdict = "✅" if r["hit"] else "❌"
            else:
                verdict = "⏳✅" if r["hit"] else "⏳❌"
            rows.append({"날짜": f"{d:%m/%d}", "AI 예측": f"{'📈' if r['prob_up'] >= 0.5 else '📉'} {r['prob_up'] * 100:.0f}%",
                         "실제": f"{r['ret'] * 100:+.1f}%", "결과": verdict})
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(f"최근 10번의 예측이에요. 상승 확률 50% 이상이면 '상승' 예측이고, 예측한 날 종가와 {p.horizon}거래일 뒤 "
                   "종가를 비교해요. ⏳는 아직 기간이 안 끝나 현재가로 중간 판정한 것이에요. 뉴스를 빼고 차트 AI 기준이에요.")

    with tab_d:
        st.markdown("**🔍 이 예측을 얼마나 믿을 수 있나요?**")
        kv_grid([("과거 검증 정확도", f"{p.cv_accuracy * 100:.1f}%", ""),
                 ("단순 찍기 정확도", f"{p.cv_baseline_accuracy * 100:.1f}%", f"<small>차이 {edge:+.1f}%p</small>"),
                 ("AUC", f"{p.cv_auc:.3f}", "<small>0.5 = 동전 던지기</small>"),
                 ("학습한 거래일", f"{p.n_samples:,}일", "")])
        if not trusted:
            st.warning("이 종목에서는 모델이 '항상 같은 방향을 찍는' 단순 기준선보다 뚜렷하게 낫지 않았어요.")
        st.caption("과거 데이터를 시간순으로 나눠, 과거로 학습하고 그 다음 구간을 맞혀 본 결과(워크포워드 검증)예요.")
        st.markdown("**🧮 현재 지표와 모델이 중요하게 본 요인**")
        f = p.latest_features
        pct_keys = {"ret_1", "ret_5", "ret_20", "sma5_ratio", "sma20_ratio", "sma60_ratio", "vol_20", "dist_52w_high"}
        rows = []
        for k in p.importances.index[::-1]:
            v = f[k]
            val = f"{v * 100:+.2f}%" if k in pct_keys else f"{v:.2f}" if k != "macd_hist" else f"{v * 100:+.3f}%"
            rows.append({"지표": FEATURE_LABELS[k], "현재 값": val, "중요도": f"{p.importances[k] * 100:.1f}%"})
        st.dataframe(rows, hide_index=True, use_container_width=True)


# 다른 화면에 가 있어도 검색어·예측 기간이 지워지지 않게 보관
for k in ("query", "horizon"):
    if k in st.session_state:
        st.session_state[k] = st.session_state[k]
st.session_state.setdefault("horizon", 5)
st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("page_last", PAGES[0])


def keep_horizon():
    if st.session_state.horizon is None:
        st.session_state.horizon = st.session_state.get("horizon_last", 5)
    st.session_state.horizon_last = st.session_state.horizon


def home_top(horizon):
    """검색 전 첫 화면: 오늘의 상승 확률 TOP 5."""
    df, meta = load_rankings()
    if df is None or df.empty:
        return
    st.markdown(f"#### 🔥 오늘의 상승 확률 TOP 5 · {horizon}일 뒤")
    d = df[df["horizon"] == horizon].nlargest(5, "prob_up")
    for i, r in enumerate(d.itertuples(), 1):
        st.button(f"{i}. {r.name} · **{r.prob_up * 100:.1f}%**", key=f"rk_home_{i}",
                  on_click=open_stock, args=(r.ticker[:6], horizon), use_container_width=True)
    st.button("전체 순위 보기 ›", key="all_rank", on_click=go_page, args=(PAGES[1],))
    st.caption(f"{meta.get('date', '')} 장 마감 기준 · 뉴스 미반영 차트 AI 확률 · 누르면 자세한 분석으로 이동")


def market_tab():
    st.markdown("#### 🌏 주요 지수")
    snap = cached_market()
    if not snap:
        st.info("지금은 지수를 불러오지 못했어요. 잠시 뒤 다시 열어 주세요.")
    else:
        cells = []
        for x in snap:
            val = f"{x['last']:,.2f}원" if "환율" in x["name"] else f"{x['last']:,.2f}"
            cls, arrow = ("upc", "▲") if x["chg"] > 0 else ("dnc", "▼") if x["chg"] < 0 else ("", "")
            cells.append((x["name"], val, f'<i class="{cls}">{arrow}{x["chg"] * 100:+.2f}%</i>'))
        kv_grid(cells)
        st.caption("전 거래일 종가 대비 · 장중에는 약 20분 늦은 값 · 미국 지수는 한국 시간 새벽에 마감해요.")
        fig = go.Figure()
        for x in snap:
            if x["name"] in ("코스피", "코스닥"):
                h = x["hist"]
                fig.add_scatter(x=h.index, y=(h / h.iloc[0] - 1) * 100, name=x["name"], mode="lines")
        fig.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10), title="최근 3개월 등락률(%)",
                          legend=dict(orientation="h", y=-0.15), xaxis=dict(tickformat="%m월"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🤖 AI가 본 시장 분위기")
    df, meta = load_rankings()
    if df is None or df.empty:
        st.info("AI 순위 데이터를 준비하고 있어요.")
        return
    mood = ai_mood(df)
    cells = []
    for h in (1, 5, 20):
        share = (df[df["horizon"] == h]["prob_up"] >= 0.5).mean()
        label = "상승 우세" if share >= 0.55 else "하락 우세" if share <= 0.45 else "팽팽"
        cls = "upc" if share >= 0.55 else "dnc" if share <= 0.45 else ""
        detail = " · ".join(f"{ko} {mood.loc[h, m] * 100:.0f}%"
                            for m, ko in (("KOSPI", "코스피"), ("KOSDAQ", "코스닥")) if m in mood.columns)
        cells.append((f"{h}일 뒤 오를 것 같은 종목", f"{share * 100:.0f}%",
                      f'<i class="{cls}">{label}</i><small>{detail}</small>'))
    kv_grid(cells)
    st.caption(f"{meta.get('date', '')} 장 마감 기준 · 코스피·코스닥 {meta.get('stocks', 0):,}개 종목 중 "
               "AI 상승 확률이 50% 이상인 종목의 비율이에요. 55% 이상이면 '상승 우세', 45% 이하면 '하락 우세'로 표시해요.")


def keep_page():
    # 이미 선택된 버튼을 한 번 더 눌러 선택이 풀리면 원래 화면으로 되돌림
    if st.session_state.page is None:
        st.session_state.page = st.session_state.page_last
    st.session_state.page_last = st.session_state.page


if st.session_state.page not in PAGES:  # 예전 메뉴 이름이 남아 있을 때
    st.session_state.page = st.session_state.page_last = PAGES[0]
page = st.segmented_control("화면", PAGES, key="page", on_change=keep_page,
                            label_visibility="collapsed") or PAGES[0]
if page == PAGES[1]:
    ranking_tab("prob")
elif page == PAGES[2]:
    ranking_tab("hit")
elif page == PAGES[3]:
    market_tab()
else:
    if st.session_state.pop("to_top", False):
        # 순위에서 종목을 눌러 넘어왔을 때 화면을 맨 위로 올림 (숫자는 매번 새로 실행되게 하는 표시)
        components.html(f"""<script>/*{time.time()}*/
          for (const q of ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', 'section.main']) {{
            const el = window.parent.document.querySelector(q); if (el) el.scrollTo(0, 0);
          }}
          window.parent.scrollTo(0, 0);</script>""", height=0)
    analysis_tab()
footer()
