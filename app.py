"""주식 AI 상승/하락 확률 분석 앱. 실행: streamlit run app.py"""
import json
import os
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

from market import ai_mood, company, halted, krx_status, market_clock, market_snapshot
from news import adjust, fetch_news, news_score
from indicators import build_1d, cols_for, fundamentals_now, indicators_for
from predictor import FEATURE_LABELS, live_price, load_history, load_index, market_index, predict, resolve_ticker

st.set_page_config(page_title="스톡시그널", page_icon="📈", layout="centered")
# 앱 껍데기(pwa/index.html)에 "화면이 다 떴어요" 신호를 보냄. 신호가 안 오면 껍데기가 자동으로 다시 불러옴
components.html("<script>window.top.postMessage('stock-ai-ready', '*')</script>", height=0)
st.markdown("""<style>
/* 위쪽 빈 공간 줄이기 */
[data-testid="stMainBlockContainer"], .block-container { padding-top: 1.2rem; }
/* 맨 위 '스톡시그널' 글자 = 홈 버튼 */
.st-key-brand [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; align-items: center; margin-bottom: -8px; }
.st-key-brand [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; }
.st-key-brand button { border: none !important; background: transparent !important; padding: 0 !important;
                       min-height: 0; box-shadow: none !important; }
.st-key-brand button p { font-size: 22px !important; font-weight: 700; }
.brand-sub { font-size: 13px; color: rgba(128,128,128,1); text-align: right; }
/* 입력칸 옆 '확인' 버튼 (휴대폰에서도 한 줄) */
div[class*="st-key-row_"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 8px; }
div[class*="st-key-row_"] [data-testid="stColumn"]:first-child { min-width: 0 !important; flex: 1 1 auto !important; width: auto !important; }
div[class*="st-key-row_"] [data-testid="stColumn"]:last-child { min-width: 0 !important; flex: 0 0 auto !important; width: auto !important; }
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
/* 관심 종목·공유 버튼은 휴대폰에서도 한 줄에 나란히 */
.st-key-actions [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 8px; }
.st-key-actions [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
.follow { font-size: 14px; margin: -2px 0 8px; } .follow small { color: rgba(128,128,128,1); }
/* 종목 비교 표 */
.cmp { width: 100%; border-collapse: collapse; font-size: 14px; margin: 8px 0; }
.cmp th, .cmp td { padding: 7px 4px; border-bottom: 1px solid rgba(128,128,128,.2); text-align: center; border-left: none; border-right: none; }
.cmp td:first-child { text-align: left; font-size: 12px; color: rgba(128,128,128,1); width: 32%; }
.cmp td.win { font-weight: 700; color: #e5484d; } .cmp td.lose { color: #3b82f6; } .cmp small { color: rgba(128,128,128,1); }
/* 장 상태 · 위험 종목 경고 · AI vs 나 */
.clock { font-size: 12px; color: rgba(128,128,128,1); margin: -2px 0 6px; } .clock b { color: inherit; font-weight: 600; }
.warn { background: rgba(245,158,11,.15); border-radius: 10px; padding: 8px 12px; font-size: 13px; margin: -4px 0 10px; }
.st-key-vote [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 8px; }
.st-key-vote [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
.st-key-nav_back { display: none; }  /* 휴대폰 '뒤로 가기'용 숨은 버튼 */
.vs { background: rgba(128,128,128,.08); border-radius: 12px; padding: 10px 14px; font-size: 14px; margin: 4px 0 8px; }
</style>""", unsafe_allow_html=True)


def go_home():
    """맨 위 '스톡시그널'을 누르면: 검색어를 비우고 첫 화면(홈)으로."""
    st.session_state.query = ""
    st.session_state.page = st.session_state.page_last = "🔍 분석"
    st.session_state.to_top = True


with st.container(key="brand"):
    _b1, _b2 = st.columns([3, 2], vertical_alignment="center")
    _b1.button("📈 스톡시그널", key="home_btn", on_click=go_home)
    _b2.markdown('<div class="brand-sub">주식 AI 확률 분석</div>', unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(q):
    return resolve_ticker(q)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_predict(ticker, name, horizon, version=6):
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


@st.cache_data(ttl=3600, show_spinner=False)  # 시가총액·PER·매매·일정은 1시간마다
def cached_company(ticker):
    return company(ticker)


@st.cache_data(ttl=300, show_spinner=False)  # 지수는 5분마다
def cached_market():
    return market_snapshot()


def predict_history(ticker):
    return load_history(ticker, years=1)


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
        if "halted" in df:  # 거래정지 종목은 순위·조건검색·시장 분위기 어디에도 안 나오게
            meta["halted_codes"] = sorted(df.loc[df["halted"] == 1, "ticker"].str[:6].unique())
            df = df[df["halted"] != 1].reset_index(drop=True)
        stopped = {c for c, v in cached_status().items() if v["halted"]}  # 오늘 새로 정지된 종목도 바로 빼기
        if stopped:
            df = df[~df["ticker"].str[:6].isin(stopped)].reset_index(drop=True)
        return df, meta
    except Exception:
        return None, None


@st.cache_data(ttl=600, show_spinner=False)  # 거래정지 여부는 10분마다 다시 확인
def cached_halted(code):
    return halted(code)


@st.cache_data(ttl=600, show_spinner=False)  # 전 종목 상태(거래정지·관리·투자주의)는 10분마다
def cached_status():
    try:
        return krx_status()
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def cached_clock():
    return market_clock()


def _kr_code(ticker):
    if ticker.endswith((".KS", ".KQ")) or (len(ticker) == 6 and ticker.isdigit()):
        return ticker[:6]
    return None


def is_halted(ticker):
    """한국 종목만 확인 (해외 종목은 항상 False)."""
    code = _kr_code(ticker)
    if not code:
        return False
    stt = cached_status().get(code)
    return stt["halted"] if stt else cached_halted(code)


ALERTS = {1: "투자주의", 2: "투자경고", 3: "투자위험"}


def warn_words(ticker):
    """['관리종목', '투자경고'] 처럼 조심할 점 (없으면 빈 목록)."""
    code = _kr_code(ticker)
    stt = cached_status().get(code) if code else None
    if not stt:
        return []
    return (["관리종목"] if stt["managed"] else []) + ([ALERTS[stt["alert"]]] if stt["alert"] in ALERTS else [])


def warn_tag(ticker):
    w = warn_words(ticker)
    return f" · ⚠️{'·'.join(w)}" if w else ""


@st.cache_data(ttl=600, show_spinner=False)
def load_prev():
    """어제 순위 (순위 변동 ▲▼🆕 표시용). 아직 없으면 None."""
    try:
        prev = pd.read_csv(RANK_URL + "prev_rankings.csv", dtype={"ticker": str})
        return prev[prev["halted"] != 1].reset_index(drop=True) if "halted" in prev else prev
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load_scorecard():
    """매일 계산할 때 쌓는 'AI 성적표' (어제 TOP 20이 오늘 실제로 맞았는지)."""
    try:
        return pd.read_csv(RANK_URL + "scorecard.csv")
    except Exception:
        return None


def scorecard_box(compact=False):
    card = load_scorecard()
    if card is None or card.empty:
        return
    r = card.iloc[-1]
    d0, d1 = (pd.Timestamp(x).strftime("%m/%d") for x in (r["date"], r["eval_date"]))
    up_pct, all_pct = r["up_hit"] / r["up_n"] * 100, r["all_up"] * 100
    if compact:  # 첫 화면용 한 줄
        st.markdown(f'<div class="follow">📋 어제 AI 성적표: 상승 TOP 20 중 <b class="upc">{int(r["up_hit"])}개</b> 실제 상승 '
                    f'<small>(전체 종목은 {all_pct:.0f}% 상승 · {d0}→{d1})</small></div>', unsafe_allow_html=True)
        return
    st.markdown(f"#### 📋 어제 AI 성적표 <small>({d0} 예측 → {d1} 결과)</small>", unsafe_allow_html=True)
    up_cls = "upc" if r["up_avg"] > 0 else "dnc"
    cells = [("📈 상승 TOP 20", f'{int(r["up_hit"])}개 올랐어요',
              f'<small>{up_pct:.0f}% 적중 · 평균 <span class="{up_cls}">{r["up_avg"] * 100:+.2f}%</span></small>'),
             ("📉 하락 TOP 20", f'{int(r["down_hit"])}개 내렸어요',
              f'<small>{r["down_hit"] / r["down_n"] * 100:.0f}% 적중 · 평균 {r["down_avg"] * 100:+.2f}%</small>'),
             ("전체 종목 (비교용)", f"{all_pct:.0f}% 올랐어요", f'<small>평균 {r["all_avg"] * 100:+.2f}%</small>')]
    if len(card) >= 2:
        last = card.tail(20)
        cells.append((f"최근 {len(last)}일 평균", f'{last["up_hit"].sum() / last["up_n"].sum() * 100:.0f}% 적중',
                      f'<small>상승 TOP 20 기준 · 전체 {last["all_up"].mean() * 100:.0f}%</small>'))
    kv_grid(cells)
    if len(card) >= 2:
        with st.expander("날짜별 성적 보기"):
            st.dataframe([{"예측일": pd.Timestamp(x.date).strftime("%m/%d"),
                           "상승 TOP 20": f"{int(x.up_hit)}/{int(x.up_n)}",
                           "하락 TOP 20": f"{int(x.down_hit)}/{int(x.down_n)}",
                           "전체 상승": f"{x.all_up * 100:.0f}%"} for x in card.iloc[::-1].head(30).itertuples()],
                         hide_index=True, use_container_width=True)
    st.caption("어제 '1거래일 뒤' 상승 확률 TOP 20 종목이 오늘 실제로 올랐는지(하락 TOP 20은 내렸는지) 매일 채점해요. "
               "전체 종목 상승 비율보다 높아야 AI가 의미 있게 고른 거예요.")


def order_of(df, h, kind):
    """같은 기준으로 줄 세운 종목 순서 → {종목코드: 등수}."""
    if df is None:
        return None
    d = df[df["horizon"] == h]
    if kind == "up":
        d = d.sort_values("prob_up", ascending=False)
    elif kind == "down":
        d = d.sort_values("prob_up")
    else:
        d = d[d["n"] >= 60].sort_values(["hit_rate", "hits"], ascending=False)
        if kind == "worst":
            d = d.iloc[::-1]
    return {t: i for i, t in enumerate(d["ticker"], 1)}


def move_txt(prev, ticker, rank, size):
    """어제보다 몇 계단 올랐는지: ▲3 / ▼2 / - / 🆕 (어제는 이 목록에 없었음)."""
    if not prev:
        return ""
    old = prev.get(ticker)
    if old is None or old > size:
        return " · 🆕"
    diff = old - rank
    return f" · ▲{diff}" if diff > 0 else f" · ▼{-diff}" if diff < 0 else " · -"


PAGES = ["🔍 분석", "💼 내 종목", "🔎 조건검색", "🏆 확률순위", "🎯 적중순위", "🆚 비교", "🌏 시장 분위기"]
P_ANALYZE, P_HOLD, P_SCREEN, P_PROB, P_HIT, P_CMP, P_MARKET = PAGES
SHARE_URL = "https://7415794-alt.github.io/stock-ai-icon/"  # 공유 링크 = 앱 껍데기 주소


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


def rank_table(df, value_col, value_name, key, horizon, prev=None):
    for i, r in enumerate(df.itertuples(), 1):
        label = (f"{i}. {r.name} ({r.market}) · {value_name} **{getattr(r, value_col) * 100:.1f}%** · "
                 f"{r.close:,.0f}원{move_txt(prev, r.ticker, i, len(df))}{warn_tag(r.ticker)}")
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
    if kind == "prob":
        scorecard_box()
    prev_df = load_prev()
    if prev_df is not None and meta.get("prev_date"):
        st.caption(f"▲▼ = {meta['prev_date']} 순위보다 오르거나 내린 계단 수 · 🆕 = 새로 들어온 종목")
    for tab, h in zip(st.tabs(["1거래일", "5거래일", "20거래일"]), (1, 5, 20)):
        with tab:
            d = df[df["horizon"] == h]
            if kind == "prob":
                d = d.assign(prob_down=1 - d["prob_up"])
                st.markdown(f"#### 📈 {h}거래일 뒤 상승 확률 TOP 20")
                rank_table(d.nlargest(20, "prob_up"), "prob_up", "상승 확률", f"up{h}", h,
                           order_of(prev_df, h, "up"))
                st.markdown(f"#### 📉 {h}거래일 뒤 하락 확률 TOP 20")
                rank_table(d.nlargest(20, "prob_down"), "prob_down", "하락 확률", f"down{h}", h,
                           order_of(prev_df, h, "down"))
            else:
                d = d[d["n"] >= 60].sort_values(["hit_rate", "hits"], ascending=False)
                st.markdown(f"#### 🥇 적중률 상위 20 ({h}거래일 예측)")
                rank_table(d.head(20), "hit_rate", "적중률", f"best{h}", h, order_of(prev_df, h, "best"))
                st.markdown(f"#### 🥶 적중률 하위 20 ({h}거래일 예측)")
                rank_table(d.tail(20).iloc[::-1], "hit_rate", "적중률", f"worst{h}", h,
                           order_of(prev_df, h, "worst"))
    if kind == "prob":
        st.caption("AI가 계산한 확률이 높은 순서일 뿐, 실제로 오르거나 내린다는 보장은 없습니다. "
                   "순위는 뉴스를 반영하지 않은 차트 AI 확률이에요. 종목을 누르면 뉴스까지 반영한 분석을 볼 수 있어요.")
    else:
        st.caption("적중률 = 최근 60번의 예측(상승 확률 50% 이상이면 상승, 미만이면 하락) 중 실제로 맞은 비율. "
                   "과거에 잘 맞았다고 앞으로도 잘 맞는다는 보장은 없습니다.")


def code_of(ticker):
    """한국 종목은 6자리 코드, 해외 종목은 티커 그대로 (검색창·관심 종목·공유 링크에 씀)."""
    return ticker[:6] if ticker.endswith((".KS", ".KQ")) else ticker


def ai_follow(t, horizon):
    """최근 1년 동안 horizon 거래일마다: AI가 '상승'(50% 이상)이면 사서 들고 있고, 아니면 쉼."""
    d = t[t["done"]]
    if len(d) < 10:
        return None
    d = d[d.index >= d.index[-1] - pd.Timedelta(days=365)].iloc[::horizon]
    if len(d) < 5:
        return None
    buy = d["prob_up"] >= 0.5
    strat = (1 + d["ret"].where(buy, 0.0)).cumprod()
    hold = (1 + d["ret"]).cumprod()
    return {"x": d["end_date"], "strat": strat, "hold": hold, "n_buy": int(buy.sum()), "n": len(d),
            "strat_ret": float(strat.iloc[-1] - 1), "hold_ret": float(hold.iloc[-1] - 1)}


def core(ticker, name, horizon):
    """분석 화면과 비교 화면이 함께 쓰는 계산 (현재가·뉴스 반영 확률·적중률·순위·AI 따라 했다면)."""
    p = cached_predict(ticker, name, horizon)
    lp = cached_live(ticker)
    close = p.history["Close"]
    if lp:
        last, when = lp
        before = close[close.index.date < pd.Timestamp(when).date()]
        prev = before.iloc[-1] if len(before) else close.iloc[-1]
        when_txt = f"{when:%m/%d %H:%M} 기준"
    else:
        last, prev, when_txt = close.iloc[-1], close.iloc[-2], f"{close.index[-1]:%m/%d} 종가"

    news = cached_news(p.name, p.ticker)
    score = news_score(news) if news else 0.0
    final = adjust(p.prob_up, score)  # 차트 AI 확률 + 뉴스 분위기
    if not news:
        mood = "뉴스 없음"
    else:
        mood = ("😀 긍정적" if score >= 0.3 else "🙂 조금 긍정적" if score >= 0.1 else
                "😟 부정적" if score <= -0.3 else "😕 조금 부정적" if score <= -0.1 else "😐 중립")

    t = p.track.copy()
    # 아직 기간이 안 끝난 예측은 방금 받은 현재가로 다시 판정
    t.loc[~t["done"], "ret"] = last / t.loc[~t["done"], "start"] - 1
    t["hit"] = (t["prob_up"] >= 0.5) == (t["ret"] > 0)
    done = t[t["done"]].iloc[-60:]
    hit_rate = done["hit"].mean() * 100 if len(done) else float("nan")

    # 이 종목의 최근 적중률이 코스피·코스닥 전 종목(같은 예측 기간) 중 상위 몇 %인지
    top, pool_n = None, 0
    rank_df, _ = load_rankings()
    if rank_df is not None and p.ticker.endswith((".KS", ".KQ")) and hit_rate == hit_rate:
        pool = rank_df[(rank_df["horizon"] == p.horizon) & (rank_df["n"] >= 60)]["hit_rate"].dropna()
        if len(pool):
            top = max(1, round(((pool > hit_rate / 100).sum() + 1) / len(pool) * 100))
            pool_n = len(pool)
    year = close[close.index >= close.index[-1] - pd.Timedelta(days=365)]
    return {"p": p, "close": close, "last": last, "chg": last / prev - 1, "when_txt": when_txt,
            "news": news, "score": score, "final": final, "mood": mood, "t": t, "done": done,
            "hit_rate": hit_rate, "top": top, "pool_n": pool_n, "follow": ai_follow(t, horizon),
            "year_ret": float(last / year.iloc[0] - 1), "short": p.name.split(" (")[0]}


def toggle_fav(code, name):
    fav = st.session_state.get("fav", [])
    if any(c == code for c, _ in fav):
        st.session_state.fav = [x for x in fav if x[0] != code]
    else:
        st.session_state.fav = ([(code, name)] + fav)[:20]


def share_button(code, horizon, title, text):
    """공유 버튼: 앱 껍데기(GitHub Pages)에 부탁해서 휴대폰 공유 창을 띄움. 안 되면 링크 복사."""
    url = f"{SHARE_URL}?s={code}&h={horizon}"
    payload = json.dumps({"type": "stock-ai-share", "url": url, "title": title, "text": text}, ensure_ascii=False)
    components.html(f"""
<style>
 button {{ width: 100%; height: 38px; border-radius: 8px; border: 1px solid rgba(128,128,128,.4);
          background: transparent; font: 15px -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
          color: #31333f; cursor: pointer; }}
 @media (prefers-color-scheme: dark) {{ button {{ color: #fafafa; }} }}
 body {{ margin: 0; }}
</style>
<button id="b">🔗 공유하기</button>
<script>
 const msg = {payload}, b = document.getElementById('b');
 let ok = false;
 addEventListener('message', (e) => {{ if (e.data === 'stock-ai-share-ok') ok = true; }});
 b.onclick = () => {{
   ok = false;
   try {{ window.top.postMessage(msg, '*'); }} catch (e) {{}}
   setTimeout(async () => {{
     if (ok) return;
     try {{ if (navigator.share) {{ await navigator.share({{title: msg.title, text: msg.text, url: msg.url}}); return; }} }}
     catch (e) {{ if (e.name === 'AbortError') return; }}
     try {{ await navigator.clipboard.writeText(msg.text + ' ' + msg.url); b.textContent = '✅ 링크를 복사했어요'; return; }}
     catch (e) {{}}
     prompt('이 링크를 복사해서 보내 주세요', msg.url);
   }}, 400);
 }};
</script>""", height=40)


def today_kst():
    return pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d")


def add_vote(ticker, name, horizon, price, me, ai):
    votes = [v for v in st.session_state.get("votes", [])
             if not (v[0] == ticker and v[2] == horizon and v[3] == today_kst())]
    st.session_state.votes = ([(ticker, name, horizon, today_kst(), float(price), me, int(round(ai)))] + votes)[:60]


def vote_box(ticker, name, horizon, price, ai):
    """AI vs 나: 오를지 내릴지 내가 먼저 찍어 보고, 나중에 AI와 누가 맞혔는지 비교."""
    mine = next((v for v in st.session_state.get("votes", [])
                 if v[0] == ticker and v[2] == horizon and v[3] == today_kst()), None)
    if mine:
        me = "📈 오른다" if mine[5] == "u" else "📉 내린다"
        ai_txt = "📈 오른다" if mine[6] >= 50 else "📉 내린다"
        st.markdown(f'<div class="vs">🗳 <b>AI vs 나</b> · 나: <b>{me}</b> · AI: <b>{ai_txt}</b> ({mine[6]}%)<br>'
                    f'<small>{horizon}거래일 뒤 누가 맞혔는지 💼 내 종목 화면에서 볼 수 있어요.</small></div>',
                    unsafe_allow_html=True)
        return
    st.markdown(f"**🗳 AI vs 나** · {horizon}거래일 뒤 이 종목, 나는 어떻게 생각해요?")
    with st.container(key="vote"):
        v1, v2 = st.columns(2)
        v1.button("📈 오른다", key="vote_u", on_click=add_vote, args=(ticker, name, horizon, price, "u", ai),
                  use_container_width=True)
        v2.button("📉 내린다", key="vote_d", on_click=add_vote, args=(ticker, name, horizon, price, "d", ai),
                  use_container_width=True)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_close(ticker):
    try:
        c = predict_history(ticker)["Close"]
        c.index = c.index.tz_localize(None).normalize() if c.index.tz is not None else c.index.normalize()
        return c
    except Exception:
        return None


def vote_results():
    """찍은 것들을 채점: [(vote, 끝났는지, 실제 수익률, 남은 거래일)]."""
    out = []
    for v in st.session_state.get("votes", []):
        ticker, _, h, day, price = v[:5]
        c = cached_close(ticker)
        if c is None or c.empty:
            continue
        pos = max(int(c.index.searchsorted(pd.Timestamp(day), side="right")) - 1, 0)  # 찍은 날(또는 그 전 거래일)
        end_i = pos + h  # h 거래일 뒤
        if end_i <= len(c) - 1:
            # 결과일이 오늘이면 장이 끝나야 확정
            done = c.index[end_i].strftime("%Y-%m-%d") < today_kst() or cached_clock()["state"] == "closed"
        else:
            done = False
        end = c.iloc[min(end_i, len(c) - 1)]
        out.append((v, done, float(end / price - 1), max(0, end_i - (len(c) - 1))))
    return out


def vote_section():
    st.markdown("#### 🗳 AI vs 나")
    res = vote_results()
    if not res:
        st.caption("종목 분석 화면에서 '📈 오른다 / 📉 내린다'를 찍어 보세요. 기간이 지나면 나와 AI 중 누가 맞혔는지 여기서 채점해요.")
        return
    done = [x for x in res if x[1]]
    if done:
        me_hit = sum((v[5] == "u") == (r > 0) for v, _, r, _ in done)
        ai_hit = sum((v[6] >= 50) == (r > 0) for v, _, r, _ in done)
        verdict = "🏆 내가 이기고 있어요!" if me_hit > ai_hit else "🤖 AI가 이기고 있어요" if ai_hit > me_hit else "🤝 비기고 있어요"
        kv_grid([("나의 적중률", f"{me_hit / len(done) * 100:.0f}%", f"<small>{len(done)}번 중 {me_hit}번</small>"),
                 ("AI 적중률", f"{ai_hit / len(done) * 100:.0f}%", f"<small>{verdict}</small>")])
    rows = []
    for v, ok, r, left in res[:20]:
        me, ai = ("📈" if v[5] == "u" else "📉"), ("📈" if v[6] >= 50 else "📉")
        if ok:
            real = "📈" if r > 0 else "📉"
            res_txt = f"{real} {r * 100:+.1f}% · 나 {'✅' if me == real else '❌'} AI {'✅' if ai == real else '❌'}"
        else:
            res_txt = f"⏳ {left}거래일 남음 ({r * 100:+.1f}%)" if left else f"⏳ 오늘 장 마감 후 결과 ({r * 100:+.1f}%)"
        rows.append({"날짜": v[3][5:].replace("-", "/"), "종목": f"{v[1]} ({v[2]}일)", "나": me, "AI": f"{ai}{v[6]}%",
                     "결과": res_txt})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption("찍은 날 가격과 기간(1·5·20거래일)이 지난 날 종가를 비교해요. 같은 종목·기간은 하루에 한 번만 찍을 수 있어요.")


def halted_view(p, close, news):
    """거래정지 종목: 확률은 보여 주지 않고 차트·뉴스·기업 정보만."""
    st.markdown('<div class="hero"><div class="big">⛔ 거래정지</div>'
                '<small>지금 거래가 멈춘 종목이라 AI 확률을 보여 드리지 않아요. 거래가 다시 시작되면 다시 계산해요.</small></div>',
                unsafe_allow_html=True)
    h = p.history.iloc[-252:]
    fig = go.Figure(go.Candlestick(x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
                                   increasing_line_color="#e5484d", decreasing_line_color="#3b82f6", name="가격"))
    fig.update_layout(height=280, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10, l=10, r=10),
                      xaxis=dict(tickformat="%y.%m"))
    st.plotly_chart(fig, use_container_width=True)
    tab_n, tab_i = st.tabs(["📰 뉴스", "🏢 기업 정보"])
    with tab_n:
        if not news:
            st.info("최근 7일 동안 이 종목 관련 뉴스가 없어요.")
        for n in (news or [])[:10]:
            st.markdown(f"[{md_escape(n['title'])}]({n['link']})  \n"
                        f":gray[{md_escape(n['source'])} · {n['when']:%m/%d %H:%M}]")
        st.caption("거래정지 이유(공시·감사의견 등)는 뉴스나 전자공시(DART)에서 확인해 보세요.")
    with tab_i:
        comp = cached_company(p.ticker)
        if comp["info"]:
            kv_grid([(k, v, "") for k, v in comp["info"].items()])
        else:
            st.info("지금은 기본 정보를 불러오지 못했어요.")


def analysis_tab():
    with st.container(key="row_query"):
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        query = c1.text_input("종목 검색", placeholder="🔍 삼성전자, 005930, NVDA", key="query",
                              label_visibility="collapsed")
        c2.button("확인", key="query_ok")  # 누르면 입력한 글자로 바로 검색 (엔터와 같음)
    horizon = st.segmented_control("예측 기간", [1, 5, 20], key="horizon", on_change=keep_horizon,
                                   format_func=lambda d: f"{d}일 뒤", label_visibility="collapsed") or 5
    recent = st.session_state.get("recent", [])
    if recent:
        st.pills("🕘 최근 본 종목", [c for c, _ in recent], key="recent_pick", on_change=pick_recent,
                 format_func=lambda c: dict(recent).get(c, c))

    if not query:
        home_fav(horizon)
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

    try:
        with st.spinner(f"{ticker} AI 분석 중..."):
            c = core(ticker, name, horizon)
    except Exception as e:
        st.error(f"분석 실패: {e}")
        return
    p, close, last, news, final, t, done = c["p"], c["close"], c["last"], c["news"], c["final"], c["t"], c["done"]
    hit_rate, top, mood, short = c["hit_rate"], c["top"], c["mood"], c["short"]
    code = code_of(p.ticker)
    remember(code, short)
    stop = is_halted(p.ticker)
    up = final * 100
    edge = (p.cv_accuracy - p.cv_baseline_accuracy) * 100

    # ① 종목 이름 + 현재가 (한 줄)
    cls = "upc" if c["chg"] > 0 else "dnc" if c["chg"] < 0 else ""
    st.markdown(f"""<div class="sh"><div><b>{short}</b><span>{p.ticker}</span></div>
<div class="pr"><b>{money(last, p.currency)}</b><i class="{cls}">{c['chg'] * 100:+.2f}%</i><span>{c['when_txt']}</span></div></div>""",
                unsafe_allow_html=True)

    # 관심 종목 · 공유 버튼 (한 줄에 나란히)
    is_fav = any(x == code for x, _ in st.session_state.get("fav", []))
    with st.container(key="actions"):
        a1, a2 = st.columns(2)
        a1.button("★ 관심 종목 해제" if is_fav else "☆ 관심 종목 추가", key="fav_btn", on_click=toggle_fav,
                  args=(code, short), use_container_width=True)
        with a2:
            share_button(code, horizon, f"스톡시그널 - {short} AI 분석",
                         f"{short} 거래정지 중 (스톡시그널)" if stop else
                         f"{short} {horizon}거래일 뒤 오를 확률 {up:.0f}% (스톡시그널 AI 분석)")

    w = warn_words(p.ticker)
    if w:
        st.markdown(f'<div class="warn">⚠️ <b>{" · ".join(w)}</b> 종목이에요. 한국거래소가 위험하다고 알린 종목이라 '
                    '값이 크게 흔들리거나 상장폐지될 수 있어요. 특히 조심하세요.</div>', unsafe_allow_html=True)

    if stop:
        halted_view(p, close, news)
        return

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
    kv3 = [("최근 적중률", f"{hit_rate:.0f}%" if hit_rate == hit_rate else "-"),
           ("적중률 순위", f"상위 {top}%" if top else "-"),
           ("뉴스", mood.split(" ", 1)[-1] if news else "없음")]
    cells = "".join(f"<div><small>{k}</small><b>{v}</b></div>" for k, v in kv3)
    st.markdown(f'<div class="kv kv3">{cells}</div>', unsafe_allow_html=True)
    fl = c["follow"]
    if fl:
        s_cls = "upc" if fl["strat_ret"] > 0 else "dnc"
        st.markdown(f'<div class="follow">🧪 최근 1년 <b>AI 따라 했다면 <span class="{s_cls}">{fl["strat_ret"] * 100:+.1f}%</span></b>'
                    f' · 그냥 갖고 있었다면 {fl["hold_ret"] * 100:+.1f}% <small>(자세히: AI 성적 탭)</small></div>',
                    unsafe_allow_html=True)

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

    vote_box(p.ticker, short, p.horizon, last, up)

    # ⑤ 자세한 내용은 탭으로 나눠서
    tab_n, tab_i, tab_t, tab_d = st.tabs(["📰 뉴스", "🏢 기업 정보", "🤖 AI 성적", "🧮 상세 분석"])

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
        comp = cached_company(p.ticker)
        st.markdown("**🏢 기본 정보**")
        if comp["info"]:
            kv_grid([(k, v, "") for k, v in comp["info"].items()])
        else:
            st.info("지금은 기본 정보를 불러오지 못했어요.")

        st.markdown("**🎯 증권사 목표주가**")
        cons = comp.get("consensus")
        if cons:
            gap = cons["target"] / last - 1
            g_cls = "upc" if gap > 0 else "dnc"
            n_txt = f" · 증권사 {cons['n']}곳" if cons.get("n") else ""
            kv_grid([("평균 목표주가", money(cons["target"], p.currency), f"<small>{cons['date'] or cons['src']}</small>"),
                     ("현재가 대비", f'<span class="{g_cls}">{gap * 100:+.1f}%</span>',
                      f"<small>{'더 오를 여지' if gap > 0 else '이미 목표 넘음'}</small>"),
                     ("투자의견", cons["opinion"], f"<small>평균{n_txt}</small>")])
            st.caption(f"증권사 애널리스트들이 낸 목표주가의 평균이에요({cons['src']}). 목표주가는 자주 바뀌고 "
                       "틀리는 경우도 많으니 참고만 하세요.")
        else:
            st.caption("이 종목은 증권사 목표주가가 없어요. (작은 회사는 분석하는 증권사가 없는 경우가 많아요)")

        st.markdown("**🧑‍🤝‍🧑 외국인·기관 매매**")
        flow = comp["flow"]
        if flow is None or flow.empty:
            st.caption("한국 종목만 볼 수 있어요." if not p.ticker.endswith((".KS", ".KQ"))
                       else "지금은 매매 정보를 불러오지 못했어요.")
        else:
            cells = []
            for who in ("외국인", "기관", "개인"):
                s = flow[who].sum()
                cls = "upc" if s > 0 else "dnc" if s < 0 else ""
                word = "샀어요" if s > 0 else "팔았어요" if s < 0 else "그대로"
                cells.append((who, f'<span class="{cls}">{s:+,.0f}주</span>', f"<small>최근 {len(flow)}거래일 합계 · {word}</small>"))
            kv_grid(cells)
            fig = go.Figure()
            for who, color in (("외국인", "#e5484d"), ("기관", "#f59e0b")):
                fig.add_bar(x=flow["날짜"].dt.strftime("%m/%d"), y=flow[who], name=who, marker_color=color)
            fig.update_layout(height=220, barmode="group", margin=dict(t=10, b=10, l=10, r=10),
                              legend=dict(orientation="h", y=1.15, x=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("+는 산 주식이 더 많다는 뜻(순매수), -는 판 주식이 더 많다는 뜻(순매도)이에요. 네이버 증권 자료예요.")

        st.markdown("**📅 실적 발표·배당 일정**")
        sched = comp["schedule"]
        if not sched:
            st.caption("알려진 일정이 없어요.")
        else:
            today = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
            for label, day, note in sched:
                dd = (day.normalize() - today).days
                dtxt = "오늘" if dd == 0 else f"D-{dd}" if dd > 0 else "지난 일정"
                st.markdown(f"{label} · **{day:%Y-%m-%d}** ({dtxt}){' · ' + note if note else ''}")
            st.caption("실적 발표 '예상일'은 야후 파이낸스 추정이라 실제와 다를 수 있어요.")
        st.caption("기본 정보·매매는 한국 종목은 네이버 증권, 해외 종목은 야후 파이낸스 자료예요. 적자 기업은 PER이 '-'로 나와요.")

    with tab_t:
        st.markdown("**📈 AI 확률 변화 (최근 60거래일)**")
        tr = t.iloc[-60:]
        fig = go.Figure()
        fig.add_scatter(x=tr.index, y=tr["prob_up"] * 100, mode="lines+markers", name="상승 확률",
                        line=dict(color="#e5484d", width=2), marker=dict(size=4))
        fig.add_scatter(x=tr.index, y=close.reindex(tr.index), name="주가", yaxis="y2",
                        line=dict(color="rgba(128,128,128,.6)", width=1))
        fig.add_hline(y=50, line_dash="dot", line_color="gray")
        fig.update_layout(height=240, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=1.15, x=0),
                          yaxis=dict(title="확률(%)", range=[0, 100]), xaxis=dict(tickformat="%m/%d"),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("빨간 선이 50% 위면 AI가 '상승', 아래면 '하락'으로 본 날이에요. 회색 선은 주가예요. "
                   "그날까지의 데이터로만 학습한 모델의 확률이에요(뉴스 미반영).")

        st.markdown("**🧪 AI 따라 했다면? (최근 1년)**")
        if fl:
            s_cls = "upc" if fl["strat_ret"] > 0 else "dnc"
            h_cls = "upc" if fl["hold_ret"] > 0 else "dnc"
            kv_grid([("AI 따라 했다면", f'<span class="{s_cls}">{fl["strat_ret"] * 100:+.1f}%</span>',
                      f'<small>{fl["n"]}번 중 {fl["n_buy"]}번 삼</small>'),
                     ("그냥 갖고 있었다면", f'<span class="{h_cls}">{fl["hold_ret"] * 100:+.1f}%</span>', "<small>비교용</small>")])
            fig = go.Figure()
            fig.add_scatter(x=fl["x"], y=(fl["strat"] - 1) * 100, name="AI 따라 하기", line=dict(color="#e5484d", width=2))
            fig.add_scatter(x=fl["x"], y=(fl["hold"] - 1) * 100, name="그냥 보유", line=dict(color="gray", width=1))
            fig.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=1.15, x=0),
                              yaxis=dict(title="수익률(%)"), xaxis=dict(tickformat="%y.%m"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{p.horizon}거래일마다 AI가 '상승'(50% 이상)이라고 하면 사서 {p.horizon}거래일 들고 있고, "
                       "'하락'이면 쉬었다고 가정해 계산했어요. 수수료·세금은 빼지 않았고, 과거 결과일 뿐 앞으로를 보장하지 않아요.")
        else:
            st.caption("계산할 기록이 아직 부족해요.")

        st.markdown("**🎯 적중 기록**")
        hits = int(done["hit"].sum())
        kv_grid([(f"최근 {len(done)}번 적중률", f"{hit_rate:.1f}%", f"<small>성공 {hits} · 실패 {len(done) - hits}</small>"),
                 ("'항상 상승' 찍기", f"{(done['ret'] > 0).mean() * 100:.1f}%", "<small>비교용 기준</small>")])
        if top:
            st.caption(f"🏅 적중률 순위: 같은 {p.horizon}거래일 예측 기준 코스피·코스닥 {c['pool_n']:,}개 종목 중 상위 {top}% "
                       "(매일 장 마감 후 계산한 순위 데이터와 비교)")
        rows = []
        for d, r in t.iloc[-11:-1][::-1].iterrows():
            verdict = ("✅" if r["hit"] else "❌") if r["done"] else ("⏳✅" if r["hit"] else "⏳❌")
            rows.append({"날짜": f"{d:%m/%d}", "AI 예측": f"{'📈' if r['prob_up'] >= 0.5 else '📉'} {r['prob_up'] * 100:.0f}%",
                         "실제": f"{r['ret'] * 100:+.1f}%", "결과": verdict})
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(f"최근 10번의 예측이에요. 상승 확률 50% 이상이면 '상승' 예측이고, 예측한 날 종가와 {p.horizon}거래일 뒤 "
                   "종가를 비교해요. ⏳는 아직 기간이 안 끝나 현재가로 중간 판정한 것이에요. 뉴스를 빼고 차트 AI 기준이에요.")

    with tab_d:
        st.markdown("**🔍 과거 검증 결과**")
        kv_grid([("과거 검증 정확도", f"{p.cv_accuracy * 100:.1f}%", ""),
                 ("단순 찍기 정확도", f"{p.cv_baseline_accuracy * 100:.1f}%", f"<small>차이 {edge:+.1f}%p</small>"),
                 ("AUC", f"{p.cv_auc:.3f}", "<small>0.5 = 동전 던지기</small>"),
                 (("학습한 데이터", f"{p.n_samples / 1e4:,.0f}만 일", "<small>전 종목</small>")
                  if getattr(p, "model", "") == "pool" else ("학습한 거래일", f"{p.n_samples:,}일", ""))])
        if edge < 2 or p.cv_auc < 0.53:
            st.warning("이 종목에서는 모델이 '항상 같은 방향을 찍는' 단순 기준선보다 뚜렷하게 낫지 않았어요.")
        if getattr(p, "model", "") == "pool":
            st.caption("🧠 한국 종목은 코스피·코스닥 전 종목의 10년치 데이터를 AI 하나가 함께 학습해요(매일 장 마감 후 다시 학습). "
                       "위 성적은 최근 1년을 4구간으로 나눠, 각 구간 전까지의 데이터로만 학습한 AI가 이 종목을 맞혀 본 결과예요.")
        else:
            st.caption("과거 데이터를 시간순으로 나눠, 과거로 학습하고 그 다음 구간을 맞혀 본 결과(워크포워드 검증)예요.")
        if set(cols_for(p.horizon)) <= set(p.latest_features.index):
            indicator_table(p)
            return
        st.markdown("**🧮 현재 지표와 모델이 중요하게 본 요인**")
        f = p.latest_features
        pct_keys = {"ret_1", "ret_5", "ret_20", "sma5_ratio", "sma20_ratio", "sma60_ratio", "vol_20", "dist_52w_high"}
        rows = []
        for k in p.importances.index[::-1]:
            v = f[k]
            val = f"{v * 100:+.2f}%" if k in pct_keys else f"{v:.2f}" if k != "macd_hist" else f"{v * 100:+.3f}%"
            rows.append({"지표": FEATURE_LABELS[k], "현재 값": val, "중요도": f"{p.importances[k] * 100:.1f}%"})
        st.dataframe(rows, hide_index=True, use_container_width=True)


PCT_HINTS = ("mom_", "rs_", "_gap", "_slope", "roc_", "atr_pct", "rvol_", "bb_width", "macd_h", "ichi_")


def _fmt(col, v):
    if v != v:
        return "-"
    if col.startswith(PCT_HINTS) or any(h in col for h in PCT_HINTS):
        return f"{v * 100:+.1f}%" if col != "macd_h" else f"{v * 100:+.2f}%"
    if col in ("hhll", "double_pat", "hs_pat", "cross_5_20", "cross_20_60"):
        return f"{v:+.0f}"
    return f"{v:.2f}"


def indicator_table(p):
    """1·5·20일 뒤 예측: 지표 50가지 표 (순위·유형·내용·현재 값·AI 중요도)."""
    st.markdown(f"**🧮 {p.horizon}일 뒤 AI 확률 분석 지표 50가지**")
    imp = p.importances
    try:  # 학습에 안 쓴 지표도 현재 값은 보여 주려고 전부 다시 계산
        f = build_1d(p.history, load_index(market_index(p.ticker))).iloc[-1]
    except Exception:
        f = p.latest_features
    stt = cached_status().get(_kr_code(p.ticker) or "", {})
    fin = fundamentals_now(stt.get("fin"))
    rows = []
    table = indicators_for(p.horizon)
    for rank, name, typ, desc, cols in table:
        if cols:
            val = " · ".join(_fmt(c, f.get(c, float("nan"))) for c in cols)
            used = (f"✅ {imp.reindex(cols).sum() * 100:.1f}%" if set(cols) <= set(imp.index)
                    else "➖ 이 기간엔 안 씀")
        else:
            val = fin.get(name, "자료 없음")
            used = "참고만" if name in fin else "✖"
        rows.append({"순위": rank, "지표": name, "유형": typ, "분석 내용": desc, "현재 값": val, "AI 반영(중요도)": used})
    st.dataframe(rows, hide_index=True, use_container_width=True, height=420)
    top = sorted(((imp.reindex(cols).sum(), name) for _, name, _, _, cols in table
                  if cols and set(cols) <= set(imp.index)), reverse=True)[:3]
    who = "전 종목에서" if getattr(p, "model", "") == "pool" else "이 종목에서"
    st.caption(f"✅ = AI가 학습한 지표(괄호 안은 {who} AI가 중요하게 본 정도) · 참고만 = 지금 값만 보여 줌 · "
               "✖ = 자료가 없어 계산 못 함"
               + (" · ➖ = 20일 뒤는 순위 상위 10개 지표만 학습해서 이번엔 안 씀(시험에서 이쪽이 더 정확했어요)"
                  if p.horizon == 20 else "")
               + f". {who} AI가 가장 중요하게 본 것: "
               + ", ".join(n for _, n in top) + ".")
    st.caption("기본적 지표(실적·재무)는 과거 날짜별 기록을 무료로 구할 수 없어 학습에는 못 넣었어요. "
               "업종 지수 자료가 없어 '시장·업종'은 시장(코스피·코스닥) 기준이고, 엘리어트 파동은 정해진 계산식이 없어 뺐어요.")


def compare_tab():
    st.caption("두 종목을 넣으면 AI 확률·적중률·기본 정보를 나란히 비교해요.")
    q1 = st.text_input("첫 번째 종목", key="cmp_a", placeholder="예: 삼성전자")
    q2 = st.text_input("두 번째 종목", key="cmp_b", placeholder="예: SK하이닉스")
    st.button("확인 (비교하기)", key="cmp_ok", type="primary", use_container_width=True)
    horizon = st.segmented_control("예측 기간", [1, 5, 20], key="horizon", on_change=keep_horizon,
                                   format_func=lambda d: f"{d}일 뒤", label_visibility="collapsed") or 5
    if not (q1 and q2):
        return
    picks = []
    for q in (q1, q2):
        cand = cached_search(q)
        if not cand:
            st.warning(f"'{q}' 검색 결과가 없어요.")
            return
        picks.append(cand[0])
    try:
        with st.spinner("두 종목을 분석하는 중... (처음엔 30초쯤 걸려요)"):
            cs = [core(tk, nm, horizon) for tk, nm in picks]
            comps = [cached_company(x["p"].ticker) for x in cs]
    except Exception as e:
        st.error(f"분석 실패: {e}")
        return

    def pct(v, digits=1):
        return f"{v:+.{digits}f}%" if v == v else "-"

    stops = [is_halted(x["p"].ticker) for x in cs]

    rows = [  # (항목, A 값, B 값, 비교할 숫자 A, B, 큰 쪽이 좋은지)
        ("현재가", *[f"{money(x['last'], x['p'].currency)}<br><small>{pct(x['chg'] * 100, 2)}</small>" for x in cs], None, None),
        (f"{horizon}일 뒤 오를 확률", *["⛔ 거래정지" if s else f"{x['final'] * 100:.1f}%" for x, s in zip(cs, stops)],
         *[float("nan") if s else x["final"] for x, s in zip(cs, stops)]),
        ("최근 적중률", *[f"{x['hit_rate']:.0f}%" if x["hit_rate"] == x["hit_rate"] else "-" for x in cs],
         cs[0]["hit_rate"], cs[1]["hit_rate"]),
        ("적중률 순위", *[f"상위 {x['top']}%" if x["top"] else "-" for x in cs],
         -(cs[0]["top"] or 999), -(cs[1]["top"] or 999)),
        ("뉴스 분위기", *[x["mood"] for x in cs], cs[0]["score"], cs[1]["score"]),
        ("AI 따라 했다면(1년)", *[pct(x["follow"]["strat_ret"] * 100) if x["follow"] else "-" for x in cs],
         *[x["follow"]["strat_ret"] if x["follow"] else float("nan") for x in cs]),
        ("1년 주가 수익률", *[pct(x["year_ret"] * 100) for x in cs], cs[0]["year_ret"], cs[1]["year_ret"]),
    ]
    gaps = [cp["consensus"]["target"] / x["last"] - 1 if cp.get("consensus") else float("nan")
            for cp, x in zip(comps, cs)]
    rows.append(("목표주가까지", *[pct(g * 100) for g in gaps], *gaps))
    for k in ("시가총액", "PER", "PBR", "배당수익률"):
        rows.append((k, *[(cp["info"] or {}).get(k, "-") for cp in comps], None, None))
    html = [f'<tr><th></th><th>{cs[0]["short"]}</th><th>{cs[1]["short"]}</th></tr>']
    for label, va, vb, na, nb in rows:
        wa = wb = ""
        if na is not None and na == na and nb == nb and na != nb:
            wa, wb = ("win", "lose") if na > nb else ("lose", "win")
        html.append(f'<tr><td>{label}</td><td class="{wa}">{va}</td><td class="{wb}">{vb}</td></tr>')
    st.markdown(f'<table class="cmp">{"".join(html)}</table>', unsafe_allow_html=True)
    st.caption("🔴 빨간 굵은 글씨 = 둘 중 큰(더 나은) 쪽 · 🔵 파란 글씨 = 작은 쪽 (순위는 더 높은 쪽이 빨강). 확률은 뉴스까지 반영한 값이고, 과거 성적이 앞으로를 보장하지는 않아요.")

    fig = go.Figure()
    for x, color in zip(cs, ("#e5484d", "#3b82f6")):
        y = x["close"][x["close"].index >= x["close"].index[-1] - pd.Timedelta(days=365)]
        fig.add_scatter(x=y.index, y=(y / y.iloc[0] - 1) * 100, name=x["short"], line=dict(color=color, width=2))
    fig.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10), title="최근 1년 주가 등락률(%)",
                      legend=dict(orientation="h", y=-0.15), xaxis=dict(tickformat="%y.%m"))
    st.plotly_chart(fig, use_container_width=True)
    b1, b2 = st.columns(2)
    for col, x in zip((b1, b2), cs):
        col.button(f"🔍 {x['short']} 자세히", key=f"cmp_open_{x['short']}", on_click=open_stock,
                   args=(code_of(x["p"].ticker), horizon), use_container_width=True)


# 앱을 처음 열 때: 앱 껍데기가 주소에 붙여 준 관심 종목·최근 본 종목·공유 링크 종목을 읽음
def _parse(text):
    return [tuple(x.split("~", 1)) for x in (text or "").split(",") if "~" in x]


if "_boot" not in st.session_state:
    st.session_state._boot = True
    qp = st.query_params
    st.session_state.fav = _parse(qp.get("fav"))[:20]
    st.session_state.recent = _parse(qp.get("recent"))[:6]
    hold = []
    for x in (qp.get("hold") or "").split(","):
        parts = x.split("~")
        if len(parts) == 4:
            try:
                hold.append((parts[0], parts[1], float(parts[2]), float(parts[3])))
            except ValueError:
                pass
    st.session_state.hold = hold[:30]
    votes = []
    for x in (qp.get("votes") or "").split(","):
        parts = x.split("~")
        if len(parts) == 7:
            try:
                votes.append((parts[0], parts[1], int(parts[2]), parts[3], float(parts[4]), parts[5], int(parts[6])))
            except ValueError:
                pass
    st.session_state.votes = votes[:60]
    if qp.get("s"):  # 공유 링크로 들어왔을 때 그 종목을 바로 분석
        st.session_state.query = qp.get("s")
        st.session_state.horizon = int(qp.get("h")) if qp.get("h") in ("1", "5", "20") else 5

# 다른 화면에 가 있어도 검색어·예측 기간이 지워지지 않게 보관
for k in ("query", "horizon", "cmp_a", "cmp_b", "sc_prob", "sc_hit", "sc_mkt", "sc_price", "sc_safe"):
    if k in st.session_state:
        st.session_state[k] = st.session_state[k]
st.session_state.setdefault("horizon", 5)
st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("page_last", PAGES[0])
st.session_state.setdefault("sc_prob", 60)
st.session_state.setdefault("sc_hit", 30)
st.session_state.setdefault("sc_safe", True)


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
    prev = order_of(load_prev(), horizon, "up")
    for i, r in enumerate(d.itertuples(), 1):
        st.button(f"{i}. {r.name} · **{r.prob_up * 100:.1f}%**{move_txt(prev, r.ticker, i, 5)}{warn_tag(r.ticker)}",
                  key=f"rk_home_{i}",
                  on_click=open_stock, args=(r.ticker[:6], horizon), use_container_width=True)
    st.button("전체 순위 보기 ›", key="all_rank", on_click=go_page, args=(P_PROB,))
    st.caption(f"{meta.get('date', '')} 장 마감 기준 · 뉴스 미반영 차트 AI 확률 · 누르면 자세한 분석으로 이동")
    scorecard_box(compact=True)


def set_fav_h():
    """관심 종목 칸의 기간 버튼을 누르면 앱 전체 예측 기간도 같이 바꿈 (선택 해제는 무시)."""
    if st.session_state.fav_h:
        st.session_state.horizon = st.session_state.horizon_last = st.session_state.fav_h


def home_fav(horizon):
    """첫 화면: 관심 종목과 오늘의 AI 확률 (한국 종목은 매일 계산한 순위 데이터에서)."""
    fav = st.session_state.get("fav", [])
    st.markdown(f"#### ⭐ 관심 종목 · {horizon}일 뒤")
    # 관심 종목 칸 안의 예측 기간 버튼 (위쪽 기간 버튼과 같은 값을 씀)
    st.session_state.fav_h = horizon
    st.segmented_control("관심 종목 예측 기간", [1, 5, 20], key="fav_h", on_change=set_fav_h,
                         format_func=lambda d: f"{d}일 뒤", label_visibility="collapsed")
    if not fav:
        st.caption("아직 관심 종목이 없어요. 종목을 검색한 뒤 **☆ 관심 종목 추가**를 누르면 "
                   "여기(첫 화면)와 💼 내 종목 화면에 모여요.")
        return
    df, _ = load_rankings()
    prev = load_prev()

    def prob_of(d, code):
        if d is None:
            return None
        m = d[(d["ticker"].str[:6] == code) & (d["horizon"] == horizon)]
        return float(m["prob_up"].iloc[0]) if len(m) else None

    rows = []
    for code, name in fav:
        now, old = prob_of(df, code), prob_of(prev, code)
        diff = (now - old) * 100 if now is not None and old is not None else None
        rows.append((code, name, now, diff))
    # 어제보다 확률이 크게 바뀐 종목을 위로
    rows.sort(key=lambda x: -abs(x[3]) if x[3] is not None else 0)
    for i, (code, name, now, diff) in enumerate(rows):
        if is_halted(code):
            label = f"{name} · ⛔ 거래정지"
        elif now is None:
            label = f"{name} · 눌러서 분석{warn_tag(code)}"
        else:
            label = f"{name} · **{now * 100:.1f}%**"
            if diff is not None:
                arrow = "▲" if diff > 0.05 else "▼" if diff < -0.05 else ""
                label += f" · 어제보다 {arrow}{abs(diff):.1f}%p" if arrow else " · 어제와 같음"
                if abs(diff) >= 5:
                    label = "🔥 " + label
            label += warn_tag(code)
        st.button(label, key=f"rk_fav_{i}", on_click=open_stock, args=(code, horizon), use_container_width=True)
    if prev is not None:
        st.caption("어제보다 확률이 많이 바뀐 종목이 위로 와요. 🔥 = 5%p 이상 변화")


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


def screener_tab():
    df, meta = load_rankings()
    if df is None or df.empty:
        st.info("순위 데이터를 준비하고 있어요. 매일 장 마감 뒤 오후 6시쯤 새로 계산됩니다.")
        return
    st.caption("원하는 조건을 고르면 코스피·코스닥 전 종목에서 맞는 종목을 찾아 드려요.")
    horizon = st.segmented_control("예측 기간", [1, 5, 20], key="horizon", on_change=keep_horizon,
                                   format_func=lambda d: f"{d}일 뒤", label_visibility="collapsed") or 5
    min_prob = st.slider("📈 상승 확률 몇 % 이상?", 50, 80, step=1, key="sc_prob", format="%d%%")
    hit_top = st.select_slider("🎯 적중률 순위", [100, 50, 30, 10], key="sc_hit",
                               format_func=lambda v: "상관없음" if v == 100 else f"상위 {v}% 안")
    c1, c2 = st.columns(2)
    market = c1.selectbox("시장", ["전체", "코스피", "코스닥"], key="sc_mkt")
    price = c2.selectbox("주가", ["전체", "1만 원 미만", "1~5만 원", "5~20만 원", "20만 원 이상"], key="sc_price")
    safe = st.checkbox("⚠️ 관리종목·투자주의/경고/위험 종목 빼기", key="sc_safe")

    d = df[df["horizon"] == horizon].copy()
    pool = d[d["n"] >= 60]
    d["hit_top"] = pool["hit_rate"].rank(ascending=False, pct=True).reindex(d.index) * 100
    d = d[d["prob_up"] >= min_prob / 100]
    if hit_top < 100:
        d = d[d["hit_top"] <= hit_top]
    if market != "전체":
        d = d[d["market"] == ("KOSPI" if market == "코스피" else "KOSDAQ")]
    lo, hi = {"1만 원 미만": (0, 1e4), "1~5만 원": (1e4, 5e4), "5~20만 원": (5e4, 2e5),
              "20만 원 이상": (2e5, float("inf"))}.get(price, (0, float("inf")))
    d = d[(d["close"] >= lo) & (d["close"] < hi)].sort_values("prob_up", ascending=False)
    if safe:
        d = d[[not warn_words(t) for t in d["ticker"]]]

    st.markdown(f"#### 🔎 조건에 맞는 종목 {len(d):,}개")
    if d.empty:
        st.info("조건에 맞는 종목이 없어요. 상승 확률을 낮추거나 적중률 조건을 풀어 보세요.")
    for i, r in enumerate(d.head(30).itertuples(), 1):
        hit = f" · 적중 {r.hit_rate * 100:.0f}%" if r.n >= 60 else ""
        st.button(f"{i}. {r.name} ({r.market}) · 상승 **{r.prob_up * 100:.1f}%**{hit} · {r.close:,.0f}원"
                  f"{warn_tag(r.ticker)}",
                  key=f"rk_sc_{i}", on_click=open_stock, args=(r.ticker[:6], horizon), use_container_width=True)
    if len(d) > 30:
        st.caption(f"상승 확률이 높은 순서로 30개만 보여 드려요. (나머지 {len(d) - 30:,}개는 조건을 좁혀 보세요)")
    st.caption(f"📅 {meta.get('date', '')} 장 마감 기준 · 뉴스 미반영 차트 AI 확률 · 적중률 순위는 최근 60번 예측 기준. "
               "조건에 맞는다고 오른다는 보장은 없어요.")


def unit(ticker):
    """보유 종목 금액 표시용 통화."""
    return "KRW" if ticker.endswith((".KS", ".KQ")) else "USD" if "." not in ticker else ""


def add_hold():
    q = (st.session_state.get("h_q") or "").strip()
    price, qty = st.session_state.get("h_price"), st.session_state.get("h_qty")
    if not q or not price or not qty:
        st.session_state.h_msg = "종목, 산 가격, 수량을 모두 넣어 주세요."
        return
    cand = cached_search(q)
    if not cand:
        st.session_state.h_msg = f"'{q}' 검색 결과가 없어요."
        return
    ticker, name = cand[0]
    short = name.split(" (")[0]
    hold = st.session_state.get("hold", [])
    old = next((x for x in hold if x[0] == ticker), None)
    if old:  # 같은 종목을 더 샀으면 평균 단가로 합침
        tq = old[3] + qty
        price, qty = (old[2] * old[3] + price * qty) / tq, tq
    st.session_state.hold = [(ticker, short, float(price), float(qty))] + [x for x in hold if x[0] != ticker]
    st.session_state.hold = st.session_state.hold[:30]
    st.session_state.h_msg = f"✅ {short}을(를) 추가했어요."
    st.session_state.h_q = ""
    st.session_state.h_price = st.session_state.h_qty = None


def del_hold():
    t = st.session_state.get("h_del")
    st.session_state.hold = [x for x in st.session_state.get("hold", []) if x[0] != t]


def holdings_tab():
    st.caption("내가 산 종목을 넣어 두면 지금 수익률과 AI 의견을 한눈에 볼 수 있어요. (이 휴대폰에만 저장돼요)")
    horizon = st.session_state.get("horizon") or 5
    hold = st.session_state.get("hold", [])
    home_fav(horizon)
    st.markdown("#### 💼 내가 산 종목")
    df, _ = load_rankings()
    if hold:
        rows, tot_buy, tot_now, foreign = [], 0.0, 0.0, False
        for ticker, name, price, qty in hold:
            lp = cached_live(ticker)
            now = lp[0] if lp else None
            if now is None and df is not None:
                m = df[df["ticker"] == ticker]
                now = float(m["close"].iloc[0]) if len(m) else None
            prob = None
            if df is not None:
                m = df[(df["ticker"] == ticker) & (df["horizon"] == horizon)]
                prob = float(m["prob_up"].iloc[0]) if len(m) else None
            rows.append((ticker, name, price, qty, now, prob))
            if now is not None and unit(ticker) == "KRW":
                tot_buy += price * qty
                tot_now += now * qty
            elif unit(ticker) != "KRW":
                foreign = True
        if tot_buy:
            gain = tot_now - tot_buy
            cls = "upc" if gain > 0 else "dnc" if gain < 0 else ""
            kv_grid([("지금 가치", f"{tot_now:,.0f}원", f"<small>산 금액 {tot_buy:,.0f}원</small>"),
                     ("수익", f'<span class="{cls}">{gain:+,.0f}원</span>',
                      f'<small class="{cls}">{gain / tot_buy * 100:+.2f}%</small>')])
            if foreign:
                st.caption("해외 종목은 통화가 달라서 합계에서 빠져요.")
        st.markdown(f"#### 💼 내 종목 {len(rows)}개 · AI {horizon}일 뒤")
        for i, (ticker, name, price, qty, now, prob) in enumerate(rows):
            cur = unit(ticker)
            if now is None:
                body = f"{name} · 현재가를 못 불러왔어요"
            else:
                r = now / price - 1
                arrow = "🔺" if r > 0 else "🔹" if r < 0 else ""
                body = (f"{name} · {arrow} **{r * 100:+.1f}%** ({(now - price) * qty:+,.{0 if cur == 'KRW' else 2}f}"
                        f"{'원' if cur == 'KRW' else ' ' + cur}) · {money(now, cur)}")
            if is_halted(ticker):
                body += " · ⛔ 거래정지"
            else:
                body += f" · AI 상승 {prob * 100:.0f}%" if prob is not None else " · AI는 눌러서 확인"
                body += warn_tag(ticker)
            st.button(body, key=f"rk_hold_{i}", on_click=open_stock, args=(code_of(ticker), horizon),
                      use_container_width=True)
        st.caption(f"🔺 수익 · 🔹 손실 · 수수료·세금은 빼지 않았어요. AI 확률은 매일 장 마감 후 계산한 {horizon}일 뒤 확률이에요"
                   "(뉴스 미반영). 누르면 자세한 분석으로 이동해요.")
    else:
        st.info("아직 넣은 종목이 없어요. 아래에서 추가해 보세요.")

    with st.expander("➕ 종목 추가", expanded=not hold):
        st.text_input("종목 이름이나 코드", key="h_q", placeholder="예: 삼성전자")
        c1, c2 = st.columns(2)
        c1.number_input("산 가격 (1주)", min_value=0.0, value=None, step=100.0, key="h_price", placeholder="예: 70000")
        c2.number_input("수량 (주)", min_value=0.0, value=None, step=1.0, key="h_qty", placeholder="예: 10")
        st.button("추가하기", key="h_add", on_click=add_hold, type="primary", use_container_width=True)
        if st.session_state.get("h_msg"):
            st.caption(st.session_state.pop("h_msg"))
    if hold:
        with st.expander("🗑 종목 빼기"):
            st.selectbox("뺄 종목", [x[0] for x in hold], key="h_del",
                         format_func=lambda t: next(f"{x[1]} ({x[3]:.10g}주)" for x in hold if x[0] == t))
            st.button("빼기", key="h_del_btn", on_click=del_hold, use_container_width=True)
    st.divider()
    vote_section()


def keep_page():
    # 이미 선택된 버튼을 한 번 더 눌러 선택이 풀리면 원래 화면으로 되돌림
    if st.session_state.page is None:
        st.session_state.page = st.session_state.page_last
    st.session_state.page_last = st.session_state.page


if st.session_state.page not in PAGES:  # 예전 메뉴 이름이 남아 있을 때
    st.session_state.page = st.session_state.page_last = PAGES[0]


def nav_back():
    """휴대폰 '뒤로 가기': 바로 전에 보던 화면(메뉴·검색한 종목)으로 되돌림."""
    nav = st.session_state.get("nav", [])
    if len(nav) > 1:
        nav.pop()
        page_, query_ = nav[-1]
        st.session_state.page = st.session_state.page_last = page_
        st.session_state.query = query_
        st.session_state.to_top = True


# 지금 보는 화면을 기록 (메뉴나 검색 종목이 바뀔 때마다 한 칸씩)
_nav = st.session_state.setdefault("nav", [])
_cur = (st.session_state.page, st.session_state.get("query") or "")
if not _nav or _nav[-1] != _cur:
    _nav.append(_cur)
    del _nav[:-30]
st.button("뒤로", key="nav_back", on_click=nav_back)
clock = cached_clock()
st.markdown(f'<div class="clock"><b>{clock["label"]}</b> · {clock["note"]}</div>', unsafe_allow_html=True)
page = st.segmented_control("화면", PAGES, key="page", on_change=keep_page,
                            label_visibility="collapsed") or PAGES[0]
if page == P_HOLD:
    holdings_tab()
elif page == P_SCREEN:
    screener_tab()
elif page == P_PROB:
    ranking_tab("prob")
elif page == P_HIT:
    ranking_tab("hit")
elif page == P_CMP:
    compare_tab()
elif page == P_MARKET:
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


def _enc(items):
    return ",".join(f"{c}~{n.replace(',', ' ').replace('~', ' ')}" for c, n in items)


# 관심 종목·최근 본 종목을 앱 껍데기(GitHub Pages)에 저장해 달라고 보냄 → 앱을 껐다 켜도 남음
components.html("<script>window.top.postMessage(" + json.dumps(
    {"type": "stock-ai-store", "fav": _enc(st.session_state.get("fav", [])),
     "recent": _enc(st.session_state.get("recent", [])),
     "hold": ",".join(f"{t}~{n.replace(',', ' ').replace('~', ' ')}~{pr:.10g}~{q:.10g}"
                      for t, n, pr, q in st.session_state.get("hold", [])),
     "votes": ",".join(f"{t}~{n.replace(',', ' ').replace('~', ' ')}~{h}~{d}~{pr:.10g}~{me}~{ai}"
                       for t, n, h, d, pr, me, ai in st.session_state.get("votes", []))}, ensure_ascii=False) + ", '*')</script>", height=0)
# 휴대폰 뒤로 가기 버튼 연결: 화면이 바뀌면 브라우저 기록에 한 칸 추가, 뒤로 가기를 누르면 숨은 '뒤로' 버튼을 눌러 줌
components.html("""<script>
const w = window.parent, n = %d;
if (!w.__stockNav) {
  w.__stockNav = true;
  w.history.replaceState({sn: n}, "");
  // 이 작은 창은 매번 새로 만들어지므로, 감시 코드는 앱 화면(부모 창) 쪽에 심어 둠
  const sc = w.document.createElement("script");
  sc.textContent = `addEventListener("popstate", (e) => {
    const s = e.state && e.state.sn;
    if (typeof s !== "number" || s >= window.__stockN) return;
    window.__stockN = s;
    const b = document.querySelector(".st-key-nav_back button");
    if (b) b.click();
  });`;
  w.document.head.appendChild(sc);
} else {
  const cur = (w.history.state && w.history.state.sn) || 0;
  if (n > cur) w.history.pushState({sn: n}, "");
  else if (n < cur) w.history.replaceState({sn: n}, "");
}
w.__stockN = n;
</script>""" % len(_nav), height=0)
footer()
