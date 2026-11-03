"""한국거래소(KIND)에서 코스피·코스닥 전 종목 목록을 받아 krx_stocks.csv로 저장.

신규 상장 종목을 반영하려면 가끔 다시 실행하세요: python build_stock_list.py
"""
import io
import urllib.request

import pandas as pd

URL = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&marketType={}"
MARKETS = {"stockMkt": ("KOSPI", ".KS"), "kosdaqMkt": ("KOSDAQ", ".KQ")}


def fetch(market_type: str) -> pd.DataFrame:
    req = urllib.request.Request(URL.format(market_type), headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode("euc-kr", errors="replace")
    return pd.read_html(io.StringIO(html))[0]


def main():
    frames = []
    for market_type, (market, suffix) in MARKETS.items():
        df = fetch(market_type)
        code = df["종목코드"].astype(str).str.zfill(6)
        frames.append(pd.DataFrame({"name": df["회사명"], "code": code,
                                    "ticker": code + suffix, "market": market}))
    out = pd.concat(frames).sort_values("name").reset_index(drop=True)
    out.to_csv("krx_stocks.csv", index=False, encoding="utf-8")
    print(f"{len(out)}개 종목 저장 완료 (krx_stocks.csv)")


if __name__ == "__main__":
    main()
