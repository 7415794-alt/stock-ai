"""장중 가격 모으기 (GitHub Actions에서 20분마다) → out/live.json → 'live' 브랜치.

새 앱은 AI 확률은 하루 한 번 계산한 파일을 쓰고, 현재가·등락률만 이 파일로 새로 고침.
네이버 증권 전 종목 시세 한 번에 받기(약 3번 요청) + 코스피·코스닥 지수.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

UA = {"User-Agent": "Mozilla/5.0"}


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def main() -> None:
    s = requests.Session()
    s.headers.update(UA)
    stocks = {}
    for page in range(10):
        d = s.get("https://stock.naver.com/api/domestic/market/stock/default",
                  params={"tradeType": "KRX", "marketType": "ALL", "orderType": "marketSum",
                          "startIdx": page, "pageSize": 1000}, timeout=20).json()
        for r in d:
            now = num(r.get("nowPrice"))
            if now is None:
                continue
            stocks[r["itemcode"]] = [int(now) if now == int(now) else now, num(r.get("prevChangeRate")),
                                     int(num(r.get("tradeVolume")) or 0), r.get("tradeStopYn") == "Y"]
        if len(d) < 1000:
            break
    idx = {}
    for k in ("KOSPI", "KOSDAQ"):
        try:
            r = s.get(f"https://m.stock.naver.com/api/index/{k}/basic", timeout=10).json()
            idx[k] = [num(r["closePrice"]), num(r["fluctuationsRatio"]), r.get("marketStatus"), r.get("localTradedAt")]
        except Exception:
            pass
    kst = timezone(timedelta(hours=9))
    out = {"t": datetime.now(kst).strftime("%Y-%m-%d %H:%M"), "idx": idx, "s": stocks}
    Path("out").mkdir(exist_ok=True)
    Path("out/live.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{out['t']} 종목 {len(stocks)}개, 지수 {idx}")


if __name__ == "__main__":
    main()
