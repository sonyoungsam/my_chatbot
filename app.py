import json
import re
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from ddgs import DDGS
from openai import OpenAI

LLM_BASE_URL = "http://192.168.0.201:18000/v1"
EMBEDDING_BASE_URL = "http://192.168.0.201:18001/v1"
EMBEDDING_MODEL = "dragonkue/bge-m3-ko"
STREAM_UI_MIN_INTERVAL = 0.15  # 초. 화면 갱신 간격 최소치 (DOM 갱신이 너무 잦아 생기는 오류 방지)
LLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

DEFAULT_SYSTEM_PROMPT = """You are a helpful, honest assistant.

- If you do not know the answer, or are not confident, say so plainly instead of guessing.
- Never invent facts, sources, numbers, or events. If information may be outdated or you are unsure, say so explicitly.
- Only answer based on what is actually known or given in the conversation. Do not go off-topic or answer a question that was not asked.
- If a question is ambiguous, ask a clarifying question instead of assuming.
- Keep answers concise and directly relevant to the user's question.
- You may be given a "크리스피드(CRESPEED) 공식 웹사이트 정보", "실시간 웹 검색 결과", "실시간 날씨 정보", or "실시간 주식 시세" context block. Use it when it helps answer the question, and mention the source briefly. Ignore it if it isn't relevant."""

NEW_CHAT_TITLE = "새 대화"
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_RETRIES = 5
NAVER_SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# 검색엔진(네이버/DDG 모두)은 "~에 대해 설명해줘" 같은 자연어 지시문이 섞이면
# 엉뚱한 결과를 준다. 흔한 지시어 꼬리를 잘라내고 핵심 키워드만 남긴다.
QUERY_CLEANUP_PATTERNS = [
    r"(이|가|은|는)?\s*(뭐야|뭔가요|무엇인가요|뭐임|뭐지)\s*[?？!！.]*$",
    r"(에\s*대해서?|에\s*관해서?)?\s*(간단히|자세히|좀)?\s*"
    r"(설명해\s*[줘죠]?|설명해주세요|설명해줄래|알려\s*[줘죠]?|알려주세요|알려줄래|말해\s*[줘죠]?|말해주세요).*$",
    r"(좀|한번)?\s*(찾아|검색해)\s*(줘|주세요|줄래).*$",
    r"[?？!！.]+$",
]
DEFAULT_WEATHER_LOCATION = "Seoul"
NOMINATIM_USER_AGENT = "my-first-chat-bot/1.0 (personal streamlit weather feature)"
WEATHER_FORECAST_DAYS = 14  # Open-Meteo 무료 플랜은 최대 16일까지 지원

# 크리스피드(CRESPEED) 공식 홈페이지 RAG. 이 회사에 대한 질문에는 일반 웹
# 검색보다 회사 공식 사이트를 직접 크롤링한 내용을 우선 사용한다.
CRESPEED_BASE_URL = "http://www.crespeed.com/2017/html/main.html"
CRESPEED_DOMAIN = "www.crespeed.com"
CRESPEED_MAX_PAGES = 45  # fnGnbLink() 메뉴까지 포함하면 실제 페이지가 40개 안팎이라 여유있게 잡음
CRESPEED_KEYWORDS = ["크리스피드", "크레스피드", "crespeed"]
CRESPEED_CACHE_PATH = Path(__file__).parent / ".crespeed_cache.json"
CRESPEED_CACHE_MAX_AGE_DAYS = 7
CRESPEED_CACHE_VERSION = 2  # 크롤러가 fnGnbLink() 메뉴를 놓치던 버그를 고치면서 올림. 옛 캐시 자동 무효화용
CRESPEED_TOP_K = 6

WEATHER_KEYWORDS = ["날씨", "기온", "체감온도", "강수", "비 와", "비와", "눈 와", "눈와", "우산", "습도", "풍속"]

# 위치 후보를 고를 때 제외할, 지역명이 아닌 흔한 단어들.
WEATHER_SKIP_TOKENS = {
    "오늘", "내일", "모레", "지금", "현재", "이번주", "이번", "주말", "여기", "이곳", "저기", "거기", "요즘",
    "알려줘", "알려주세요", "알려줄래", "어때", "어떄", "어떠니", "궁금해", "궁금합니다",
    "좀", "정도", "관련", "정보", "얼마나", "말해줘", "말해주세요", "확인해줘", "확인해주세요",
    "그럼", "그러면", "그런데", "그냥", "아니", "저기요", "혹시", "음", "네", "예", "아",
}

# 지역명 뒤에 흔히 붙는 조사. 긴 것부터 검사해야 짧은 조사가 먼저 잘못 걸리지 않는다.
KOREAN_PARTICLE_SUFFIXES = ("에서의", "에서", "에게", "으로", "부터", "까지", "의", "은", "는", "이", "가", "을", "를", "도", "에")

# 한국 행정구역 접미사. 이걸로 끝나는 단어는 "그럼", "정도" 같은 일반 단어보다
# 지역명일 가능성이 훨씬 높으므로 위치 후보를 고를 때 우선한다.
PLACE_SUFFIXES = (
    "특별자치시", "특별자치도", "광역시", "특별시", "자치구",
    "시", "도", "구", "군", "읍", "면", "동", "리",
)

STOCK_KEYWORDS = ["주가", "주식", "시세", "종목", "증시", "상한가", "하한가", "매수", "매도", "코스피", "코스닥", "나스닥"]

STOCK_SKIP_TOKENS = {
    "오늘", "내일", "모레", "지금", "현재", "이번주", "이번", "요즘",
    "알려줘", "알려주세요", "알려줄래", "어때", "어떄", "얼마", "얼마야", "얼마임", "궁금해", "궁금합니다",
    "좀", "정도", "관련", "정보", "말해줘", "말해주세요", "확인해줘", "확인해주세요",
    "그럼", "그러면", "그런데", "그냥", "아니", "혹시", "음", "네", "예", "아",
}

# 한국어 종목명/별칭 -> Yahoo Finance 티커. yfinance의 검색 기능이 한글 질의에는
# 거의 응답하지 않아서(예: "삼성전자" 검색 시 결과 0건), 흔히 찾는 종목은 직접
# 매핑해둔다. 매핑에 없는 영문 티커/회사명은 yfinance.Search로 대체 조회한다.
STOCK_ALIASES = {
    # KOSPI 대형주
    "삼성전자": "005930.KS", "삼성전자우": "005935.KS", "삼전": "005930.KS",
    "SK하이닉스": "000660.KS", "하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS", "엘지에너지솔루션": "373220.KS", "LG엔솔": "373220.KS",
    "삼성바이오로직스": "207940.KS", "현대차": "005380.KS", "현대자동차": "005380.KS",
    "기아": "000270.KS", "셀트리온": "068270.KS",
    "POSCO홀딩스": "005490.KS", "포스코홀딩스": "005490.KS", "포스코": "005490.KS",
    "네이버": "035420.KS", "NAVER": "035420.KS",
    "카카오": "035720.KS", "카카오뱅크": "323410.KS", "카카오페이": "377300.KS",
    "LG화학": "051910.KS", "삼성SDI": "006400.KS", "현대모비스": "012330.KS",
    "KB금융": "105560.KS", "신한지주": "055550.KS", "하나금융지주": "086790.KS", "우리금융지주": "316140.KS",
    "SK이노베이션": "096770.KS", "SK텔레콤": "017670.KS", "KT": "030200.KS", "KT&G": "033780.KS",
    "LG전자": "066570.KS", "한국전력": "015760.KS", "삼성물산": "028260.KS",
    "두산에너빌리티": "034020.KS", "한화에어로스페이스": "012450.KS", "HD현대중공업": "329180.KS",
    "크래프톤": "259960.KS", "엔씨소프트": "036570.KS", "넷마블": "251270.KS",
    "삼성생명": "032830.KS", "삼성화재": "000810.KS", "LG": "003550.KS", "SK": "034730.KS",
    # KOSDAQ
    "에코프로": "086520.KQ", "에코프로비엠": "247540.KQ", "알테오젠": "196170.KQ",
    "셀트리온헬스케어": "091990.KQ", "펄어비스": "263750.KQ",
    # 미국 대형주
    "애플": "AAPL", "테슬라": "TSLA", "마이크로소프트": "MSFT",
    "구글": "GOOGL", "알파벳": "GOOGL", "아마존": "AMZN", "엔비디아": "NVDA",
    "메타": "META", "페이스북": "META", "넷플릭스": "NFLX", "코카콜라": "KO",
    "스타벅스": "SBUX", "디즈니": "DIS", "나이키": "NKE", "인텔": "INTC", "퀄컴": "QCOM",
    "보잉": "BA", "월마트": "WMT", "맥도날드": "MCD", "버크셔해서웨이": "BRK-B", "알리바바": "BABA",
}

NAVER_EXCHANGE_TO_YAHOO_SUFFIX = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}

STOCK_TREND_KEYWORDS = ["추이", "추세", "차트", "그래프", "히스토리", "흐름", "변동"]

# (기간을 나타내는 정규식, yfinance period 문자열). 위에서부터 먼저 매치되는 걸 쓴다.
STOCK_PERIOD_PATTERNS = [
    (r"(5|다섯)\s*년", "5y"),
    (r"(3|세)\s*년", "3y"),
    (r"(1|일)?\s*년|작년|올해|연초", "1y"),
    (r"(6|여섯)\s*개월|반년", "6mo"),
    (r"(3|세)\s*개월|분기", "3mo"),
    (r"(1|한)?\s*개월|한\s*달|1\s*달|지난\s*달", "1mo"),
    (r"(1|한|일)\s*주일?|일주일|지난\s*주", "5d"),
    (r"며칠", "5d"),
]
STOCK_DEFAULT_TREND_PERIOD = "1mo"

# WMO Weather interpretation codes (open-meteo.com 기준)
WMO_WEATHER_DESCRIPTIONS = {
    0: "맑음", 1: "대체로 맑음", 2: "부분적으로 흐림", 3: "흐림",
    45: "안개", 48: "짙은 안개(서리)",
    51: "약한 이슬비", 53: "이슬비", 55: "강한 이슬비",
    56: "약한 어는 이슬비", 57: "강한 어는 이슬비",
    61: "약한 비", 63: "비", 65: "강한 비",
    66: "약한 어는 비", 67: "강한 어는 비",
    71: "약한 눈", 73: "눈", 75: "강한 눈", 77: "싸락눈",
    80: "약한 소나기", 81: "소나기", 82: "강한 소나기",
    85: "약한 소낙눈", 86: "강한 소낙눈",
    95: "뇌우", 96: "약한 우박을 동반한 뇌우", 99: "강한 우박을 동반한 뇌우",
}


def _partial_tag_len(text: str, tag: str) -> int:
    """text 끝부분이 tag의 앞부분과 겹치는 길이를 반환한다 (청크 경계에서 태그가 잘리는 경우 대비)."""
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


def _stream_visible_chunks(stream, min_interval: float = STREAM_UI_MIN_INTERVAL):
    """OpenAI 스트림에서 <think>...</think>를 걸러낸 화면 표시용 텍스트 조각을 순서대로 내보낸다.

    st.write_stream에 넘기기 위한 제너레이터다. 매 토큰마다 직접
    placeholder.markdown()을 호출하던 이전 방식은 브라우저 쪽 React DOM
    갱신이 너무 잦아져 "NotFoundError: removeChild" 오류를 유발했다.
    st.write_stream으로 바꿔도(내부적으로 결국 비슷하게 자주 갱신하므로)
    같은 오류가 재현될 수 있어서, 여기서 직접 최소 시간 간격(min_interval)
    이상 모아뒀다가 한 번에 내보내 실제 DOM 갱신 횟수 자체를 줄인다.
    """
    buffer = ""
    in_think = False
    visible_response = ""
    yielded_len = 0
    pending = ""
    last_yield = time.monotonic()
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # 일부 서버는 추론 내용을 별도의 reasoning_content 필드로 보내므로 무시한다.
        piece = getattr(delta, "content", None)
        if not piece:
            continue
        buffer += piece

        # <think>...</think> 블록은 스트리밍 중에도 화면에 노출되지 않도록 걸러낸다.
        while True:
            tag = "</think>" if in_think else "<think>"
            idx = buffer.find(tag)
            if idx != -1:
                if not in_think:
                    visible_response += buffer[:idx]
                buffer = buffer[idx + len(tag):]
                in_think = not in_think
                continue

            # 태그가 청크 경계에서 잘렸을 수 있으니 안전한 부분까지만 흘려보낸다.
            partial = _partial_tag_len(buffer, tag)
            if not in_think:
                visible_response += buffer[: len(buffer) - partial]
            buffer = buffer[len(buffer) - partial:]
            break

        if len(visible_response) > yielded_len:
            pending += visible_response[yielded_len:]
            yielded_len = len(visible_response)
            now = time.monotonic()
            if now - last_yield >= min_interval:
                yield pending
                pending = ""
                last_yield = now

    if pending:
        yield pending


def _new_conversation():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"title": NEW_CHAT_TITLE, "messages": []}
    st.session_state.conversation_order.append(conv_id)
    st.session_state.current_id = conv_id


def _json_str_unescape(raw: str) -> str:
    """정규식으로 뽑아낸 JSON 문자열 리터럴 내용(\\n, \\", \\uXXXX 등)을 해제한다."""
    try:
        return json.loads(f'"{raw}"')
    except Exception:
        return raw


def _naver_web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS):
    """네이버 통합검색 결과 페이지에서 정보를 추출한다 (비공식, API 키 불필요).

    ddgs(DuckDuckGo)가 몇 초~몇십 초씩 완전히 응답하지 않는 경우를 직접
    확인했고, 한국 기업/기관에 대해서는 네이버 쪽 결과가 훨씬 정확하고
    풍부하다 (예: "크리스피드" 검색 시 네이버는 업종·사원수·대표자명까지
    포함한 AI 요약을 페이지에 내장하지만, ddgs 뉴스 검색은 무관한 결과만
    주는 경우가 있었다). 검색결과 페이지 HTML에 내장된 JSON 데이터에서
    AI 요약/기업정보/스니펫 필드를 정규식으로 직접 뽑아낸다.

    비공식 스크레이핑이라 네이버가 페이지 구조를 바꾸면 깨질 수 있다.
    실패하거나 아무것도 못 찾으면 조용히 빈 리스트를 반환하고, 호출부에서
    ddgs로 대체한다.
    """
    search_url = f"https://search.naver.com/search.naver?query={requests.utils.quote(query)}"
    try:
        resp = requests.get(
            "https://search.naver.com/search.naver",
            params={"query": query},
            headers={"User-Agent": NAVER_SEARCH_USER_AGENT},
            timeout=8,
        )
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return []

    results = []
    seen = set()

    def add(title: str, body: str):
        body = re.sub(r"</?mark>", "", body).strip()
        if body and len(body) > 5 and body not in seen:
            seen.add(body)
            results.append({"title": title, "body": body, "href": search_url})

    # 네이버 AI 요약 (신뢰도가 가장 높음)
    for m in re.finditer(r'"aiSourceInfoText":"((?:[^"\\]|\\.)*)"', html):
        add(f"{query} - 네이버 AI 요약", _json_str_unescape(m.group(1)))

    # 업종/사원수/대표자명 같은 구조화된 기업정보
    facts = re.findall(r'"key":"([^"]+)","valueData":\{"text":"((?:[^"\\]|\\.)*)"', html)
    if facts:
        fact_str = ", ".join(f"{k}: {_json_str_unescape(v)}" for k, v in facts[:6])
        add(f"{query} - 네이버 기업정보", fact_str)

    # 일반 검색결과 스니펫
    for m in re.finditer(r'"bodyText":"((?:[^"\\]|\\.)*)"', html):
        if len(results) >= max_results:
            break
        add(f"{query} - 네이버 검색", _json_str_unescape(m.group(1)))

    return results[:max_results]


def _ddgs_web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS, retries: int = WEB_SEARCH_RETRIES):
    """DuckDuckGo(ddgs)에서 실시간 정보를 검색한다. 네이버 검색이 실패했을 때의 대체 수단.

    뉴스 검색(news)은 발행일(date)이 함께 오기 때문에 최신성 판단에 유리해
    우선하지만, "이 회사가 뭐 하는 곳이냐" 같은 일반 질문에는 기업정보
    페이지 같은 일반 텍스트 검색(text) 결과가 훨씬 유용할 때가 많다. news 결과가
    "있기는 하지만" 부실한 경우를 놓치지 않도록 둘 다 가져와 합친다.

    ddgs 라이브러리는 여러 백엔드를 돌아가며 쓰는데 특정 백엔드가 타임아웃되거나
    "No results found"를 반환하는 등 꽤 불안정하다(같은 검색어로 3번 연속 시도했을 때
    정상/타임아웃/결과없음이 각각 나온 걸 직접 확인함). 그래서 실패하면 잠깐 쉬었다가
    몇 번 재시도한다. 그래도 실패하면 빈 리스트를 반환한다.
    """
    news_results, text_results = [], []
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                news_results = list(ddgs.news(query, max_results=max_results))
                text_results = list(ddgs.text(query, max_results=max_results))
            if news_results or text_results:
                break
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))  # 1.5s, 3s, 4.5s, ... 백엔드가 잠깐 죽어있는 경우를 더 버틴다

    combined = news_results + text_results
    seen = set()
    deduped = []
    for r in combined:
        key = r.get("url") or r.get("href") or r.get("link") or r.get("title")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
        if len(deduped) >= max_results:
            break
    return deduped


def _clean_search_query(text: str) -> str:
    """검색어에서 "~에 대해 설명해줘" 같은 지시문 꼬리를 잘라 핵심 키워드만 남긴다.

    검증해보니 네이버/DDG 둘 다 "크리스피드에 대해 설명해줘"라는 문장 전체를
    검색하면 완전히 무관한 결과를 주지만, "크리스피드"만 검색하면 정확한
    기업정보를 찾는다. 사용자 메시지를 그대로 검색어로 쓰던 이전 방식의
    근본 문제였다. 다 잘라내서 빈 문자열이 되면 원문을 그대로 쓴다.
    """
    cleaned = text.strip()
    for pattern in QUERY_CLEANUP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned).strip()
    return cleaned or text.strip()


def _web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS):
    """네이버 검색을 우선 시도하고, 결과가 없으면 DuckDuckGo로 대체한다."""
    query = _clean_search_query(query)
    naver_results = _naver_web_search(query, max_results=max_results)
    if naver_results:
        return naver_results
    return _ddgs_web_search(query, max_results=max_results)


def _looks_like_crespeed_query(text: str) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in CRESPEED_KEYWORDS)


def _crespeed_crawl():
    """crespeed.com을 같은 도메인 내에서 얕게 크롤링한다.

    이 사이트는 옛날 방식의 frameset 구조라 실제 콘텐츠는 메인 프레임
    페이지(CRESPEED_BASE_URL)에 있고, 인코딩도 EUC-KR이다.

    상단 메뉴 중 "회사소개"(CEO인사말, 회사개요, 회사연혁 등) 하위 항목은
    <a href>가 아니라 <li onclick="javascript:fnGnbLink('URL')">로 구현돼
    있어서, <a href> 태그만 훑는 크롤러로는 이 섹션 전체(약 15페이지)를
    통째로 놓친다는 걸 직접 확인했다("CEO 인사말 알려줘" 질문에 RAG가 엉뚱한
    답을 한 원인). 그래서 fnGnbLink() 자바스크립트 호출도 정규식으로 함께
    찾아 링크 큐에 추가한다. 홈페이지 링크를 따라가며 최대 CRESPEED_MAX_PAGES
    페이지까지만 수집한다(무한 크롤링 방지).
    """
    headers = {"User-Agent": NAVER_SEARCH_USER_AGENT}
    visited = set()
    queue = [CRESPEED_BASE_URL]
    pages = []
    while queue and len(visited) < CRESPEED_MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            resp.encoding = "euc-kr"
            html = resp.text
        except Exception:
            continue

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            title = lines[0]
            pages.append({"url": url, "title": title, "lines": lines})

        candidate_hrefs = [a["href"].strip() for a in soup.find_all("a", href=True)]
        candidate_hrefs += re.findall(r"fnGnbLink\('([^']+)'\)", html)

        for href in candidate_hrefs:
            if not href or href.startswith(("javascript:", "#", "mailto:", "tel:")):
                continue
            full = urljoin(url, href).split("#")[0]
            if urlparse(full).netloc == CRESPEED_DOMAIN and full not in visited and full not in queue:
                queue.append(full)

    return pages


def _crespeed_build_chunks(pages: list):
    """페이지들에서 반복되는 내비게이션/푸터 텍스트(보일러플레이트)를 제거하고 청크로 묶는다.

    모든 페이지에 공통으로 나오는 메뉴/주소 같은 줄은 대부분 절반 이상의
    페이지에서 그대로 반복되므로, 등장 빈도로 감지해서 제거한다(사이트별로
    일일이 하드코딩하지 않아도 되게).
    """
    if not pages:
        return []

    line_counts = Counter()
    for p in pages:
        for line in set(p["lines"]):
            line_counts[line] += 1
    boilerplate_threshold = max(2, int(len(pages) * 0.5))
    boilerplate = {line for line, cnt in line_counts.items() if cnt >= boilerplate_threshold}

    chunks = []
    for p in pages:
        content_lines = [line for line in p["lines"] if line not in boilerplate]
        buf, buf_len = [], 0
        for line in content_lines:
            buf.append(line)
            buf_len += len(line)
            if buf_len >= 500:
                chunks.append({"url": p["url"], "title": buf[0], "text": "\n".join(buf)})
                buf, buf_len = [], 0
        if buf:
            chunks.append({"url": p["url"], "title": buf[0], "text": "\n".join(buf)})

    return [c for c in chunks if len(c["text"]) >= 20]


def _embed_texts(texts: list):
    """임베딩 서버(dragonkue/bge-m3-ko, OpenAI 호환)로 텍스트를 벡터로 변환한다.

    메인 채팅용 LLM과 별도 서버(EMBEDDING_BASE_URL)를 쓴다. 실패하면 None을
    반환한다.
    """
    if not texts:
        return []
    try:
        resp = embedding_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in resp.data]
    except Exception:
        return None


def _load_crespeed_cache():
    try:
        if CRESPEED_CACHE_PATH.exists():
            age_seconds = time.time() - CRESPEED_CACHE_PATH.stat().st_mtime
            if age_seconds < CRESPEED_CACHE_MAX_AGE_DAYS * 86400:
                with open(CRESPEED_CACHE_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("version") == CRESPEED_CACHE_VERSION:
                    return data.get("chunks"), data.get("embeddings")
    except Exception:
        pass
    return None, None


def _save_crespeed_cache(chunks: list, embeddings: list):
    try:
        with open(CRESPEED_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"version": CRESPEED_CACHE_VERSION, "chunks": chunks, "embeddings": embeddings},
                f,
                ensure_ascii=False,
            )
    except Exception:
        pass


@st.cache_resource(show_spinner=False)
def _load_crespeed_index(_cache_bust: float = 0.0):
    """크리스피드 사이트 청크 + 임베딩을 (재)빌드하거나 디스크 캐시에서 불러온다.

    st.cache_resource로 감싸서 같은 서버 프로세스에서는 매 상호작용마다
    다시 계산하지 않는다. _cache_bust 값을 바꾸면(예: 새로고침 버튼) 강제로
    다시 빌드된다.
    """
    chunks, embeddings = _load_crespeed_cache()
    if chunks and embeddings:
        return chunks, np.array(embeddings)

    pages = _crespeed_crawl()
    chunks = _crespeed_build_chunks(pages)
    if not chunks:
        return [], None

    embeddings = _embed_texts([c["text"] for c in chunks])
    if not embeddings:
        return [], None

    _save_crespeed_cache(chunks, embeddings)
    return chunks, np.array(embeddings)


def _crespeed_search(query: str, top_k: int = CRESPEED_TOP_K):
    chunks, embeddings = _load_crespeed_index()
    if not chunks or embeddings is None:
        return []

    query_vec = _embed_texts([query])
    if not query_vec:
        return []

    q = np.array(query_vec[0])
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    d_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = d_norm @ q_norm

    ranked_idx = np.argsort(-sims)[:top_k]
    results = []
    for i in ranked_idx:
        c = chunks[int(i)]
        results.append({
            "title": f"크리스피드 공식 사이트 - {c['title']}",
            "body": c["text"][:800],
            "href": c["url"],
        })
    return results


def _build_crespeed_context(results: list) -> str:
    lines = [
        "[크리스피드(CRESPEED) 공식 웹사이트 정보]",
        "아래는 크리스피드 공식 홈페이지(crespeed.com)를 직접 수집해 질문과 관련도가",
        "높은 순으로 뽑은 내용입니다. 일반 웹 검색 결과나 너의 사전 지식보다 이 내용을",
        "우선하라. 여기 없는 내용은 추측하지 말고 모른다고 답하라.",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['title']}]\n{r['body']}\n출처: {r['href']}\n")
    return "\n".join(lines)


def _looks_like_weather_query(text: str) -> bool:
    return any(keyword in text for keyword in WEATHER_KEYWORDS)


def _strip_korean_particle(token: str) -> str:
    """조사를 뗀다. "까지의"처럼 조사가 겹쳐 붙는 경우까지 대비해 최대 2번 반복한다."""
    for _ in range(2):
        for suffix in KOREAN_PARTICLE_SUFFIXES:
            if len(token) > len(suffix) + 1 and token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        else:
            break
    return token


def _looks_like_place(token: str) -> bool:
    return len(token) >= 2 and token.endswith(PLACE_SUFFIXES)


def _best_place_form(token: str) -> tuple:
    """조사를 뗄지 말지를 결정한다.

    "홍천군의"는 조사 '의'를 떼야 지역 접미사 '군'이 드러나지만("홍천군"),
    "강원도"는 원본 자체가 이미 지역 접미사 '도'로 끝나므로 조사 제거 로직이
    "~도(역시)" 조사로 오인해 "강원"으로 잘라버리면 안 된다. 원본이 이미
    지역명처럼 보이면 원본을 우선하고, 그렇지 않을 때만 조사를 뗀 형태를 쓴다.
    """
    if _looks_like_place(token):
        return token, True
    stripped = _strip_korean_particle(token)
    return stripped, _looks_like_place(stripped)


def _weather_location_candidates(text: str, default: str):
    """문장에서 지역명일 가능성이 있는 후보들을 우선순위대로 뽑아낸다.

    한국어는 지역명 뒤에 조사('의', '은' 등)가 자유롭게 붙고, "송파구의 오늘
    날씨"처럼 '날씨' 바로 앞이 아닌 곳에 지역명이 올 수도 있어서, 정규식으로
    위치를 한 번에 콕 집어내려던 이전 방식은 "오늘"처럼 엉뚱한 단어를 뽑아내는
    문제가 있었다. 그 다음 버전(조사만 떼고 전부 후보로 삼는 방식)도 "그럼"
    같은 감탄사가 Nominatim에서 엉뚱한 가게 이름과 우연히 매칭되는 문제가
    있었다.

    이제는: (1) 시/도/구/군/읍/면/동 같은 한국 행정구역 접미사로 끝나는 단어를
    지역명 후보로 최우선 취급하고, (2) 그런 단어가 문장에서 연달아 나오면
    ("강원도" 다음에 "홍천군") 합쳐서("강원도 홍천군") 더 정확한 후보로 만든다.
    이런 후보가 다 실패한 뒤에야 나머지 일반 단어를 시도한다. 최종 판단은
    지오코딩 API가 실제로 찾아지는지로 검증한다 (_fetch_weather_for_query 참고).
    """
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text)
    forms = []  # (form, is_place_like)
    for tok in tokens:
        form, place_like = _best_place_form(tok)
        if len(form) < 2:
            continue
        if form in WEATHER_SKIP_TOKENS or "날씨" in form or "기온" in form:
            continue
        forms.append((form, place_like))

    combos = []
    for (form_a, place_a), (form_b, place_b) in zip(forms, forms[1:]):
        if place_a and place_b:
            combos.append(f"{form_a} {form_b}")

    place_singles = [f for f, is_place in forms if is_place]
    other_singles = [f for f, is_place in forms if not is_place]

    candidates = []
    for c in combos + place_singles + other_singles:
        if c not in candidates:
            candidates.append(c)
    candidates.append(default)
    return candidates


def _geocode(location: str):
    """Nominatim(OpenStreetMap, 무료·API 키 불필요)으로 지역명을 좌표로 변환한다.

    Open-Meteo 자체 지오코딩은 "송파구" 같은 한국 구 단위 행정구역을 아예
    데이터베이스에 갖고 있지 않거나("송파구" 검색 시 결과 0건), 같은 이름의
    동네가 여러 나라에 있을 때 엉뚱한 나라를 1순위로 반환하는 문제(예: "문정동"
    -> 북한 황해북도)가 있어 Nominatim으로 교체했다. Nominatim은 실제 주소/행정
    경계 데이터라 "송파구", "문정동" 모두 정확히 서울로 찾는다.
    """
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": location, "format": "json", "limit": 5, "accept-language": "ko"},
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=6,
    )
    resp.raise_for_status()
    results = resp.json() or []
    if not results:
        return None
    # 식당/상점 같은 개별 시설(class="amenity"/"shop" 등)이 지역명과 우연히
    # 이름이 겹쳐 엉뚱하게 매칭되는 걸 막기 위해, 행정구역/지명(class가 boundary
    # 또는 place)을 우선한다. 그런 결과가 없으면 어쩔 수 없이 1순위를 쓴다.
    administrative = [r for r in results if r.get("class") in ("boundary", "place")]
    return (administrative or results)[0]


def _fetch_weather(location: str):
    """Nominatim으로 지오코딩 후 Open-Meteo로 실시간 날씨 + 시간대별 예보를 가져온다.

    실패하면(지오코딩 실패, API 오류 등) None을 반환한다.
    """
    try:
        place = _geocode(location)
        if not place:
            return None
        lat, lon = float(place["lat"]), float(place["lon"])

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "hourly": "temperature_2m,precipitation_probability,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
                "forecast_days": WEATHER_FORECAST_DAYS,
                "timezone": "auto",
            },
            timeout=6,
        )
        weather_resp.raise_for_status()
        payload = weather_resp.json()
        current = payload.get("current")
        if not current:
            return None

        hourly = payload.get("hourly") or {}
        hourly_forecast = []
        times = hourly.get("time") or []
        if times:
            now_str = current.get("time", "")
            start_idx = next((i for i, t in enumerate(times) if t >= now_str), 0)
            for i in range(start_idx, min(start_idx + 12, len(times))):
                hourly_forecast.append({
                    "time": times[i],
                    "temperature": hourly["temperature_2m"][i],
                    "precipitation_probability": hourly["precipitation_probability"][i],
                    "description": WMO_WEATHER_DESCRIPTIONS.get(hourly["weather_code"][i], "알 수 없음"),
                })

        daily = payload.get("daily") or {}
        daily_forecast = []
        for i, day in enumerate(daily.get("time") or []):
            daily_forecast.append({
                "date": day,
                "description": WMO_WEATHER_DESCRIPTIONS.get(daily["weather_code"][i], "알 수 없음"),
                "temp_max": daily["temperature_2m_max"][i],
                "temp_min": daily["temperature_2m_min"][i],
                "precipitation_probability": daily["precipitation_probability_max"][i],
                "precipitation_sum": daily["precipitation_sum"][i],
            })

        return {
            "location": place.get("display_name", location),
            "observed_at": current.get("time"),
            "description": WMO_WEATHER_DESCRIPTIONS.get(current.get("weather_code"), "알 수 없음"),
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
            "hourly": hourly_forecast,
            "daily": daily_forecast,
        }
    except Exception:
        return None


def _fetch_weather_for_query(user_text: str, default_location: str):
    """문장에서 뽑은 지역 후보를 앞에서부터 시도해 지오코딩이 실제로 성공하는 첫 결과를 채택한다."""
    for candidate in _weather_location_candidates(user_text, default_location)[:6]:
        result = _fetch_weather(candidate)
        if result:
            return result
    return None


def _build_weather_context(weather: dict) -> str:
    lines = [
        "[실시간 날씨 정보 (Nominatim 지오코딩 + Open-Meteo 예보 API)]",
        f"지역: {weather['location']}",
        f"관측 시각: {weather['observed_at']}",
        f"현재 날씨: {weather['description']}",
        f"기온: {weather['temperature']}°C (체감 {weather['feels_like']}°C)",
        f"습도: {weather['humidity']}%",
        f"강수량: {weather['precipitation']}mm",
        f"풍속: {weather['wind_speed']}km/h",
    ]
    if weather.get("hourly"):
        lines.append("")
        lines.append("시간대별 예보 (다음 12시간):")
        for h in weather["hourly"]:
            lines.append(
                f"- {h['time']}: {h['description']}, 기온 {h['temperature']}°C, 강수확률 {h['precipitation_probability']}%"
            )
    if weather.get("daily"):
        lines.append("")
        lines.append(f"날짜별 예보 (앞으로 {len(weather['daily'])}일):")
        for d in weather["daily"]:
            lines.append(
                f"- {d['date']}: {d['description']}, 최고 {d['temp_max']}°C / 최저 {d['temp_min']}°C, "
                f"강수확률 {d['precipitation_probability']}%, 강수량 {d['precipitation_sum']}mm"
            )
    lines.append("")
    lines.append(
        "위 실시간 날씨 데이터를 사실로 받아들여 답변하라. 데이터에 없는 미래 시점(예: 예보 기간보다 먼 미래)에 "
        "대해서는 모른다고 답하고 추측하지 마라."
    )
    return "\n".join(lines)


def _render_weather(weather: dict, key: str = "live"):
    with st.container(border=True, key=f"weather_{key}"):
        st.markdown(f"🌤️ **{weather['location']}** 실시간 날씨  ·  {weather['observed_at']}")
        cols = st.columns(4)
        cols[0].metric("날씨", weather["description"])
        cols[1].metric("기온", f"{weather['temperature']}°C")
        cols[2].metric("체감", f"{weather['feels_like']}°C")
        cols[3].metric("습도", f"{weather['humidity']}%")

        if weather.get("hourly"):
            with st.expander(f"⏱️ 시간대별 예보 ({len(weather['hourly'])}시간)", key=f"weather_hourly_{key}"):
                for h in weather["hourly"]:
                    time_label = h["time"][-5:]
                    st.markdown(
                        f"- **{time_label}**  {h['description']}, {h['temperature']}°C · 강수확률 {h['precipitation_probability']}%"
                    )

        if weather.get("daily"):
            with st.expander(f"📅 날짜별 예보 ({len(weather['daily'])}일)", key=f"weather_daily_{key}"):
                for d in weather["daily"]:
                    st.markdown(
                        f"- **{d['date']}**  {d['description']}, "
                        f"{d['temp_min']}°C ~ {d['temp_max']}°C · 강수확률 {d['precipitation_probability']}% "
                        f"({d['precipitation_sum']}mm)"
                    )


def _looks_like_stock_query(text: str) -> bool:
    return any(keyword in text for keyword in STOCK_KEYWORDS)


def _looks_like_stock_trend_query(text: str) -> bool:
    if any(keyword in text for keyword in STOCK_TREND_KEYWORDS):
        return True
    return any(re.search(pattern, text) for pattern, _ in STOCK_PERIOD_PATTERNS)


def _detect_stock_period(text: str) -> str:
    for pattern, period in STOCK_PERIOD_PATTERNS:
        if re.search(pattern, text):
            return period
    return STOCK_DEFAULT_TREND_PERIOD


def _stock_candidates(text: str):
    """문장에서 종목명/티커일 가능성이 있는 후보들을 뽑아낸다.

    회사명은 지역명과 달리 공통 접미사가 없어 "지역 접미사 우선" 같은 규칙이
    통하지 않는다. 문장의 모든 단어에서 조사를 뗀 형태를 후보로 남기되,
    조사를 뗀 깔끔한 형태("하이닉스")를 원본("하이닉스의")보다 먼저 시도해서
    매칭됐을 때 표시 이름도 깔끔하게 나오게 한다.
    """
    tokens = re.findall(r"[가-힣A-Za-z0-9&.\-]+", text)
    candidates = []
    for tok in tokens:
        for form in dict.fromkeys([_strip_korean_particle(tok), tok]):
            if not form:
                continue
            if form in STOCK_SKIP_TOKENS or any(kw in form for kw in STOCK_KEYWORDS):
                continue
            if form not in candidates:
                candidates.append(form)
    return candidates


def _alias_lookup(candidate: str):
    """STOCK_ALIASES에서 종목을 찾는다. 정확히 일치하지 않으면 부분일치도 시도한다.

    "SK하이닉스"는 사전에 있어도 사람들은 흔히 "하이닉스"라고만 말한다.
    이런 줄임말/별칭을 일일이 다 등록할 수 없으니, 후보가 사전의 어느
    종목명과 부분적으로 겹치면(예: "하이닉스" ⊂ "SK하이닉스") 그것도 채택한다.
    후보가 너무 짧으면(2자 이하) 엉뚱한 종목과 겹칠 위험이 있어 제외한다.
    """
    if candidate in STOCK_ALIASES:
        return STOCK_ALIASES[candidate]
    if len(candidate) >= 3:
        for name, ticker in STOCK_ALIASES.items():
            if candidate in name or name in candidate:
                return ticker
    return None


def _naver_stock_search(query: str, max_results: int = 3):
    """네이버 금융 자동완성 검색(무료·API 키 불필요)으로 실제 상장 종목을 찾는다.

    STOCK_ALIASES는 손으로 만든 목록이라 "삼성전기" 같은 종목이 빠지면 그냥
    못 찾는다. 이 검색은 KRX/해외 상장 종목 전체를 대상으로 하고, "그럼" 같은
    무의미한 단어는 빈 결과를 반환해서(Nominatim처럼 엉뚱한 걸 억지로
    매칭하지 않음) 안전하다. 실패하면 빈 리스트를 반환한다.
    """
    try:
        resp = requests.get(
            "https://ac.stock.naver.com/ac",
            params={"q": query, "target": "stock,index,marketindicator"},
            timeout=5,
        )
        resp.raise_for_status()
        return (resp.json().get("items") or [])[:max_results]
    except Exception:
        return []


def _naver_item_to_ticker(item: dict):
    code = item.get("code")
    if not code:
        return None
    if item.get("nationCode") == "KOR":
        suffix = NAVER_EXCHANGE_TO_YAHOO_SUFFIX.get(item.get("typeCode"))
        return f"{code}{suffix}" if suffix else None
    # 해외 종목은 code가 이미 야후 티커 형식(예: TSLA)이다.
    return code


def _ticker_guesses(candidate: str):
    """후보 단어 하나에서 시도해볼 만한 (티커, 표시용 이름) 조합을 생성한다."""
    alias_ticker = _alias_lookup(candidate)
    if alias_ticker:
        yield alias_ticker, candidate
        return

    if re.fullmatch(r"\d{6}", candidate):
        # 한국 종목코드. 코스피(.KS)를 먼저, 코스닥(.KQ)을 그다음으로 시도한다.
        yield f"{candidate}.KS", candidate
        yield f"{candidate}.KQ", candidate
        return

    if candidate.isascii() and re.fullmatch(r"[A-Za-z]{1,5}(-[A-Za-z])?(\.[A-Za-z]{1,3})?", candidate):
        yield candidate.upper(), candidate.upper()
        return

    if len(candidate) < 2:
        return

    # 사전에 없는 종목명(국내/해외 모두)은 네이버 검색으로 실제 상장 여부를 확인한다.
    for item in _naver_stock_search(candidate):
        ticker = _naver_item_to_ticker(item)
        if ticker:
            yield ticker, item.get("name") or candidate


def _fetch_stock(ticker: str, display_name: str):
    """yfinance(Yahoo Finance, 무료·API 키 불필요)로 실시간 시세를 조회한다.

    quote_type이 "EQUITY"가 아니면 버린다. 종목코드를 ".KQ"로 잘못 추측했을 때
    (예: "005930.KQ") 엉뚱한 뮤추얼펀드가 그럴듯한 가격과 함께 매칭되는 걸
    직접 확인해서 추가한 안전장치다. 실패하면 None을 반환한다.
    """
    try:
        info = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None)
        quote_type = getattr(info, "quote_type", None)
        if price is None or quote_type != "EQUITY":
            return None

        previous_close = getattr(info, "previous_close", None) or getattr(
            info, "regular_market_previous_close", None
        )
        change = (price - previous_close) if previous_close else None
        change_pct = (change / previous_close * 100) if change is not None and previous_close else None

        return {
            "ticker": ticker,
            "name": display_name or ticker,
            "currency": getattr(info, "currency", "") or "",
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_pct": change_pct,
            "day_high": getattr(info, "day_high", None),
            "day_low": getattr(info, "day_low", None),
            "volume": getattr(info, "last_volume", None),
            "market_cap": getattr(info, "market_cap", None),
        }
    except Exception:
        return None


def _fetch_stock_history(ticker: str, period: str):
    """yfinance로 기간별 시세 추이를 가져온다. 실패하거나 데이터가 없으면 None을 반환한다."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty:
            return None
        closes = hist["Close"]
        start_price = float(closes.iloc[0])
        end_price = float(closes.iloc[-1])
        return {
            "period": period,
            "start_date": closes.index[0].strftime("%Y-%m-%d"),
            "end_date": closes.index[-1].strftime("%Y-%m-%d"),
            "start_price": start_price,
            "end_price": end_price,
            "period_high": float(hist["High"].max()),
            "period_low": float(hist["Low"].min()),
            "change": end_price - start_price,
            "change_pct": ((end_price - start_price) / start_price * 100) if start_price else None,
            # 차트 렌더링용 일별 종가. 세션에 저장했다가 재구성해야 하므로 날짜 문자열로 직렬화한다.
            "series": {d.strftime("%Y-%m-%d"): float(v) for d, v in closes.items()},
        }
    except Exception:
        return None


def _fetch_stock_for_query(user_text: str):
    """문장에서 뽑은 종목 후보를 앞에서부터 시도해 실제로 조회되는 첫 결과를 채택한다.

    "추이/차트/1달간" 같은 기간 표현이 있으면 현재가에 더해 해당 기간의
    시세 추이(history)도 함께 가져온다.
    """
    for candidate in _stock_candidates(user_text)[:6]:
        for ticker, display_name in _ticker_guesses(candidate):
            result = _fetch_stock(ticker, display_name)
            if result:
                if _looks_like_stock_trend_query(user_text):
                    period = _detect_stock_period(user_text)
                    history = _fetch_stock_history(ticker, period)
                    if history:
                        result["history"] = history
                return result
    return None


def _build_stock_context(stock: dict) -> str:
    lines = [
        "[실시간 주식 시세 (Yahoo Finance / yfinance)]",
        f"종목: {stock['name']} ({stock['ticker']})",
        f"현재가: {stock['price']:,.2f} {stock['currency']}",
    ]
    if stock.get("previous_close") is not None:
        lines.append(f"전일종가: {stock['previous_close']:,.2f} {stock['currency']}")
    if stock.get("change") is not None:
        lines.append(f"등락: {stock['change']:+,.2f} ({stock['change_pct']:+.2f}%)")
    if stock.get("day_high") is not None:
        lines.append(f"당일 고가/저가: {stock['day_high']:,.2f} / {stock['day_low']:,.2f}")
    if stock.get("volume") is not None:
        lines.append(f"거래량: {stock['volume']:,}")
    if stock.get("market_cap") is not None:
        lines.append(f"시가총액: {stock['market_cap']:,}")

    history = stock.get("history")
    if history:
        lines.append("")
        lines.append(f"기간별 추이 ({history['start_date']} ~ {history['end_date']}, period={history['period']}):")
        lines.append(f"- 시작가: {history['start_price']:,.2f} / 종료가(최근): {history['end_price']:,.2f}")
        lines.append(f"- 기간 등락: {history['change']:+,.2f} ({history['change_pct']:+.2f}%)")
        lines.append(f"- 기간 고가/저가: {history['period_high']:,.2f} / {history['period_low']:,.2f}")

    lines.append("")
    lines.append(
        "위 실시간 시세 데이터를 사실로 받아들여 답변하라. 이 데이터는 실시간이 아니라 최대 15~20분 지연될 "
        "수 있음을 밝혀라. 목표주가나 미래 전망처럼 데이터에 없는 내용은 추측하지 말고 모른다고 답하라."
    )
    return "\n".join(lines)


def _render_stock(stock: dict, key: str = "live"):
    with st.container(border=True, key=f"stock_{key}"):
        st.markdown(f"📈 **{stock['name']}** ({stock['ticker']})")
        cols = st.columns(4)
        change_label = None
        if stock.get("change") is not None:
            change_label = f"{stock['change']:+,.2f} ({stock['change_pct']:+.2f}%)"
        cols[0].metric("현재가", f"{stock['price']:,.2f} {stock['currency']}", change_label)
        cols[1].metric(
            "전일종가",
            f"{stock['previous_close']:,.2f}" if stock.get("previous_close") is not None else "-",
        )
        cols[2].metric(
            "고가/저가",
            f"{stock['day_high']:,.2f} / {stock['day_low']:,.2f}" if stock.get("day_high") is not None else "-",
        )
        cols[3].metric("거래량", f"{stock['volume']:,}" if stock.get("volume") is not None else "-")
        st.caption("Yahoo Finance 기준, 실시간이 아니라 최대 15~20분 지연될 수 있습니다.")

        history = stock.get("history")
        if history:
            st.markdown(
                f"**{history['start_date']} ~ {history['end_date']} 추이**  "
                f"{history['start_price']:,.2f} → {history['end_price']:,.2f} "
                f"({history['change_pct']:+.2f}%) · 고가 {history['period_high']:,.2f} / 저가 {history['period_low']:,.2f}"
            )
            series = pd.Series(history["series"])
            series.index = pd.to_datetime(series.index)
            st.line_chart(series)


def _today_note() -> str:
    """오늘 날짜를 항상 시스템 프롬프트에 포함시킨다 (검색 결과 유무와 무관하게).

    검색이 실패하거나 관련 결과가 없을 때도 모델이 현재 날짜를 알아야
    '아직 일어나지 않았다' 같은 잘못된 추측을 피할 수 있다.
    """
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return (
        f"[오늘 날짜: {today}]\n"
        "이 날짜를 반드시 사실로 받아들이고, 시점 관련 판단(이미 일어난 일인지 등)의 기준으로 삼아라. "
        "너의 사전 학습 지식에 있는 날짜 감각이나 추정과 다르더라도 이 날짜를 우선하라."
    )


def _build_search_context(results):
    lines = [
        "[실시간 웹 검색 결과]",
        "아래는 사용자의 질문과 관련해 방금 검색한 실제 웹 검색 결과입니다.",
        "이 정보는 너의 사전 학습 지식보다 최신이다. 학습 지식과 내용이 다르거나",
        "이 사건 자체를 학습하지 못했더라도, 아래 결과와 위에서 알려준 오늘 날짜를 사실로 받아들여라.",
        "각 결과에 발행일이 있으면 오늘 날짜와 비교해 사건이 이미 일어났는지 판단에 활용하고,",
        "너의 판단만으로 '아직 일어나지 않았다'고 추측하지 마라.",
        "결과들의 시점이 서로 다르면 발행일이 더 최근인 결과를 우선하라.",
        "관련이 있으면 이 정보를 바탕으로 답변하고, 관련이 없으면 무시하라.",
        "답변에 사용한 내용이 있다면 마지막에 출처 URL을 간단히 남겨라.",
        "이 결과는 검색 스니펫일 뿐, 실시간 시세/날씨 API처럼 구조화된 정확한 수치 데이터가 아니다.",
        "스니펫에 정확히 적혀 있지 않은 구체적인 숫자(예: 정확한 주가, 등락률, 기온)를 마치 실시간",
        "데이터를 조회한 것처럼 지어내지 마라. 정확한 수치가 없으면 대략적으로만 말하거나 모른다고 하라.",
        "",
    ]
    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        body = r.get("body") or ""
        href = r.get("href") or r.get("link") or r.get("url") or ""
        date = r.get("date")
        date_note = f" (발행일: {date})" if date else ""
        lines.append(f"{i}. {title}{date_note}\n{body}\n출처: {href}\n")
    return "\n".join(lines)


def _render_sources(sources, key: str = "live", label: str = "🔎 참고한 웹 검색 결과"):
    with st.expander(f"{label} {len(sources)}건", key=f"sources_{key}"):
        for r in sources:
            title = r.get("title") or "(제목 없음)"
            href = r.get("href") or r.get("link") or r.get("url") or ""
            body = r.get("body") or ""
            date = r.get("date")
            date_note = f" · {date[:10]}" if date else ""
            st.markdown(f"**[{title}]({href})**{date_note}  \n{body}")


st.set_page_config(page_title="My First Chat Bot", page_icon="🤖", layout="wide")

st.title("My First Chat Bot")

if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    st.session_state.conversation_order = []
    _new_conversation()

with st.sidebar:
    st.header("LLM 옵션")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.05)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    max_tokens = st.slider("Max output tokens", min_value=64, max_value=8192, value=1024, step=64)
    system_prompt = st.text_area(
        "System prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=200,
    )

    st.divider()
    st.subheader("🔎 실시간 정보")
    use_web_search = st.checkbox(
        "웹 검색으로 답변 보강",
        value=True,
        help="질문 내용으로 웹을 검색해 최신 정보를 답변에 반영합니다. LLM 자체는 실시간 정보를 알 수 없기 때문에 필요합니다.",
    )
    use_weather_api = st.checkbox(
        "🌤️ 날씨 질문에 실시간 날씨 API 사용",
        value=True,
        help="'~날씨'가 포함된 질문에는 검색 대신 실시간 날씨 API로 정확한 값을 가져옵니다. 현재 날씨/12시간 시간대별/앞으로 14일 예보까지 포함됩니다.",
    )
    default_weather_location = st.text_input(
        "기본 날씨 지역",
        value=DEFAULT_WEATHER_LOCATION,
        help="질문에 지역명이 없을 때(예: '오늘 날씨 어때?') 사용할 기본 지역입니다.",
    )
    use_stock_api = st.checkbox(
        "📈 주식 질문에 실시간 시세 API 사용",
        value=True,
        help="'~주가/시세' 등이 포함된 질문에는 검색 대신 Yahoo Finance 실시간 시세를 가져옵니다. 국내(삼성전자 등)/해외(AAPL 등) 종목을 모두 지원하며, '추이/차트/1달간' 같은 표현이 있으면 기간별 추이 차트도 함께 보여줍니다.",
    )
    use_crespeed_rag = st.checkbox(
        "🏢 크리스피드 질문에 공식 사이트 RAG 사용",
        value=True,
        help="'크리스피드'가 포함된 질문에는 crespeed.com을 직접 크롤링해 임베딩 검색(dragonkue/bge-m3-ko)한 내용을 최우선으로 사용합니다.",
    )
    if st.button("🔄 크리스피드 사이트 다시 수집", use_container_width=True):
        _load_crespeed_index.clear()
        try:
            CRESPEED_CACHE_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        st.toast("크리스피드 사이트 캐시를 지웠습니다. 다음 질문부터 새로 수집합니다.")

    if st.button("🆕 새 대화", use_container_width=True):
        _new_conversation()
        st.rerun()

    st.caption(f"Model: {LLM_MODEL}")
    st.caption(f"Endpoint: {LLM_BASE_URL}")

    st.divider()
    st.subheader("대화 목록")
    for conv_id in reversed(st.session_state.conversation_order):
        conv = st.session_state.conversations[conv_id]
        is_current = conv_id == st.session_state.current_id
        icon = "💬" if is_current else "🗨️"
        if st.button(
            f"{icon} {conv['title']}",
            key=f"conv_btn_{conv_id}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            st.session_state.current_id = conv_id
            st.rerun()

current_conv = st.session_state.conversations[st.session_state.current_id]
messages = current_conv["messages"]

client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
embedding_client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key="not-needed")

for idx, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message.get("crespeed_sources"):
            _render_sources(message["crespeed_sources"], key=f"hist_crespeed_{idx}", label="🏢 크리스피드 공식 사이트 참고")
        if message.get("weather"):
            _render_weather(message["weather"], key=f"hist_{idx}")
        if message.get("stock"):
            _render_stock(message["stock"], key=f"hist_{idx}")
        st.markdown(message["content"])
        if message.get("sources"):
            _render_sources(message["sources"], key=f"hist_{idx}")
        elif message.get("search_failed"):
            st.warning("⚠️ 실시간 웹 검색에 실패했습니다 (검색 엔진 응답 없음). 이 답변은 실시간 정보 없이 생성된 것이니 사실 확인이 필요합니다.")

user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    if current_conv["title"] == NEW_CHAT_TITLE:
        current_conv["title"] = user_input[:30] + ("…" if len(user_input) > 30 else "")

    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    weather_info = None
    stock_info = None
    crespeed_results = []
    search_results = []
    search_attempted = False
    if use_crespeed_rag and _looks_like_crespeed_query(user_input):
        with st.spinner("🏢 크리스피드 공식 사이트 조회 중..."):
            crespeed_results = _crespeed_search(user_input)
        if not crespeed_results and use_web_search:
            with st.spinner("🔎 웹에서 최신 정보를 검색하는 중..."):
                search_attempted = True
                search_results = _web_search(user_input)
    elif use_weather_api and _looks_like_weather_query(user_input):
        with st.spinner("🌤️ 날씨 조회 중..."):
            weather_info = _fetch_weather_for_query(user_input, default_weather_location)
        if not weather_info and use_web_search:
            with st.spinner("🔎 웹에서 최신 정보를 검색하는 중..."):
                search_attempted = True
                search_results = _web_search(user_input)
    elif use_stock_api and _looks_like_stock_query(user_input):
        with st.spinner("📈 시세 조회 중..."):
            stock_info = _fetch_stock_for_query(user_input)
        if not stock_info and use_web_search:
            with st.spinner("🔎 웹에서 최신 정보를 검색하는 중..."):
                search_attempted = True
                search_results = _web_search(user_input)
    elif use_web_search:
        with st.spinner("🔎 웹에서 최신 정보를 검색하는 중..."):
            search_attempted = True
            search_results = _web_search(user_input)

    history_for_request = [{"role": m["role"], "content": m["content"]} for m in messages]
    combined_system_prompt = system_prompt + "\n\n" + _today_note()
    if crespeed_results:
        combined_system_prompt += "\n\n" + _build_crespeed_context(crespeed_results)
    if weather_info:
        combined_system_prompt += "\n\n" + _build_weather_context(weather_info)
    if stock_info:
        combined_system_prompt += "\n\n" + _build_stock_context(stock_info)
    if search_results:
        combined_system_prompt += "\n\n" + _build_search_context(search_results)
    elif search_attempted:
        # 검색을 시도했지만 결과가 없거나 실패한 경우, 이 사실을 명시적으로 알려서
        # 모델이 "검색 결과를 바탕으로"라며 없는 정보를 지어내지 못하게 한다.
        combined_system_prompt += (
            "\n\n[실시간 웹 검색 결과]\n"
            "이 질문에 대해 웹 검색을 시도했지만 결과를 찾지 못했다(검색 실패 또는 결과 없음).\n"
            "검색 결과를 받은 것처럼 말하지 마라. 이 주제에 대해 확실히 아는 것이 없다면 "
            "모른다고 솔직히 답하고, 절대로 사실이나 세부 정보를 지어내지 마라."
        )
    request_messages = [{"role": "system", "content": combined_system_prompt}] + history_for_request

    with st.chat_message("assistant"):
        if crespeed_results:
            _render_sources(crespeed_results, label="🏢 크리스피드 공식 사이트 참고")
        if weather_info:
            _render_weather(weather_info)
        if stock_info:
            _render_stock(stock_info)
        if search_results:
            _render_sources(search_results)
        elif search_attempted:
            # 검색을 시도했지만 아무 결과도 못 얻은 경우. 모델이 "검색 결과를
            # 바탕으로"라며 지어낼 수 있으니, 실패 사실 자체를 화면에도 명확히 남긴다
            # (시스템 프롬프트 지시만으로는 이 로컬 모델이 종종 무시했다).
            st.warning("⚠️ 실시간 웹 검색에 실패했습니다 (검색 엔진 응답 없음). 아래 답변은 실시간 정보 없이 생성된 것이니 사실 확인이 필요합니다.")

        try:
            stream = client.chat.completions.create(
                model=LLM_MODEL,
                messages=request_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            full_response = st.write_stream(_stream_visible_chunks(stream)).strip()
        except Exception as e:
            full_response = f"오류가 발생했습니다: {e}"
            st.error(full_response)

    assistant_message = {"role": "assistant", "content": full_response}
    if crespeed_results:
        assistant_message["crespeed_sources"] = crespeed_results
    if weather_info:
        assistant_message["weather"] = weather_info
    if stock_info:
        assistant_message["stock"] = stock_info
    if search_results:
        assistant_message["sources"] = search_results
    elif search_attempted:
        assistant_message["search_failed"] = True
    messages.append(assistant_message)
