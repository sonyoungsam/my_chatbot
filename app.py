import re
import uuid
from datetime import datetime

import requests
import streamlit as st
from ddgs import DDGS
from openai import OpenAI

LLM_BASE_URL = "http://192.168.0.201:18000/v1"
LLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

DEFAULT_SYSTEM_PROMPT = """You are a helpful, honest assistant.

- If you do not know the answer, or are not confident, say so plainly instead of guessing.
- Never invent facts, sources, numbers, or events. If information may be outdated or you are unsure, say so explicitly.
- Only answer based on what is actually known or given in the conversation. Do not go off-topic or answer a question that was not asked.
- If a question is ambiguous, ask a clarifying question instead of assuming.
- Keep answers concise and directly relevant to the user's question.
- You may be given a "실시간 웹 검색 결과" or "실시간 날씨 정보" context block. Use it when it helps answer the question, and mention the source briefly. Ignore it if it isn't relevant."""

NEW_CHAT_TITLE = "새 대화"
WEB_SEARCH_MAX_RESULTS = 5
DEFAULT_WEATHER_LOCATION = "Seoul"
NOMINATIM_USER_AGENT = "my-first-chat-bot/1.0 (personal streamlit weather feature)"
WEATHER_FORECAST_DAYS = 14  # Open-Meteo 무료 플랜은 최대 16일까지 지원

WEATHER_KEYWORDS = ["날씨", "기온", "체감온도", "강수", "비 와", "비와", "눈 와", "눈와", "우산", "습도", "풍속"]

# 위치 후보를 고를 때 제외할, 지역명이 아닌 흔한 단어들.
WEATHER_SKIP_TOKENS = {
    "오늘", "내일", "모레", "지금", "현재", "이번주", "이번", "주말", "여기", "이곳", "저기", "거기", "요즘",
    "알려줘", "알려주세요", "알려줄래", "어때", "어떄", "어떠니", "궁금해", "궁금합니다",
    "좀", "정도", "관련", "정보", "얼마나", "말해줘", "말해주세요", "확인해줘", "확인해주세요",
}

# 지역명 뒤에 흔히 붙는 조사. 긴 것부터 검사해야 짧은 조사가 먼저 잘못 걸리지 않는다.
KOREAN_PARTICLE_SUFFIXES = ("에서의", "에서", "에게", "으로", "부터", "까지", "의", "은", "는", "이", "가", "을", "를", "도", "에")

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


def _new_conversation():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"title": NEW_CHAT_TITLE, "messages": []}
    st.session_state.conversation_order.append(conv_id)
    st.session_state.current_id = conv_id


def _web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS):
    """DuckDuckGo에서 실시간 정보를 검색한다.

    뉴스 검색(news)은 발행일(date)이 함께 오기 때문에, 날짜가 없어 모델이
    최신성을 판단할 근거가 없는 일반 텍스트 검색(text)보다 우선한다.
    뉴스 결과가 없을 때만 텍스트 검색으로 대체한다. 실패하면 빈 리스트를 반환한다.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
            if not results:
                results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception:
        return []


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


def _weather_location_candidates(text: str, default: str):
    """문장에서 지역명일 가능성이 있는 후보들을 순서대로 뽑아낸다.

    한국어는 지역명 뒤에 조사('의', '은' 등)가 자유롭게 붙고, "송파구의 오늘
    날씨"처럼 '날씨' 바로 앞이 아닌 곳에 지역명이 올 수도 있어서, 정규식으로
    위치를 한 번에 콕 집어내려던 이전 방식은 "오늘"처럼 엉뚱한 단어를
    뽑아내는 문제가 있었다. 대신 문장의 모든 단어에서 흔한 조사를 떼어낸 뒤,
    지역명이 아닐 게 뻔한 단어만 걸러내고 나머지는 전부 후보로 남긴다.
    실제로 어느 후보가 진짜 지역인지는 지오코딩 API가 검증하게 한다
    (_fetch_weather_for_query 참고).
    """
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text)
    candidates = []
    for tok in tokens:
        stripped = _strip_korean_particle(tok)
        if len(stripped) < 2:
            continue
        if stripped in WEATHER_SKIP_TOKENS or "날씨" in stripped or "기온" in stripped:
            continue
        if stripped not in candidates:
            candidates.append(stripped)
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
        params={"q": location, "format": "json", "limit": 1, "accept-language": "ko"},
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=6,
    )
    resp.raise_for_status()
    results = resp.json()
    return results[0] if results else None


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
    for candidate in _weather_location_candidates(user_text, default_location)[:5]:
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


def _render_weather(weather: dict):
    with st.container(border=True):
        st.markdown(f"🌤️ **{weather['location']}** 실시간 날씨  ·  {weather['observed_at']}")
        cols = st.columns(4)
        cols[0].metric("날씨", weather["description"])
        cols[1].metric("기온", f"{weather['temperature']}°C")
        cols[2].metric("체감", f"{weather['feels_like']}°C")
        cols[3].metric("습도", f"{weather['humidity']}%")

        if weather.get("hourly"):
            with st.expander(f"⏱️ 시간대별 예보 ({len(weather['hourly'])}시간)"):
                for h in weather["hourly"]:
                    time_label = h["time"][-5:]
                    st.markdown(
                        f"- **{time_label}**  {h['description']}, {h['temperature']}°C · 강수확률 {h['precipitation_probability']}%"
                    )

        if weather.get("daily"):
            with st.expander(f"📅 날짜별 예보 ({len(weather['daily'])}일)"):
                for d in weather["daily"]:
                    st.markdown(
                        f"- **{d['date']}**  {d['description']}, "
                        f"{d['temp_min']}°C ~ {d['temp_max']}°C · 강수확률 {d['precipitation_probability']}% "
                        f"({d['precipitation_sum']}mm)"
                    )


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


def _render_sources(sources):
    with st.expander(f"🔎 참고한 웹 검색 결과 {len(sources)}건"):
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

for message in messages:
    with st.chat_message(message["role"]):
        if message.get("weather"):
            _render_weather(message["weather"])
        st.markdown(message["content"])
        if message.get("sources"):
            _render_sources(message["sources"])

user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    if current_conv["title"] == NEW_CHAT_TITLE:
        current_conv["title"] = user_input[:30] + ("…" if len(user_input) > 30 else "")

    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    weather_info = None
    search_results = []
    if use_weather_api and _looks_like_weather_query(user_input):
        with st.spinner("🌤️ 날씨 조회 중..."):
            weather_info = _fetch_weather_for_query(user_input, default_weather_location)
        if not weather_info and use_web_search:
            with st.spinner("🔎 웹에서 최신 정보를 검색하는 중..."):
                search_results = _web_search(user_input)
    elif use_web_search:
        with st.spinner("🔎 웹에서 최신 정보를 검색하는 중..."):
            search_results = _web_search(user_input)

    history_for_request = [{"role": m["role"], "content": m["content"]} for m in messages]
    combined_system_prompt = system_prompt + "\n\n" + _today_note()
    if weather_info:
        combined_system_prompt += "\n\n" + _build_weather_context(weather_info)
    if search_results:
        combined_system_prompt += "\n\n" + _build_search_context(search_results)
    request_messages = [{"role": "system", "content": combined_system_prompt}] + history_for_request

    with st.chat_message("assistant"):
        if weather_info:
            _render_weather(weather_info)
        if search_results:
            _render_sources(search_results)

        placeholder = st.empty()
        buffer = ""
        in_think = False
        visible_response = ""
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

                placeholder.markdown(visible_response.strip() + "▌")
            full_response = visible_response.strip()
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"오류가 발생했습니다: {e}"
            placeholder.error(full_response)

    assistant_message = {"role": "assistant", "content": full_response}
    if weather_info:
        assistant_message["weather"] = weather_info
    if search_results:
        assistant_message["sources"] = search_results
    messages.append(assistant_message)
