import uuid
from datetime import datetime

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
- You may be given a "실시간 웹 검색 결과" context block. Use it when it helps answer the question, and mention the source briefly. Ignore it if it isn't relevant."""

NEW_CHAT_TITLE = "새 대화"
WEB_SEARCH_MAX_RESULTS = 5


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
    """DuckDuckGo에서 실시간 웹 검색을 수행한다. 실패하면 빈 리스트를 반환한다."""
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []


def _build_search_context(results):
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"[실시간 웹 검색 결과 / 오늘 날짜: {today}]",
        "아래는 사용자의 질문과 관련해 방금 검색한 웹 검색 결과입니다.",
        "관련이 있으면 이 정보를 바탕으로 답변하고, 관련이 없으면 무시하세요.",
        "답변에 사용한 내용이 있다면 마지막에 출처 URL을 간단히 남기세요.",
        "",
    ]
    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        body = r.get("body") or ""
        href = r.get("href") or r.get("link") or ""
        lines.append(f"{i}. {title}\n{body}\n출처: {href}\n")
    return "\n".join(lines)


def _render_sources(sources):
    with st.expander(f"🔎 참고한 웹 검색 결과 {len(sources)}건"):
        for r in sources:
            title = r.get("title") or "(제목 없음)"
            href = r.get("href") or r.get("link") or ""
            body = r.get("body") or ""
            st.markdown(f"**[{title}]({href})**  \n{body}")


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

    search_results = []
    if use_web_search:
        with st.spinner("🔎 웹에서 최신 정보를 검색하는 중..."):
            search_results = _web_search(user_input)

    history_for_request = [{"role": m["role"], "content": m["content"]} for m in messages]
    combined_system_prompt = system_prompt
    if search_results:
        combined_system_prompt += "\n\n" + _build_search_context(search_results)
    request_messages = [{"role": "system", "content": combined_system_prompt}] + history_for_request

    with st.chat_message("assistant"):
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
    if search_results:
        assistant_message["sources"] = search_results
    messages.append(assistant_message)
