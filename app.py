import streamlit as st
from openai import OpenAI

LLM_BASE_URL = "http://192.168.0.201:18000/v1"
LLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

DEFAULT_SYSTEM_PROMPT = """You are a helpful, honest assistant.

- If you do not know the answer, or are not confident, say so plainly instead of guessing.
- Never invent facts, sources, numbers, or events. If information may be outdated or you are unsure, say so explicitly.
- Only answer based on what is actually known or given in the conversation. Do not go off-topic or answer a question that was not asked.
- If a question is ambiguous, ask a clarifying question instead of assuming.
- Keep answers concise and directly relevant to the user's question."""


def _partial_tag_len(text: str, tag: str) -> int:
    """text 끝부분이 tag의 앞부분과 겹치는 길이를 반환한다 (청크 경계에서 태그가 잘리는 경우 대비)."""
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0

st.set_page_config(page_title="My First Chat Bot", page_icon="🤖", layout="wide")

st.title("My First Chat Bot")

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

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Model: {LLM_MODEL}")
    st.caption(f"Endpoint: {LLM_BASE_URL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    request_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    with st.chat_message("assistant"):
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

    st.session_state.messages.append({"role": "assistant", "content": full_response})
