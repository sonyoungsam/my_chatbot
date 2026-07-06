# 🤖 My First Chat Bot

로컬 네트워크에 구축된 **자체 호스팅 LLM**(Qwen3)과 대화할 수 있는 [Streamlit](https://streamlit.io/) 기반 챗봇입니다.
OpenAI 호환 API를 사용하므로, 엔드포인트만 바꾸면 다른 로컬/사설 LLM 서버에도 쉽게 연결할 수 있습니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|---|---|
| 💬 실시간 스트리밍 응답 | 토큰이 생성되는 즉시 화면에 순차 출력 |
| 🧠 Reasoning(생각 과정) 자동 숨김 | Qwen3의 `<think>...</think>` 블록을 요청 단계 + 스트리밍 필터 이중으로 차단, 최종 답변만 표시 |
| 🎛️ 사이드바 LLM 옵션 | Temperature, Top P, Max output tokens, System prompt를 UI에서 즉시 조절 |
| 🗂️ 대화 히스토리 유지 | 세션 동안 대화 맥락을 기억하며, 버튼 한 번으로 초기화 가능 |
| ⚠️ 오류 처리 | LLM 서버 연결 실패 등 예외 상황을 채팅창에 바로 표시 |

---

## 🏗️ 아키텍처

```
┌─────────────────┐        OpenAI-compatible API        ┌──────────────────────────┐
│   Streamlit UI    │  ──────────────────────────────▶  │   Local LLM Server        │
│   (app.py)         │  ◀──────────────────────────────  │   Qwen/Qwen3.6-35B-A3B-FP8│
│                     │      streaming chat completion    │   192.168.0.201:18000     │
└─────────────────┘                                      └──────────────────────────┘
```

- **Frontend / Orchestration**: Streamlit (`app.py`) — 채팅 UI, 세션 상태 관리, 스트리밍 렌더링
- **LLM Client**: `openai` 파이썬 SDK — `base_url`만 로컬 서버로 지정해 OpenAI SDK를 그대로 사용
- **Model**: `Qwen/Qwen3.6-35B-A3B-FP8` (사내망 vLLM/SGLang 등 OpenAI 호환 서버에서 서빙)

---

## 📁 프로젝트 구조

```
my_chatbot/
├── app.py              # 메인 Streamlit 앱
├── requirements.txt     # 파이썬 의존성
├── prompts/
│   └── plan.md          # 최초 기획 문서
├── .gitignore
└── README.md
```

---

## 🚀 시작하기

### 1. 사전 요구사항

- Python 3.10 이상
- 사설망에서 접근 가능한 OpenAI 호환 LLM 서버 (`http://192.168.0.201:18000/v1`)

### 2. 가상환경 생성 및 활성화

**Windows (PowerShell)**
```powershell
cd d:\projects\my_chatbot
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
cd my_chatbot
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 앱 실행

```bash
streamlit run app.py
```

실행 후 터미널에 표시되는 주소로 접속하면 됩니다.

```
Local URL: http://localhost:8501
```

브라우저가 자동으로 열리지 않으면 위 주소를 직접 입력해서 접속하세요.

### 5. (Windows) 아이콘 더블클릭으로 실행하기

매번 터미널을 열지 않아도 되도록 실행 파일과 바탕화면 아이콘을 만들어 두었습니다.

- [run_chatbot.bat](run_chatbot.bat) — 더블클릭하면 가상환경 활성화 후 `streamlit run app.py` 를 자동 실행합니다. (최초 1회는 `venv`가 준비되어 있어야 합니다.)
- 바탕화면의 **"My Chat Bot"** 바로가기 — [assets/icon.ico](assets/icon.ico) 아이콘을 사용하는 `run_chatbot.bat` 바로가기입니다.

다른 PC에 새로 바로가기를 만들고 싶다면 PowerShell에서:

```powershell
$WshShell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$Shortcut = $WshShell.CreateShortcut("$desktop\My Chat Bot.lnk")
$Shortcut.TargetPath = "<프로젝트 경로>\run_chatbot.bat"
$Shortcut.WorkingDirectory = "<프로젝트 경로>"
$Shortcut.IconLocation = "<프로젝트 경로>\assets\icon.ico"
$Shortcut.Save()
```

---

## ⚙️ 설정

### LLM 엔드포인트 / 모델 변경

[app.py](app.py) 상단의 상수를 수정하면 다른 서버·모델로 전환할 수 있습니다.

```python
LLM_BASE_URL = "http://192.168.0.201:18000/v1"
LLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
```

### 사이드바 옵션

| 옵션 | 범위 | 기본값 | 설명 |
|---|---|---|---|
| Temperature | 0.0 ~ 2.0 | 0.7 | 값이 높을수록 응답이 창의적/무작위적 |
| Top P | 0.0 ~ 1.0 | 1.0 | 누적 확률 기반 샘플링 범위 |
| Max output tokens | 64 ~ 8192 | 1024 | 응답 최대 길이 |
| System prompt | 자유 입력 | `You are a helpful assistant.` | 모델의 역할/성격 지정 |

---

## 🧠 Reasoning(생각 과정) 숨김 처리

Qwen3 계열 모델은 최종 답변 전에 `<think>...</think>` 형태로 내부 추론 과정을 함께 생성하는 경우가 있습니다.
이 앱은 두 단계로 이를 걸러내어 **최종 답변만** 보여줍니다.

1. **요청 시 비활성화**: API 호출에 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 를 포함해, 서버가 이를 지원하면 애초에 생각 과정을 생성하지 않도록 요청합니다.
2. **스트리밍 필터 (안전장치)**: 서버가 그래도 `<think>...</think>` 블록을 보낼 경우, 스트리밍 도중 실시간으로 감지해 화면에 노출되지 않도록 제거합니다. 태그가 여러 토큰(청크)에 걸쳐 잘려서 오는 경우까지 처리합니다.

대화 히스토리에도 최종 답변만 저장되므로, 다음 턴 요청 시 이전 생각 과정이 컨텍스트로 재전송되지 않습니다.

---

## 🩺 문제 해결 (Troubleshooting)

| 증상 | 원인 / 해결 |
|---|---|
| `오류가 발생했습니다: Connection error.` | LLM 서버(`192.168.0.201:18000`)가 꺼져 있거나, 같은 네트워크(사설망)에 있지 않은 경우입니다. 서버 상태와 IP/포트를 확인하세요. |
| 첫 실행 시 터미널이 이메일 입력에서 멈춤 | Streamlit 최초 실행 시 나오는 온보딩 프롬프트입니다. `streamlit run app.py --browser.gatherUsageStats false` 로 실행하거나 빈 값으로 Enter를 누르면 됩니다. |
| 생각 과정(`<think>`)이 그대로 보임 | 서버가 `enable_thinking` 옵션을 지원하지 않을 수 있습니다. 스트리밍 필터가 안전장치로 동작하지만, 서버가 다른 태그/필드를 쓰는 경우 [app.py](app.py)의 필터 로직 조정이 필요합니다. |
| 응답이 중간에 잘림 | 사이드바의 `Max output tokens` 값을 늘려보세요. |
| 포트 충돌 (`8501` 사용 중) | `streamlit run app.py --server.port 8502` 처럼 다른 포트를 지정하세요. |

---

## 🗺️ 향후 개선 아이디어

- [ ] 대화 내용 파일로 저장/불러오기
- [ ] 멀티 세션(여러 대화 탭) 지원
- [ ] 이미지/파일 첨부 (멀티모달)
- [ ] 인증(비밀번호) 추가
- [ ] Docker 컨테이너화

---

## 📄 라이선스

개인 학습용 프로젝트입니다.
