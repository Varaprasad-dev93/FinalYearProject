"""
translate.py — Voice-to-English translation helper
Uses st.audio_input (Streamlit >= 1.37) — browser-native mic,
works both locally and on deployed apps.

Usage in app.py:
    from translate import render_voice_input
    query = render_voice_input()      # returns English string or None
    if query:
        response = chain.invoke({"input": query}, ...)
"""

import os
import io
import base64

import requests
import streamlit as st


def _cfg():
    return {
        "user_id":      os.getenv("VITE_USER_ID", "e98a6b8b96ef4d88811ae4a1a238e983"),
        "ulca_api_key": os.getenv("VITE_ULCA_API_KEY", "58c1c8d5a0-88f9-4e72-89fc-fc3b0ce428e0"),
        "pipeline_id":  os.getenv("VITE_PIPELINE_ID", "64392f96daac500b55c543cd"),
    }

PIPELINE_URL  = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
INFERENCE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
FLAC_CONV_URL = "https://voice-translation-6a2f.onrender.com/api/convert"
TARGET_LANG   = "en"

SUPPORTED_LANGUAGES = {
    
    "Hindi":     "hi",
    "Telugu":    "te",
    "Tamil":     "ta",
    "Bengali":   "bn",
    "Kannada":   "kn",
    "Malayalam": "ml",
    "Marathi":   "mr",
    "Punjabi":   "pa",
    "Gujarati":  "gu",
    "Odia":      "or",
    "Assamese":  "as",
    "Urdu":      "ur",
    "Sanskrit":  "sa",
    "Nepali":    "ne",
}

# ── Session state (prefixed to avoid collision with app.py keys) ──────────────
_PFX = "vt_"

def _s(key):
    return _PFX + key

def _init_state():
    defaults = {
        "status":        "idle",   # idle | processing | done | error
        "error_msg":     None,
        "asr_text":      None,
        "english_text":  None,
        "last_audio_id": None,     # tracks which recording was last processed
    }
    for k, v in defaults.items():
        if _s(k) not in st.session_state:
            st.session_state[_s(k)] = v


# ── Bhashini pipeline ─────────────────────────────────────────────────────────
def _to_flac_b64(wav_bytes: bytes) -> str | None:
    try:
        wav_b64 = base64.b64encode(wav_bytes).decode()
        res = requests.post(FLAC_CONV_URL, json={"audio_base64": wav_b64}, timeout=30)

        print("STATUS:", res.status_code)
        print("HEADERS:", res.headers.get("content-type"))
        print("SIZE:", len(res.content))

        if "application/json" in res.headers.get("content-type", ""):
            print("ERROR RESPONSE:", res.text)
            raise Exception("Conversion API returned JSON instead of FLAC")

        return base64.b64encode(res.content).decode()

    except Exception as e:
        st.session_state[_s("error_msg")] = f"FLAC conversion failed: {e}"
        return None


def _get_pipeline(source_lang: str) -> dict | None:
    cfg = _cfg()
    try:
        res = requests.post(
            PIPELINE_URL,
            json={
                "pipelineTasks": [
                    {"taskType": "asr",
                     "config": {"language": {"sourceLanguage": source_lang}}},
                    {"taskType": "translation",
                     "config": {"language": {"sourceLanguage": source_lang,
                                             "targetLanguage": TARGET_LANG}}},
                ],
                "pipelineRequestConfig": {"pipelineId": cfg["pipeline_id"]},
            },
            headers={
                "userID":       cfg["user_id"],
                "ulcaApiKey":   cfg["ulca_api_key"],
                "Content-Type": "application/json",
            },
            timeout=20,
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        msg = ""
        try: msg = res.json().get("message", "")
        except Exception: pass
        st.session_state[_s("error_msg")] = f"Pipeline config failed: {msg or e}"
        return None


def _run_inference(source_lang: str, flac_b64: str, pipeline: dict) -> bool:
    try:
        cfg_resp = pipeline["pipelineResponseConfig"]
        auth_key = pipeline["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
        asr_svc  = cfg_resp[0]["config"][0]["serviceId"]
        nmt_svc  = cfg_resp[1]["config"][0]["serviceId"]

        res = requests.post(
            INFERENCE_URL,
            json={
                "pipelineTasks": [
                    {"taskType": "asr",
                     "config": {"language":     {"sourceLanguage": source_lang},
                                "serviceId":    asr_svc,
                                "audioFormat":  "flac",
                                "samplingRate": 16000}},
                    {"taskType": "translation",
                     "config": {"language": {"sourceLanguage": source_lang,
                                             "targetLanguage": TARGET_LANG},
                                "serviceId": nmt_svc}},
                ],
                "inputData": {"audio": [{"audioContent": flac_b64}]},
            },
            headers={"Authorization": auth_key, "Content-Type": "application/json"},
            timeout=60,
        )
        res.raise_for_status()
        pr = res.json().get("pipelineResponse", [])
        st.session_state[_s("asr_text")]     = pr[0]["output"][0]["source"] if pr else ""
        st.session_state[_s("english_text")] = pr[1]["output"][0]["target"] if len(pr) > 1 else ""
        return True
    except Exception as e:
        msg = ""
        try: msg = res.json().get("message", "")
        except Exception: pass
        st.session_state[_s("error_msg")] = f"Inference failed: {msg or e}"
        return False


def _translate(source_lang: str, audio_bytes: bytes):
    """WAV bytes -> FLAC -> ASR -> NMT -> stores results in session state."""
    st.session_state[_s("status")]       = "processing"
    st.session_state[_s("error_msg")]    = None
    st.session_state[_s("asr_text")]     = None
    st.session_state[_s("english_text")] = None

    flac_b64 = _to_flac_b64(audio_bytes)
    if not flac_b64:
        st.session_state[_s("status")] = "error"
        return

    pipeline = _get_pipeline(source_lang)
    if not pipeline:
        st.session_state[_s("status")] = "error"
        return

    ok = _run_inference(source_lang, flac_b64, pipeline)
    st.session_state[_s("status")] = "done" if ok else "error"


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
.vt-badge {
    display:inline-flex; align-items:center; gap:0.35rem;
    font-size:0.7rem; font-family:monospace; letter-spacing:.06em;
    padding:0.2rem 0.65rem; border-radius:999px; border:1px solid;
    white-space:nowrap;
}
.vt-idle    { border-color:#3a3f52; color:#6b7280; }
.vt-process { border-color:#4f9eff; color:#4f9eff;
              animation:vt-pulse 1.1s ease infinite; }
.vt-done    { border-color:#34d399; color:#34d399; }
.vt-error   { border-color:#f87171; color:#f87171; }
@keyframes vt-pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

.vt-result {
    background:#0f1117; border:1px solid #242836;
    border-left:3px solid #34d399;
    border-radius:8px; padding:0.6rem 0.9rem;
    font-size:0.95rem; color:#e8eaf0;
    margin-top:0.4rem; word-break:break-word;
}
.vt-asr {
    border-left-color:#a78bfa;
    font-size:0.8rem; color:#9ca3af; margin-top:0.3rem;
}
</style>
"""

def render_voice_input(container=None) -> str | None:
    _init_state()
    ctx = container or st

    ctx.markdown(_CSS, unsafe_allow_html=True)


    col_lang, col_status = ctx.columns([2, 1])

    with col_lang:
        lang_name = ctx.selectbox(
            "Speak in",
            list(SUPPORTED_LANGUAGES.keys()),
            key=_s("lang_select"),
            label_visibility="collapsed",
        )
    source_lang = SUPPORTED_LANGUAGES[lang_name]

    status = st.session_state[_s("status")]
    badge_map = {
        "idle":       ("vt-idle",    "○", "ready"),
        "processing": ("vt-process", "◌", "translating…"),
        "done":       ("vt-done",    "✓", "done"),
        "error":      ("vt-error",   "✕", "error"),
    }
    badge_cls, badge_icon, badge_label = badge_map.get(status, badge_map["idle"])

    with col_status:
        ctx.markdown(
            f'<div style="padding-top:6px">'
            f'<span class="vt-badge {badge_cls}">{badge_icon} {badge_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


    audio_file = ctx.audio_input(
        "Record your question",
        key=_s("audio_input"),
        label_visibility="visible",
    )

    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        audio_id = hash(audio_bytes)
        if audio_id != st.session_state[_s("last_audio_id")]:
            st.session_state[_s("last_audio_id")] = audio_id
            audio_bytes  = audio_file.getvalue()
            with ctx.status("Translating…", expanded=False):
                _translate(source_lang, audio_bytes)
            st.rerun()


    if st.session_state[_s("error_msg")]:
        ctx.error(st.session_state[_s("error_msg")])
        if ctx.button("↺ Clear & retry", key=_s("btn_clear_err")):
            st.session_state[_s("error_msg")] = None
            st.session_state[_s("status")]    = "idle"
            st.rerun()


    english = st.session_state[_s("english_text")]
    asr = st.session_state[_s("asr_text")]

    if english:
        ctx.markdown(
            f'<div class="vt-result">🌐 <strong>English:</strong> {english}</div>',
            unsafe_allow_html=True,
        )
    if asr:
        ctx.markdown(
            f'<div class="vt-result vt-asr">'
            f'🔤 Transcribed ({lang_name}): {asr}</div>',
            unsafe_allow_html=True,
        )


    if status == "done" and english:
        st.session_state[_s("status")] = "idle"
        return english.strip()

    return None