import streamlit as st
import pandas as pd
import json
import requests
import os
from datetime import datetime
import time
import threading
import base64
from io import BytesIO
from gtts import gTTS
from dotenv import load_dotenv
import uuid
import tempfile
import shutil
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "messages": [],
    "session_id": f"session_{int(time.time())}",
    "response_cache": {},
    "feedback_ratings": {},
    "contact_form_submitted": False,
    "career_goals": [],
    "daily_tip_shown": False,
    "user_profile": {
        "name": "",
        "email": "",
        "phone": "",
        "experience": 0,
        "skills": [],
        "education": [],
        "work_history": [],
        "preferred_language": "en",
        "resume_data": None,
        "job_target": "",
        "linkedin": "",
        "github": "",
        "portfolio": "",
    },
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  NVIDIA API CLIENT
# ─────────────────────────────────────────────
class NvidiaAPIClient:
    """Wrapper for NVIDIA NIM API (OpenAI-compatible endpoint)."""

    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    MODEL    = "nvidia/llama-3.1-nemotron-70b-instruct"

    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY", "")
        self.available = bool(self.api_key)

    def chat(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
        if not self.available:
            return (
                "⚠️ NVIDIA API key not configured. "
                "Add `NVIDIA_API_KEY` to your `.env` file and restart."
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        try:
            resp = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            return f"❌ API error {resp.status_code}: {resp.text}"
        except Exception as e:
            return f"❌ Request failed: {str(e)}"


nvidia_client = NvidiaAPIClient()

# ─────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are CareerPilot, a sharp and empowering Personal Career Assistant AI.
Your mission: help users accelerate their careers through guidance on job searching, resume building, interview prep, skill development, networking, and personal branding.

Guidelines:
- Be concise, direct, and actionable. Avoid fluff.
- Celebrate wins and encourage boldly.
- Ask clarifying questions when intent is unclear.
- Provide structured advice (numbered steps, bullet points) when helpful.
- When discussing career paths, highlight both traditional and unconventional routes.
- Keep voice-friendly responses brief (under 120 words).
- You are NOT affiliated with any external job platform. You are the user's personal AI career coach.
- Your name is CareerPilot.
"""

# ─────────────────────────────────────────────
#  DAILY CAREER TIPS
# ─────────────────────────────────────────────
DAILY_TIPS = [
    "💡 Update your LinkedIn headline to reflect your *target* role, not your current one.",
    "💡 Send one cold outreach message today — even a short one builds momentum.",
    "💡 Quantify your impact: replace 'managed a team' with 'led 8 engineers to ship X 2 weeks early'.",
    "💡 Rejections are data points, not verdicts. Log them and look for patterns.",
    "💡 Your portfolio URL belongs in your email signature, resume, and LinkedIn bio.",
    "💡 Practice one interview answer aloud today using the STAR method.",
    "💡 Join one industry Slack or Discord this week. Communities open doors.",
    "💡 A two-page resume is fine — white space matters more than page count.",
    "💡 Research the interviewer on LinkedIn before every call.",
    "💡 Skill gaps aren't barriers — they're your learning roadmap.",
]

def get_daily_tip() -> str:
    day_index = datetime.now().timetuple().tm_yday % len(DAILY_TIPS)
    return DAILY_TIPS[day_index]

# ─────────────────────────────────────────────
#  CORE AI HELPERS
# ─────────────────────────────────────────────
def build_messages(user_query: str, extra_context: str = "") -> list[dict]:
    history = []
    for msg in st.session_state.messages[-6:]:
        history.append({"role": msg["role"], "content": msg["content"]})

    profile = st.session_state.user_profile
    profile_ctx = ""
    if profile["name"]:
        profile_ctx = f"""
User profile:
- Name: {profile['name']}
- Years of experience: {profile['experience']}
- Skills: {', '.join(profile['skills']) or 'not specified'}
- Target role: {profile.get('job_target', 'not specified')}
"""

    system = SYSTEM_PROMPT
    if profile_ctx:
        system += f"\n\n{profile_ctx}"
    if extra_context:
        system += f"\n\nContext:\n{extra_context}"

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_query})
    return messages


def careerpilot_chat(user_query: str, extra_context: str = "", max_tokens: int = 512) -> str:
    cache_key = f"{user_query}_{extra_context}"
    if st.session_state.get("enable_caching", True) and cache_key in st.session_state.response_cache:
        return st.session_state.response_cache[cache_key]
    msgs = build_messages(user_query, extra_context)
    result = nvidia_client.chat(msgs, max_tokens=max_tokens)
    st.session_state.response_cache[cache_key] = result
    return result


def generate_resume(profile: dict) -> str:
    prompt = f"""
Create a polished, ATS-friendly resume in Markdown for:
{json.dumps(profile, indent=2)}

Structure:
# [Full Name]
Contact | LinkedIn | GitHub | Portfolio

## Professional Summary (3 sentences, punchy)

## Core Skills (2-column bullet list)

## Work Experience (reverse chronological, bullet-point achievements, quantify where possible)

## Education

## Projects / Certifications (if applicable)

Use bold for company names and titles. Be concise and impactful.
"""
    msgs = [
        {"role": "system", "content": "You are an expert resume writer. Output clean Markdown only."},
        {"role": "user", "content": prompt},
    ]
    return nvidia_client.chat(msgs, max_tokens=1500, temperature=0.4)


def analyze_resume_feedback(parsed: dict) -> str:
    prompt = f"""
Review this resume data and give 5 concise, actionable improvement tips:
{json.dumps(parsed, indent=2)}

Format as numbered list. Be specific and critical.
"""
    msgs = [
        {"role": "system", "content": "You are a brutally honest but supportive resume reviewer."},
        {"role": "user", "content": prompt},
    ]
    return nvidia_client.chat(msgs, max_tokens=600, temperature=0.5)


def generate_cover_letter(profile: dict, job_desc: str) -> str:
    prompt = f"""
Write a compelling cover letter for this candidate applying to:
Job Description: {job_desc}

Candidate profile:
{json.dumps(profile, indent=2)}

Keep it under 300 words. Professional, confident, and specific. Use Markdown.
"""
    msgs = [
        {"role": "system", "content": "You are an expert career coach specialising in cover letters."},
        {"role": "user", "content": prompt},
    ]
    return nvidia_client.chat(msgs, max_tokens=800, temperature=0.6)


def generate_interview_questions(job_title: str, skills: list) -> str:
    prompt = f"""
Generate 8 targeted interview questions for a {job_title} role.
The candidate's key skills: {', '.join(skills) or 'general'}.

Mix: 3 behavioral (STAR), 3 technical, 2 situational.
After each question, add a one-line tip on how to answer it well.
"""
    msgs = [
        {"role": "system", "content": "You are a senior hiring manager and interview coach."},
        {"role": "user", "content": prompt},
    ]
    return nvidia_client.chat(msgs, max_tokens=900, temperature=0.5)

# ─────────────────────────────────────────────
#  VOICE HELPERS
# ─────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi",
}

def text_to_speech(text: str, lang: str = "en") -> BytesIO | None:
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

def autoplay_audio(audio_fp: BytesIO):
    b64 = base64.b64encode(audio_fp.getvalue()).decode()
    st.markdown(
        f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )

def speech_to_text(lang: str = "en") -> str | None:
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        r.energy_threshold = 3500
        r.dynamic_energy_threshold = True
        st.info(f"🎙️ Listening in {SUPPORTED_LANGUAGES.get(lang, 'English')}… speak now")
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = r.listen(source, timeout=8, phrase_time_limit=15)
                text = r.recognize_google(audio, language=lang)
                st.success(f"Recognised: {text}")
                return text
            except sr.WaitTimeoutError:
                st.warning("Didn't catch that — please try again.")
            except sr.UnknownValueError:
                st.warning("Couldn't understand the audio.")
            except sr.RequestError as e:
                st.error(f"Speech service error: {e}")
    except ImportError:
        st.error("Install `SpeechRecognition` and `pyaudio` for voice input.")
    return None

# ─────────────────────────────────────────────
#  FEEDBACK COMPONENT
# ─────────────────────────────────────────────
def display_feedback(message_id: str):
    existing = st.session_state.feedback_ratings.get(message_id)
    if existing == "up":
        st.caption("👍 Thanks for the feedback!")
    elif existing == "down":
        st.caption("👎 Noted — I'll do better!")
    else:
        c1, c2, _ = st.columns([1, 1, 8])
        with c1:
            if st.button("👍", key=f"up_{message_id}"):
                st.session_state.feedback_ratings[message_id] = "up"
                st.rerun()
        with c2:
            if st.button("👎", key=f"dn_{message_id}"):
                st.session_state.feedback_ratings[message_id] = "down"
                st.rerun()

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.02em;
    }

    .stApp {
        background: #0a0a0f;
        color: #e8e8f0;
    }

    /* Glowing accent line under main title */
    .nova-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #76e3ff 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }

    .nova-sub {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Daily tip card */
    .tip-card {
        background: linear-gradient(135deg, #1e1b4b22, #0f172a);
        border: 1px solid #312e8133;
        border-left: 3px solid #a78bfa;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #c4b5fd;
    }

    /* Chat messages */
    .stChatMessage {
        background: #111118 !important;
        border-radius: 12px !important;
        border: 1px solid #1e1e2e !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d0d14;
        border-bottom: 1px solid #1e1e2e;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #6b7280 !important;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        padding: 0.6rem 1.2rem;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        color: #a78bfa !important;
        background: #1a1a2e !important;
        border-bottom: 2px solid #a78bfa !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        letter-spacing: 0.03em;
        transition: all 0.2s;
        padding: 0.5rem 1.2rem;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px #7c3aed44;
    }

    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div {
        background: #111118 !important;
        border: 1px solid #2d2d3d !important;
        color: #e8e8f0 !important;
        border-radius: 8px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d0d14;
        border-right: 1px solid #1e1e2e;
    }

    /* Chat input */
    .stChatInputContainer {
        background: #0d0d14 !important;
        border-top: 1px solid #1e1e2e !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 10px;
        padding: 0.8rem;
    }

    /* Info / success / warning boxes */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
    }

    /* Progress */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4f46e5, #a78bfa) !important;
        border-radius: 4px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #111118 !important;
        border-radius: 8px !important;
        color: #c4b5fd !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: transparent !important;
        border: 1px solid #4f46e5 !important;
        color: #a78bfa !important;
    }

    .stDownloadButton > button:hover {
        background: #4f46e510 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PROCESS QUERY
# ─────────────────────────────────────────────
def process_user_query(user_query: str, lang: str = "en"):
    if not user_query or not user_query.strip():
        return
    msg_id = f"user_{uuid.uuid4().hex[:8]}"
    st.session_state.messages.append({"role": "user", "content": user_query, "id": msg_id})

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("CareerPilot is thinking…"):
            response = careerpilot_chat(user_query)
            resp_id = f"asst_{uuid.uuid4().hex[:8]}"
            st.session_state.messages.append({"role": "assistant", "content": response, "id": resp_id})
            st.write(response)
            display_feedback(resp_id)
            if st.session_state.get("voice_output", False):
                audio = text_to_speech(response[:500], lang)
                if audio:
                    autoplay_audio(audio)

# ─────────────────────────────────────────────
#  PROFILE COMPLETENESS
# ─────────────────────────────────────────────
def profile_completeness(p: dict) -> int:
    fields = ["name", "email", "phone", "job_target"]
    list_fields = ["skills", "education", "work_history"]
    score = sum(1 for f in fields if p.get(f)) + sum(1 for f in list_fields if p.get(f))
    return int((score / (len(fields) + len(list_fields))) * 100)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="CareerPilot — Personal Career Assistant",
        page_icon="✈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        # ── API KEY GATE ─────────────────────
        st.markdown("**🔑 NVIDIA API Key**")
        st.caption("Required to use CareerPilot. Get yours free at [build.nvidia.com](https://build.nvidia.com)")

        user_api_key = st.text_input(
            "Enter your NVIDIA API Key",
            value="",
            type="password",
            placeholder="your_api_key",
            label_visibility="collapsed",
        )

        if user_api_key:
            if user_api_key != st.session_state.get("user_api_key"):
                st.session_state["user_api_key"] = user_api_key
                nvidia_client.api_key = user_api_key
                nvidia_client.available = True
                st.success("✅ API key set!")
            else:
                nvidia_client.api_key = user_api_key
                nvidia_client.available = True
        else:
            nvidia_client.available = False
            nvidia_client.api_key = 

        st.divider()

        # Block the rest of the app if no key provided
        if not nvidia_client.available:
            st.warning("⚠️ Enter your API key above to unlock CareerPilot.")
            st.stop()

        pct = profile_completeness(st.session_state.user_profile)
        st.caption(f"Profile completeness — {pct}%")
        st.progress(pct / 100)

        st.divider()
        st.markdown("**⚙️ Settings**")
        st.session_state["voice_output"] = st.toggle("🔊 Voice output", value=False)
        st.session_state["enable_caching"] = st.toggle("⚡ Cache responses", value=True)

        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        if st.button("🧹 Clear Cache"):
            st.session_state.response_cache = {}
            st.success("Cache cleared.")

        st.divider()
        st.caption(f"Session: `{st.session_state.session_id[:20]}…`")
        st.caption(f"Model: `nvidia/llama-3.1-nemotron-70b`")
        st.caption("🟢 Connected")

    # ── TABS ─────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Chat", "👤 Profile", "📄 Resume Builder",
        "🎯 Interview Prep", "✉️ Cover Letter", "📬 Contact",
    ])

    # ══════════════════════════════════════════
    #  TAB 1 — CHAT
    # ══════════════════════════════════════════
    with tab1:
        st.markdown('<h1 class="nova-title">✈ CareerPilot</h1>', unsafe_allow_html=True)
        st.markdown('<p class="nova-sub">Your AI-Powered Personal Career Assistant</p>', unsafe_allow_html=True)

        # Daily tip
        st.markdown(f'<div class="tip-card">✦ Daily Tip &nbsp;·&nbsp; {get_daily_tip()}</div>', unsafe_allow_html=True)

        # Quick action chips
        st.markdown("**Quick actions:**")
        qcols = st.columns(4)
        quick_prompts = [
            ("🔍 Job Search Tips", "Give me 5 actionable job search strategies for today's market."),
            ("📝 Improve My Resume", "What are the biggest mistakes people make on resumes?"),
            ("🤝 Networking Script", "Write me a short LinkedIn connection request message for reaching out to someone in my target industry."),
            ("💰 Salary Negotiation", "How should I negotiate a salary offer? Give me a script."),
        ]
        for col, (label, prompt) in zip(qcols, quick_prompts):
            with col:
                if st.button(label, use_container_width=True):
                    process_user_query(prompt)

        st.divider()

        # Chat history
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant":
                    display_feedback(msg.get("id", f"msg_{idx}"))

        # Input row
        col_input, col_lang = st.columns([6, 1.5])
        with col_lang:
            lang = st.selectbox(
                "Lang", options=list(SUPPORTED_LANGUAGES.keys()),
                format_func=lambda x: SUPPORTED_LANGUAGES[x],
                label_visibility="collapsed",
            )

        with col_input:
            user_input = st.chat_input("Ask CareerPilot anything about your career…")
            if user_input:
                process_user_query(user_input, lang)

    # ══════════════════════════════════════════
    #  TAB 2 — PROFILE
    # ══════════════════════════════════════════
    with tab2:
        st.markdown("## 👤 My Profile")
        st.caption("Keep this up-to-date for personalised advice and AI-generated documents.")

        p = st.session_state.user_profile

        with st.form("profile_form"):
            st.subheader("Personal Info")
            c1, c2 = st.columns(2)
            with c1:
                name  = st.text_input("Full Name", value=p.get("name", ""))
                email = st.text_input("Email", value=p.get("email", ""))
                phone = st.text_input("Phone", value=p.get("phone", ""))
            with c2:
                job_target  = st.text_input("Target Job Title", value=p.get("job_target", ""))
                experience  = st.slider("Years of Experience", 0, 40, value=p.get("experience", 0))
                preferred_lang = st.selectbox(
                    "Preferred Language",
                    list(SUPPORTED_LANGUAGES.keys()),
                    format_func=lambda x: SUPPORTED_LANGUAGES[x],
                    index=list(SUPPORTED_LANGUAGES.keys()).index(p.get("preferred_language", "en")),
                )

            st.subheader("Online Presence")
            c3, c4, c5 = st.columns(3)
            with c3: linkedin  = st.text_input("LinkedIn URL", value=p.get("linkedin", ""))
            with c4: github    = st.text_input("GitHub URL", value=p.get("github", ""))
            with c5: portfolio = st.text_input("Portfolio URL", value=p.get("portfolio", ""))

            if st.form_submit_button("💾 Save Profile"):
                st.session_state.user_profile.update({
                    "name": name, "email": email, "phone": phone,
                    "job_target": job_target, "experience": experience,
                    "preferred_language": preferred_lang,
                    "linkedin": linkedin, "github": github, "portfolio": portfolio,
                })
                st.success("Profile saved!")

        # Skills
        st.subheader("🛠️ Skills")
        with st.form("skill_form"):
            c1, c2 = st.columns([4, 1])
            with c1: new_skill = st.text_input("Add a skill", label_visibility="collapsed", placeholder="e.g. Python, Product Management, UX Research")
            with c2: add_skill = st.form_submit_button("Add")
            if add_skill and new_skill:
                p.setdefault("skills", []).append(new_skill.strip())
                st.rerun()

        if p.get("skills"):
            cols = st.columns(4)
            for i, skill in enumerate(p["skills"]):
                with cols[i % 4]:
                    if st.button(f"✕  {skill}", key=f"rm_sk_{i}", use_container_width=True):
                        p["skills"].pop(i)
                        st.rerun()

        # Education
        st.subheader("🎓 Education")
        if p.get("education"):
            for i, edu in enumerate(p["education"]):
                with st.expander(f"{edu.get('degree','?')} — {edu.get('institution','')}"):
                    c1, c2, c3 = st.columns([3, 3, 2])
                    with c1: deg  = st.text_input("Degree", edu.get("degree",""), key=f"ed_d_{i}")
                    with c2: inst = st.text_input("Institution", edu.get("institution",""), key=f"ed_i_{i}")
                    with c3: yr   = st.text_input("Year", edu.get("year",""), key=f"ed_y_{i}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Update", key=f"upd_edu_{i}"):
                            p["education"][i] = {"degree": deg, "institution": inst, "year": yr}
                            st.success("Updated!")
                    with col_b:
                        if st.button("Remove", key=f"rm_edu_{i}"):
                            p["education"].pop(i)
                            st.rerun()

        with st.form("edu_form"):
            c1, c2, c3 = st.columns([3, 3, 2])
            with c1: nd = st.text_input("Degree")
            with c2: ni = st.text_input("Institution")
            with c3: ny = st.text_input("Year")
            if st.form_submit_button("➕ Add Education") and nd and ni:
                p.setdefault("education", []).append({"degree": nd, "institution": ni, "year": ny})
                st.success("Added!")

        # Work History
        st.subheader("💼 Work History")
        if p.get("work_history"):
            for i, w in enumerate(p["work_history"]):
                with st.expander(f"{w.get('title','?')} @ {w.get('company','')}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        t  = st.text_input("Title", w.get("title",""), key=f"wt_{i}")
                        co = st.text_input("Company", w.get("company",""), key=f"wc_{i}")
                    with c2:
                        du = st.text_input("Duration", w.get("duration",""), key=f"wd_{i}")
                        de = st.text_area("Key Achievements", w.get("description",""), key=f"wde_{i}")
                    ca, cb = st.columns(2)
                    with ca:
                        if st.button("Update", key=f"upd_wk_{i}"):
                            p["work_history"][i] = {"title": t, "company": co, "duration": du, "description": de}
                            st.success("Updated!")
                    with cb:
                        if st.button("Remove", key=f"rm_wk_{i}"):
                            p["work_history"].pop(i)
                            st.rerun()

        with st.form("work_form"):
            c1, c2 = st.columns(2)
            with c1:
                nt = st.text_input("Job Title")
                nc = st.text_input("Company")
            with c2:
                ndu = st.text_input("Duration (e.g. 2021–2024)")
                nde = st.text_area("Key Achievements")
            if st.form_submit_button("➕ Add Work Experience") and nt and nc:
                p.setdefault("work_history", []).append({"title": nt, "company": nc, "duration": ndu, "description": nde})
                st.success("Added!")

    # ══════════════════════════════════════════
    #  TAB 3 — RESUME BUILDER
    # ══════════════════════════════════════════
    with tab3:
        st.markdown("## 📄 Resume Builder")

        option = st.radio("Mode:", ["✨ AI-Generate Resume", "📂 Upload & Analyse"], horizontal=True)

        if option == "✨ AI-Generate Resume":
            p = st.session_state.user_profile
            ready = p.get("name") and p.get("skills") and p.get("education") and p.get("work_history")

            if not ready:
                st.warning("Complete your Profile tab (name, skills, education, work history) before generating.")
            else:
                if st.button("🚀 Generate My Resume", use_container_width=True):
                    with st.spinner("Crafting your resume with NVIDIA Nemotron…"):
                        resume_md = generate_resume(p)
                        st.session_state.user_profile["resume_data"] = resume_md

                if p.get("resume_data"):
                    st.markdown("---")
                    st.markdown(p["resume_data"])
                    fn = (p.get("name") or "resume").replace(" ", "_")
                    st.download_button(
                        "⬇️ Download Resume (Markdown)",
                        data=p["resume_data"],
                        file_name=f"{fn}_resume.md",
                        mime="text/markdown",
                    )

        else:
            uploaded = st.file_uploader("Upload your resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
            if uploaded:
                st.success(f"Uploaded: **{uploaded.name}**")
                if st.button("🔍 Analyse Resume"):
                    with st.spinner("Analysing with NVIDIA Nemotron…"):
                        # Stub parser — replace with real parser (pdfminer, python-docx, etc.)
                        parsed = {
                            "name": st.session_state.user_profile.get("name", "Unknown"),
                            "skills": st.session_state.user_profile.get("skills", []),
                            "education": st.session_state.user_profile.get("education", []),
                            "work_history": st.session_state.user_profile.get("work_history", []),
                        }
                        feedback = analyze_resume_feedback(parsed)
                        st.subheader("🎯 AI Feedback")
                        st.markdown(feedback)

    # ══════════════════════════════════════════
    #  TAB 4 — INTERVIEW PREP  (NEW ✨)
    # ══════════════════════════════════════════
    with tab4:
        st.markdown("## 🎯 Interview Prep")
        st.caption("AI-generated questions tailored to your target role and skills.")

        p = st.session_state.user_profile
        job_title_input = st.text_input(
            "Target Job Title",
            value=p.get("job_target", ""),
            placeholder="e.g. Senior Data Scientist",
        )
        skills_input = st.text_input(
            "Your Key Skills (comma-separated)",
            value=", ".join(p.get("skills", [])),
            placeholder="e.g. Python, Machine Learning, SQL",
        )

        if st.button("🎲 Generate Interview Questions", use_container_width=True):
            if job_title_input:
                with st.spinner("Generating personalised questions…"):
                    skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]
                    questions = generate_interview_questions(job_title_input, skills_list)
                    st.session_state["interview_questions"] = questions
            else:
                st.warning("Please enter a target job title.")

        if st.session_state.get("interview_questions"):
            st.markdown("---")
            st.markdown(st.session_state["interview_questions"])
            st.download_button(
                "⬇️ Download Questions",
                data=st.session_state["interview_questions"],
                file_name="interview_questions.md",
                mime="text/markdown",
            )

        st.divider()
        st.subheader("🗣️ Practice Mode")
        st.caption("Type your answer below and CareerPilot will give you instant feedback.")
        practice_q = st.text_area("Paste an interview question:")
        practice_a = st.text_area("Your answer:")
        if st.button("📊 Get Feedback on My Answer"):
            if practice_q and practice_a:
                with st.spinner("Evaluating…"):
                    fb = careerpilot_chat(
                        f"Interview question: {practice_q}\n\nMy answer: {practice_a}\n\nGive structured feedback: strengths, weaknesses, and an improved version of my answer.",
                        max_tokens=600,
                    )
                    st.markdown(fb)
            else:
                st.warning("Enter both a question and your answer.")

    # ══════════════════════════════════════════
    #  TAB 5 — COVER LETTER  (NEW ✨)
    # ══════════════════════════════════════════
    with tab5:
        st.markdown("## ✉️ Cover Letter Generator")
        st.caption("Paste a job description and CareerPilot writes a tailored cover letter in seconds.")

        job_desc = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste the full job description here…",
        )
        tone = st.select_slider(
            "Tone",
            options=["Formal", "Balanced", "Conversational"],
            value="Balanced",
        )

        if st.button("✍️ Generate Cover Letter", use_container_width=True):
            if job_desc:
                p = st.session_state.user_profile
                if not p.get("name"):
                    st.warning("Add your name in the Profile tab first.")
                else:
                    with st.spinner("Writing your cover letter…"):
                        letter = generate_cover_letter(p, f"[Tone: {tone}]\n{job_desc}")
                        st.session_state["cover_letter"] = letter
            else:
                st.warning("Please paste a job description.")

        if st.session_state.get("cover_letter"):
            st.markdown("---")
            st.markdown(st.session_state["cover_letter"])
            fn = (st.session_state.user_profile.get("name") or "cover_letter").replace(" ", "_")
            st.download_button(
                "⬇️ Download Cover Letter",
                data=st.session_state["cover_letter"],
                file_name=f"{fn}_cover_letter.md",
                mime="text/markdown",
            )

    # ══════════════════════════════════════════
    #  TAB 6 — CONTACT
    # ══════════════════════════════════════════
    with tab6:
        st.markdown("## 📬 Contact")

        if st.session_state.contact_form_submitted:
            st.success("✅ Message sent! I'll get back to you soon.")
            if st.button("Send another message"):
                st.session_state.contact_form_submitted = False
                st.rerun()
        else:
            with st.form("contact_form"):
                p = st.session_state.user_profile
                c1, c2 = st.columns(2)
                with c1: cn = st.text_input("Name", value=p.get("name",""))
                with c2: ce = st.text_input("Email", value=p.get("email",""))
                cs = st.selectbox("Subject", [
                    "General Inquiry", "Bug Report",
                    "Feature Request", "Resume Review",
                    "Feedback on CareerPilot", "Other",
                ])
                cm = st.text_area("Message", height=140)
                if st.form_submit_button("📤 Send Message"):
                    if cn and ce and cm:
                        st.session_state.contact_form_submitted = True
                        st.session_state.contact_data = {
                            "name": cn, "email": ce, "subject": cs,
                            "message": cm, "ts": datetime.now().isoformat(),
                        }
                        st.rerun()
                    else:
                        st.error("Please fill in all fields.")

            st.divider()
            st.markdown("**Direct contact**")
            st.write("📧 your@email.com")
            st.write("🌐 yourportfolio.com")
            st.write("⏰ Response time: within 48 hours")


if __name__ == "__main__":
    main()
