import streamlit as st
import pandas as pd
import json
import requests
import os
from datetime import datetime
import time
import hashlib
import threading
import queue
import base64
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
import uuid
import random
import tempfile
import shutil
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()

if "is_listening" not in st.session_state:
    st.session_state.is_listening = False

if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "name": "",
        "email": "",
        "phone": "",
        "experience": 0,
        "skills": [],
        "education": [],
        "work_history": [],
        "preferred_language": "en",
        "resume_data": None
    }

if "feedback_ratings" not in st.session_state:
    st.session_state.feedback_ratings = {}

if "contact_form_submitted" not in st.session_state:
    st.session_state.contact_form_submitted = False

# API Configuration
class JobsForHerAPI:
    def __init__(self):
        self.base_url = os.getenv("JFH_API_BASE_URL", "https://api.jobsforher.com/v1")
        self.api_key = os.getenv("JFH_API_KEY", "")
        self.api_secret = os.getenv("JFH_API_SECRET", "")
        self.api_available = bool(self.api_key and self.api_secret)
        
    def generate_auth_token(self):
        timestamp = str(int(time.time()))
        signature = hashlib.sha256(f"{self.api_key}{timestamp}{self.api_secret}".encode()).hexdigest()
        return {
            "X-API-Key": self.api_key,
            "X-Timestamp": timestamp,
            "X-Signature": signature
        }
    
    def get_job_listings(self, limit=10, offset=0, filters=None):
        if not self.api_available:
            return self._get_sample_job_data()
        try:
            endpoint = f"{self.base_url}/jobs"
            headers = self.generate_auth_token()
            params = {"limit": limit, "offset": offset}
            if filters:
                params.update(filters)
            response = requests.get(endpoint, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to fetch job listings: {response.status_code}")
                return self._get_sample_job_data()
        except Exception as e:
            st.error(f"Error connecting to JobsForHer API: {str(e)}")
            return self._get_sample_job_data()
    
    def get_events(self, limit=10, offset=0, filters=None):
        if not self.api_available:
            return self._get_sample_event_data()
        try:
            endpoint = f"{self.base_url}/events"
            headers = self.generate_auth_token()
            params = {"limit": limit, "offset": offset}
            if filters:
                params.update(filters)
            response = requests.get(endpoint, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to fetch events: {response.status_code}")
                return self._get_sample_event_data()
        except Exception as e:
            st.error(f"Error connecting to JobsForHer API: {str(e)}")
            return self._get_sample_event_data()
    
    def search_resources(self, query, limit=5):
        if not self.api_available:
            return self._get_sample_resources()
        try:
            endpoint = f"{self.base_url}/resources/search"
            headers = self.generate_auth_token()
            params = {"query": query, "limit": limit}
            response = requests.get(endpoint, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return self._get_sample_resources()
        except Exception as e:
            return self._get_sample_resources()
    
    def _get_sample_job_data(self):
        try:
            df = pd.read_csv("data/job_listing_data.csv")
            return {"jobs": df.to_dict(orient='records')}
        except FileNotFoundError:
            sample_data = {
                "jobs": [
                    {
                        "job_id": 1,
                        "title": "Software Engineer",
                        "company": "TechCorp",
                        "location": "Bangalore",
                        "job_type": "Full-time",
                        "description": "Develop and maintain software applications",
                        "requirements": "Bachelor's in Computer Science, 2+ years experience",
                        "posted_date": "2023-05-15"
                    },
                    {
                        "job_id": 2,
                        "title": "Data Analyst",
                        "company": "DataInsights",
                        "location": "Mumbai",
                        "job_type": "Part-time",
                        "description": "Analyze data and create insights",
                        "requirements": "Experience with SQL, Python, and data visualization",
                        "posted_date": "2023-05-16"
                    },
                    {
                        "job_id": 3,
                        "title": "Project Manager",
                        "company": "ManagementPro",
                        "location": "Remote",
                        "job_type": "Full-time",
                        "description": "Lead project teams and ensure timely delivery",
                        "requirements": "PMP certification, 5+ years in project management",
                        "posted_date": "2023-05-17"
                    }
                ]
            }
            return sample_data
    
    def _get_sample_event_data(self):
        try:
            with open("data/session_details.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            sample_events = {
                "sessions": [
                    {
                        "id": 1,
                        "title": "Breaking the Glass Ceiling",
                        "description": "Discussion on overcoming workplace barriers",
                        "date": "2023-06-15T14:00:00",
                        "duration": "60 minutes",
                        "speaker": "Dr. Maya Patel"
                    },
                    {
                        "id": 2,
                        "title": "Resume Building Workshop",
                        "description": "Learn how to create an effective resume",
                        "date": "2023-06-20T11:00:00",
                        "duration": "90 minutes",
                        "speaker": "Sarah Johnson"
                    },
                    {
                        "id": 3,
                        "title": "Tech Career Pathways",
                        "description": "Exploring opportunities in technology",
                        "date": "2023-06-25T16:00:00",
                        "duration": "75 minutes",
                        "speaker": "Lisa Wang"
                    }
                ]
            }
            return sample_events
    
    def _get_sample_resources(self):
        sample_resources = {
            "resources": [
                {
                    "id": 1,
                    "title": "Negotiation Skills for Women",
                    "type": "Article",
                    "description": "Learn effective strategies for salary negotiation"
                },
                {
                    "id": 2,
                    "title": "Building Your Personal Brand",
                    "type": "Video",
                    "description": "How to create a powerful professional image"
                },
                {
                    "id": 3,
                    "title": "Women in Leadership",
                    "type": "E-book",
                    "description": "Stories and strategies from successful women leaders"
                }
            ]
        }
        return sample_resources

# Load bias prevention rules
@st.cache_data
def load_bias_rules():
    bias_rules = {
        "flagged_terms": [
            "women can't", "women are not", "females can't", "girls can't",
            "women should stay", "woman's place", "belongs in the kitchen",
            "too emotional", "not technical enough"
        ],
        "redirect_responses": [
            "I'd like to share that women have been successful leaders across industries. Would you like to explore some leadership success stories instead?",
            "Research shows diverse teams perform better. Would you like to learn about the positive impact of women in various professional roles?",
            "I'm here to provide information that supports career growth for everyone. Can I help you with job listings or professional development resources?"
        ]
    }
    return bias_rules

# Response caching decorator
def cache_response(func):
    def wrapper(prompt, system_prompt=None):
        cache_key = f"{prompt}_{system_prompt}"
        if cache_key in st.session_state.response_cache:
            return st.session_state.response_cache[cache_key]
        result = func(prompt, system_prompt)
        st.session_state.response_cache[cache_key] = result
        return result
    return wrapper

# Function to query Ollama with caching
@cache_response
def query_ollama(prompt, system_prompt=None):
    url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    if system_prompt is None:
        system_prompt = """
        You are Asha, an AI assistant for the JobsForHer Foundation. Your purpose is to help women with their career journeys by providing information about job listings, community events, sessions, mentorship programs, and addressing FAQs.
        You should always:
        1. Be supportive, professional, and empowering in your responses
        2. Provide factual information about careers, jobs, and professional development
        3. Highlight women's achievements and capabilities in the workplace
        4. Avoid any gender stereotypes or biases
        5. Focus on being helpful with job search and career advancement information
        6. Keep responses concise and to-the-point for better voice readability
        You are integrated with the JobsForHer.com platform and can provide information about real job listings, events, and resources available on the platform.
        When discussing job opportunities, mention that users can find more details by visiting the JobsForHer website.
        """
    data = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "I apologize, but I couldn't generate a response.")
        else:
            return f"Error: Received status code {response.status_code} from Ollama service."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama service: {str(e)}. Make sure Ollama is running with '{model}' model."

# Check for bias in user input
def check_for_bias(user_input, bias_rules):
    user_input_lower = user_input.lower()
    for term in bias_rules["flagged_terms"]:
        if term in user_input_lower:
            return random.choice(bias_rules["redirect_responses"])
    return None

# Function to process user query considering context
def process_query(query, api_client, bias_rules):
    bias_response = check_for_bias(query, bias_rules)
    if bias_response:
        return bias_response
    job_data = {"jobs": []}
    event_data = {"sessions": []}
    resource_data = {"resources": []}
    def fetch_jobs():
        nonlocal job_data
        job_data = api_client.get_job_listings(limit=5)
    def fetch_events():
        nonlocal event_data
        event_data = api_client.get_events(limit=5)
    def fetch_resources():
        nonlocal resource_data
        resource_data = api_client.search_resources(query, limit=3)
    threads = [
        threading.Thread(target=fetch_jobs),
        threading.Thread(target=fetch_events),
        threading.Thread(target=fetch_resources)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    user_profile_data = ""
    if st.session_state.user_profile["name"]:
        user_profile_data = f"""
        User Profile Information:
        Name: {st.session_state.user_profile["name"]}
        Experience: {st.session_state.user_profile["experience"]} years
        Skills: {", ".join(st.session_state.user_profile["skills"])}
        Education: {json.dumps(st.session_state.user_profile["education"])}
        Work History: {json.dumps(st.session_state.user_profile["work_history"])}
        Preferred Language: {st.session_state.user_profile["preferred_language"]}
        """
    context_prompt = f"""
    User query: {query}
    Available job listings (sample):
    {json.dumps(job_data.get('jobs', [])[:3], indent=2)}
    Available sessions and events (sample):
    {json.dumps(event_data.get('sessions', [])[:3], indent=2)}
    Relevant resources:
    {json.dumps(resource_data.get('resources', []), indent=2)}
    {user_profile_data}
    Current date: {datetime.now().strftime('%Y-%m-%d')}
    Session ID: {st.session_state.session_id}
    Previous conversation:
    {format_conversation_history()}
    Please provide a helpful, concise response to the user's query based on the context provided.
    Keep your response brief and to the point for better voice readability.
    Remember to mention that more information can be found on the JobsForHer website for specific job listings or events.
    If the user is asking about resume building or needs help with their resume, refer to their profile information if available.
    """
    response = query_ollama(context_prompt)
    return response

# Format conversation history for context
def format_conversation_history():
    history = ""
    for msg in st.session_state.messages[-5:]:
        role = "User" if msg["role"] == "user" else "Asha"
        history += f"{role}: {msg['content']}\n"
    return history

# Languages supported for voice input/output
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi"
}

# Voice-related functions
def text_to_speech(text, lang_code="en"):
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        if lang_code != "en":
            st.warning("Falling back to English for voice output")
            return text_to_speech(text, "en")
        return None

def autoplay_audio(audio_data):
    b64 = base64.b64encode(audio_data.getvalue()).decode()
    md = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

def speech_to_text(lang_code="en"):
    try:
        r = sr.Recognizer()
        r.energy_threshold = 4000
        r.dynamic_energy_threshold = True
        st.info(f"🎙️ Listening in {SUPPORTED_LANGUAGES.get(lang_code, 'English')}... Speak now")
        st.session_state.is_listening = True
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = r.listen(source, timeout=8, phrase_time_limit=15)
                try:
                    text = r.recognize_google(audio, language=lang_code)
                    st.success(f"Recognized: {text}")
                    st.session_state.is_listening = False
                    return text
                except sr.UnknownValueError:
                    st.warning("I couldn't understand what you said. Please try again.")
                    st.session_state.is_listening = False
                    return None
                except sr.RequestError as e:
                    st.error(f"Speech recognition service error: {e}")
                    st.session_state.is_listening = False
                    return None
            except sr.WaitTimeoutError:
                st.warning("I didn't hear anything. Please try again when you're ready to speak.")
                st.session_state.is_listening = False
                return None
    except Exception as e:
        st.error(f"Microphone error: {str(e)}")
        st.session_state.is_listening = False
        return None

# Resume Builder functions
def generate_ai_resume(profile_data):
    system_prompt = """
    You are an expert resume writer with experience creating professional resumes for women in various industries.
    Your task is to create a well-formatted, professional resume for the user based on their profile information.
    The resume should highlight their skills, experience, and education effectively.
    Format the resume in Markdown with clear sections for:
    - Contact Information
    - Professional Summary
    - Skills
    - Work Experience (with bullet points for responsibilities and achievements)
    - Education
    - Additional sections as appropriate (certifications, volunteer work, etc.)
    Keep the content professional, concise, and impactful. Focus on achievements and results where possible.
    """
    profile_json = json.dumps(profile_data, indent=2)
    prompt = f"""
    Please create a professional resume based on the following profile information:
    {profile_json}
    Create a resume that will help this person stand out to potential employers.
    Focus on their strengths and format the resume in a clean, professional way.
    """
    resume_content = query_ollama(prompt, system_prompt)
    return resume_content

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                shutil.copyfileobj(uploaded_file, tmp)
                return tmp.name
        except Exception as e:
            st.error(f"Error saving uploaded file: {str(e)}")
            return None
    return None

def parse_resume(file_path):
    try:
        return {
            "name": "Sample Name",
            "email": "sample@email.com",
            "phone": "1234567890",
            "skills": ["Communication", "Leadership", "Project Management"],
            "education": [
                {
                    "degree": "Bachelor of Science",
                    "institution": "Sample University",
                    "year": "2018-2022"
                }
            ],
            "work_history": [
                {
                    "title": "Project Manager",
                    "company": "Sample Company",
                    "duration": "2022-Present",
                    "description": "Led cross-functional teams in delivering projects on time and under budget."
                }
            ]
        }
    except Exception as e:
        st.error(f"Error parsing resume: {str(e)}")
        return None

# User feedback component
def display_feedback_component(message_id):
    feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 3, 1])
    with feedback_col2:
        st.write("Was this response helpful?")
        existing_rating = st.session_state.feedback_ratings.get(message_id, None)
        if existing_rating is not None:
            if existing_rating == "thumbs_up":
                st.success("👍 Thank you for your positive feedback!")
            elif existing_rating == "thumbs_down":
                st.error("👎 Thank you for your feedback. We'll work to improve.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Yes", key=f"thumbs_up_{message_id}"):
                    st.session_state.feedback_ratings[message_id] = "thumbs_up"
                    st.rerun()
            with col2:
                if st.button("👎 No", key=f"thumbs_down_{message_id}"):
                    st.session_state.feedback_ratings[message_id] = "thumbs_down"
                    st.rerun()

# Contact form submission handler
def handle_contact_form(name, email, message, subject):
    contact_data = {
        "name": name,
        "email": email,
        "subject": subject,
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.contact_form_submitted = True
    st.session_state.contact_data = contact_data
    return True

# Main app interface
def main():
    st.set_page_config(
        page_title="Asha - JobsForHer AI Assistant",
        page_icon="👩‍💼",
        layout="wide"
    )
    tab1, tab2, tab3, tab4 = st.tabs(["Chat with Asha", "My Profile", "Resume Builder", "Contact Us"])
    
    with tab1:
        st.title("Asha - JobsForHer AI Assistant")
        st.caption("Your AI guide for career development and job opportunities")
        api_client = JobsForHerAPI()
        if not api_client.api_available:
            st.sidebar.warning("⚠️ Running with sample data. To connect to JobsForHer API, please add API credentials in .env file.")
        else:
            st.sidebar.success("✅ Connected to JobsForHer API")
        bias_rules = load_bias_rules()
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant":
                    message_id = message.get("id", f"msg_{idx}")
                    display_feedback_component(message_id)
        col1, col2, col3 = st.columns([5, 2, 1])
        with col2:
            lang_code = st.selectbox(
                "Language", 
                options=list(SUPPORTED_LANGUAGES.keys()), 
                format_func=lambda x: SUPPORTED_LANGUAGES[x],
                index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.user_profile.get("preferred_language", "en"))
            )
            st.session_state.user_profile["preferred_language"] = lang_code
        with col3:
            if st.button("🎤 Voice", key="voice_button"):
                user_query = speech_to_text(lang_code)
                if user_query and len(user_query.strip()) > 0:
                    process_user_query(user_query, api_client, bias_rules, lang_code)
        with col1:
            user_query = st.chat_input("How can I help you with your career journey today?")
            if user_query:
                process_user_query(user_query, api_client, bias_rules, lang_code)
        with st.sidebar:
            voice_output = st.toggle("Enable Voice Output", value=True, key="auto_play_voice")
            st.header("About Asha")
            st.info("""
            Asha is your AI assistant for the JobsForHer Foundation. 
            I can help you with:
            - Finding job opportunities
            - Information about upcoming events and sessions
            - Career advice and resources
            - Women empowerment initiatives
            - Resume building and career guidance
            I'm here to support your professional journey!
            """)
            st.header("Voice Settings")
            st.header("API Configuration")
            if st.checkbox("Show API Settings", value=False):
                with st.form("api_settings"):
                    api_url = st.text_input("JobsForHer API URL", value=os.getenv("JFH_API_BASE_URL", "https://api.jobsforher.com/v1"))
                    api_key = st.text_input("API Key", value=os.getenv("JFH_API_KEY", ""), type="password")
                    api_secret = st.text_input("API Secret", value=os.getenv("JFH_API_SECRET", ""), type="password")
                    ollama_url = st.text_input("Ollama API URL", value=os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate"))
                    ollama_model = st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "llama3.2"))
                    if st.form_submit_button("Save Configuration"):
                        with open(".env", "w") as env_file:
                            env_file.write(f"JFH_API_BASE_URL={api_url}\n")
                            env_file.write(f"JFH_API_KEY={api_key}\n")
                            env_file.write(f"JFH_API_SECRET={api_secret}\n")
                            env_file.write(f"OLLAMA_API_URL={ollama_url}\n")
                            env_file.write(f"OLLAMA_MODEL={ollama_model}\n")
                        st.success("Configuration saved! Please restart the application for changes to take effect.")
            st.header("Performance Options")
            st.checkbox("Enable response caching for faster replies", value=True, key="enable_caching")
            if st.button("Clear Cache"):
                st.session_state.response_cache = {}
                st.success("Response cache cleared!")
            st.header("Session Information")
            st.text(f"Session ID: {st.session_state.session_id}")
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
    
    with tab2:
        st.title("My Profile")
        st.caption("Manage your professional information to get personalized assistance")
        with st.form("user_profile_form"):
            st.subheader("Personal Information")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name", value=st.session_state.user_profile.get("name", ""))
                email = st.text_input("Email", value=st.session_state.user_profile.get("email", ""))
            with col2:
                phone = st.text_input("Phone", value=st.session_state.user_profile.get("phone", ""))
                experience = st.slider("Years of Experience", min_value=0, max_value=30, value=st.session_state.user_profile.get("experience", 0))
            st.subheader("Communication Preferences")
            preferred_lang = st.selectbox(
                "Preferred Language",
                options=list(SUPPORTED_LANGUAGES.keys()),
                format_func=lambda x: SUPPORTED_LANGUAGES[x],
                index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.user_profile.get("preferred_language", "en"))
            )
            if st.form_submit_button("Save Profile"):
                st.session_state.user_profile.update({
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "experience": experience,
                    "preferred_language": preferred_lang
                })
                st.success("Profile saved successfully!")
        
        st.subheader("Skills")
        with st.form("add_skill_form"):
            skill_input = st.text_input("Add a skill")
            if st.form_submit_button("Add Skill"):
                if skill_input:
                    if "skills" not in st.session_state.user_profile:
                        st.session_state.user_profile["skills"] = []
                    st.session_state.user_profile["skills"].append(skill_input)
                    st.success(f"Skill '{skill_input}' added!")
        if "skills" in st.session_state.user_profile and len(st.session_state.user_profile["skills"]) > 0:
            st.write("Your skills:")
            cols = st.columns(3)
            for i, skill in enumerate(st.session_state.user_profile["skills"]):
                col_idx = i % 3
                with cols[col_idx]:
                    if st.button(f"❌ {skill}", key=f"remove_skill_{i}"):
                        st.session_state.user_profile["skills"].remove(skill)
                        st.rerun()
        
        st.subheader("Education")
        if "education" in st.session_state.user_profile and len(st.session_state.user_profile["education"]) > 0:
            for i, edu in enumerate(st.session_state.user_profile["education"]):
                with st.expander(f"{edu.get('degree', 'Education')} - {edu.get('institution', '')}"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        degree = st.text_input("Degree", value=edu.get("degree", ""), key=f"edu_degree_{i}")
                    with col2:
                        institution = st.text_input("Institution", value=edu.get("institution", ""), key=f"edu_inst_{i}")
                    with col3:
                        year = st.text_input("Year", value=edu.get("year", ""), key=f"edu_year_{i}")
                    if st.button("Remove", key=f"remove_edu_{i}"):
                        st.session_state.user_profile["education"].pop(i)
                        st.rerun()
                    if st.button("Update", key=f"update_edu_{i}"):
                        st.session_state.user_profile["education"][i] = {
                            "degree": degree,
                            "institution": institution,
                            "year": year
                        }
                        st.success("Education updated!")
        with st.form("add_education_form"):
            st.write("Add Education")
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                new_degree = st.text_input("Degree")
            with col2:
                new_institution = st.text_input("Institution")
            with col3:
                new_year = st.text_input("Year")
            if st.form_submit_button("Add Education"):
                if new_degree and new_institution:
                    if "education" not in st.session_state.user_profile:
                        st.session_state.user_profile["education"] = []
                    st.session_state.user_profile["education"].append({
                        "degree": new_degree,
                        "institution": new_institution,
                        "year": new_year
                    })
                    st.success("Education added!")
        
        st.subheader("Work History")
        if "work_history" in st.session_state.user_profile and len(st.session_state.user_profile["work_history"]) > 0:
            for i, work in enumerate(st.session_state.user_profile["work_history"]):
                with st.expander(f"{work.get('title', 'Position')} at {work.get('company', '')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        title = st.text_input("Job Title", value=work.get("title", ""), key=f"work_title_{i}")
                        company = st.text_input("Company", value=work.get("company", ""), key=f"work_company_{i}")
                    with col2:
                        duration = st.text_input("Duration", value=work.get("duration", ""), key=f"work_duration_{i}")
                        description = st.text_area("Description", value=work.get("description", ""), key=f"work_desc_{i}")
                    if st.button("Remove", key=f"remove_work_{i}"):
                        st.session_state.user_profile["work_history"].pop(i)
                        st.rerun()
                    if st.button("Update", key=f"update_work_{i}"):
                        st.session_state.user_profile["work_history"][i] = {
                            "title": title,
                            "company": company,
                            "duration": duration,
                            "description": description
                        }
                        st.success("Work history updated!")
        with st.form("add_work_form"):
            st.write("Add Work Experience")
            col1, col2 = st.columns(2)
            with col1:
                new_title = st.text_input("Job Title")
                new_company = st.text_input("Company")
            with col2:
                new_duration = st.text_input("Duration (e.g., 2020-2023)")
                new_description = st.text_area("Description")
            if st.form_submit_button("Add Work Experience"):
                if new_title and new_company:
                    if "work_history" not in st.session_state.user_profile:
                        st.session_state.user_profile["work_history"] = []
                    st.session_state.user_profile["work_history"].append({
                        "title": new_title,
                        "company": new_company,
                        "duration": new_duration,
                        "description": new_description
                    })
                    st.success("Work experience added!")
    
    with tab3:
        st.title("Resume Builder")
        st.caption("Create a professional resume tailored to your skills and experience")
        option = st.radio("Choose an option:", ["Generate AI Resume", "Upload Existing Resume"])
        if option == "Generate AI Resume":
            st.info("Let our AI create a professional resume based on your profile information")
            profile_complete = (
                st.session_state.user_profile.get("name", "") != "" and
                len(st.session_state.user_profile.get("skills", [])) > 0 and
                len(st.session_state.user_profile.get("education", [])) > 0 and
                len(st.session_state.user_profile.get("work_history", [])) > 0
            )
            if not profile_complete:
                st.warning("Please complete your profile in the 'My Profile' tab before generating a resume")
            else:
                if st.button("Generate Resume"):
                    with st.spinner("Generating your professional resume..."):
                        resume_content = generate_ai_resume(st.session_state.user_profile)
                        st.session_state.user_profile["resume_data"] = resume_content
                if "resume_data" in st.session_state.user_profile and st.session_state.user_profile["resume_data"]:
                    st.subheader("Your Generated Resume")
                    st.markdown(st.session_state.user_profile["resume_data"])
                    resume_download = st.session_state.user_profile["resume_data"]
                    st.download_button(
                        label="Download Resume (Markdown)",
                        data=resume_download,
                        file_name=f"{st.session_state.user_profile['name'].replace(' ', '_')}_resume.md",
                        mime="text/markdown",
                    )
        else:
            st.info("Upload your existing resume for analysis and suggestions")
            uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
            if uploaded_file is not None:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    st.success("Resume uploaded successfully!")
                    if st.button("Analyze Resume"):
                        with st.spinner("Analyzing your resume..."):
                            parsed_data = parse_resume(file_path)
                            if parsed_data:
                                st.session_state.user_profile.update(parsed_data)
                                st.success("Resume analyzed successfully! Profile information has been updated.")
                                st.subheader("Extracted Information")
                                st.json(parsed_data)
                                st.subheader("AI Feedback on Your Resume")
                                feedback_prompt = f"""
                                Provide professional feedback on this resume:
                                {json.dumps(parsed_data, indent=2)}
                                Focus on:
                                1. Strengths of the resume
                                2. Areas for improvement
                                3. Specific suggestions to make it more appealing to employers
                                Keep your feedback concise, constructive, and actionable.
                                """
                                feedback = query_ollama(feedback_prompt)
                                st.markdown(feedback)
    
    with tab4:
        st.title("Contact Us")
        st.caption("Get in touch with the JobsForHer team for any questions or assistance")
        if st.session_state.contact_form_submitted:
            st.success("Thank you for your message! We'll get back to you soon.")
            if st.button("Send another message"):
                st.session_state.contact_form_submitted = False
                st.rerun()
        else:
            with st.form("contact_form"):
                contact_name = st.text_input("Your Name", value=st.session_state.user_profile.get("name", ""))
                contact_email = st.text_input("Your Email", value=st.session_state.user_profile.get("email", ""))
                contact_subject = st.selectbox(
                    "Subject",
                    options=[
                        "General Inquiry",
                        "Technical Support",
                        "Job Application Help",
                        "Resume Review Request",
                        "Feedback on Asha",
                        "Other"
                    ]
                )
                contact_message = st.text_area("Your Message", height=150)
                if st.form_submit_button("Send Message"):
                    if contact_name and contact_email and contact_message:
                        success = handle_contact_form(contact_name, contact_email, contact_message, contact_subject)
                        if success:
                            st.success("Thank you for your message! We'll get back to you soon.")
                            st.session_state.contact_form_submitted = True
                            st.rerun()
                    else:
                        st.error("Please fill in all required fields")
            st.header("Other Ways to Reach Us")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("JobsForHer Foundation")
                st.write("123 Career Avenue")
                st.write("Bangalore, Karnataka 560001")
                st.write("India")
            with col2:
                st.subheader("Contact Information")
                st.write("Email: psnehadeepika2006@gmail.com")
                st.write("Phone: +91-80-1234-5678")
                st.write("Hours: Mon-Fri, 9 AM - 6 PM IST")

def process_user_query(user_query, api_client, bias_rules, lang_code="en"):
    if user_query:
        message_id = f"user_{int(time.time())}"
        st.session_state.messages.append({"role": "user", "content": user_query, "id": message_id})
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_query(user_query, api_client, bias_rules)
                response_id = f"assistant_{int(time.time())}"
                st.session_state.messages.append({"role": "assistant", "content": response, "id": response_id})
                st.write(response)
                display_feedback_component(response_id)
                if st.session_state.get("auto_play_voice", False):
                    with st.spinner("Generating voice..."):
                        audio_data = text_to_speech(response, lang_code)
                        if audio_data:
                            autoplay_audio(audio_data)

if __name__ == "__main__":
    main()
