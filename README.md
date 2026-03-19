# ✈ CareerPilot — Personal Career Assistant

> An AI-powered career companion built with Streamlit and NVIDIA's Nemotron-70B model. Get personalised career advice, generate resumes, prep for interviews, write cover letters, and more — all in one place.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **AI Chat** | Conversational career coach powered by NVIDIA Nemotron-70B |
| 📄 **Resume Builder** | Generate a polished, ATS-friendly resume from your profile |
| 🎯 **Interview Prep** | AI-generated interview questions + live answer feedback |
| ✉️ **Cover Letter Generator** | Paste any job description, get a tailored cover letter instantly |
| 👤 **Profile Manager** | Save your skills, education, and work history for personalised output |
| 🔊 **Voice Output** | Text-to-speech responses in 10 Indian languages |
| 💡 **Daily Career Tips** | A fresh tip every day to keep you moving forward |
| ⚡ **Response Caching** | Faster replies for repeated questions |

---

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **AI Model:** `nvidia/llama-3.1-nemotron-70b-instruct` via NVIDIA NIM API
- **TTS:** gTTS (Google Text-to-Speech)
- **Languages supported:** English, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/careerpilot.git
cd careerpilot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your NVIDIA API key

Get a free API key from [build.nvidia.com](https://build.nvidia.com), then:

```bash
cp .env.example .env
```

Open `.env` and add your key:

```
NVIDIA_API_KEY=your_actual_key_here
```

> ⚠️ Never commit your `.env` file. It's already in `.gitignore`.

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
careerpilot/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # Your secret API key (never commit this)
├── .env.example         # Template for environment variables
├── .gitignore           # Ensures .env is never pushed to GitHub
└── README.md            # You're reading it!
```

---

## 🔒 Keeping Your API Key Safe

This project uses a `.env` file to store your API key locally. Here's how it's protected:

- `.env` is listed in `.gitignore` — Git will **never** track or upload it
- The app reads the key silently using `python-dotenv` — it's never displayed in the UI
- Share `.env.example` (with a blank placeholder) so collaborators know what's needed without seeing your real key

---

## 📦 Requirements

```
streamlit>=1.32.0
requests>=2.31.0
pandas>=2.0.0
python-dotenv>=1.0.0
gTTS>=2.4.0
SpeechRecognition>=3.10.0
pyaudio>=0.2.14
pydub>=0.25.1
```

> **Note:** `pyaudio` may require additional system dependencies.
> - **Mac:** `brew install portaudio`
> - **Linux:** `sudo apt-get install portaudio19-dev`
> - **Windows:** Install via `pipwin install pyaudio`

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

<p align="center">Built with ❤️ using Streamlit & NVIDIA NIM</p>
