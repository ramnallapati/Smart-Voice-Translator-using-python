<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&pause=1000&color=6C63FF&center=true&vCenter=true&width=600&lines=🎤+Smart+Voice+Translator;Speak+Any+Language.+Be+Understood+in+All." alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ramnallapati/Smart-Voice-Translator-using-python?style=for-the-badge&color=yellow)](https://github.com/ramnallapati/Smart-Voice-Translator-using-python/stargazers)

<br/>

> **Record your voice → Auto-detect language → Translate → Hear it back.**  
> Powered by AI. Works in 30+ languages. No setup needed.

</div>

---

## ✨ What It Does

<table>
<tr>
<td width="50%">

**In just 3 seconds:**

🎙️ You speak in **any language**  
🧠 AI **auto-detects** what you said  
🌍 Translates it to your **chosen language**  
🔊 Speaks the translation **back to you**  
🗑️ Cleans up temp files **automatically**

</td>
<td width="50%">

```
You say:     "नमस्ते, आप कैसे हैं?"
Detected:    Hindi (98.2% confidence)
Translated:  "Hello, how are you?"
Spoken:      🔊 English voice plays
```

</td>
</tr>
</table>

---

## 🧠 How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  🎤 Browser  │────▶│  Flask Backend   │────▶│  Faster-Whisper   │
│  Records     │     │  /process_audio  │     │  (Transcribe +    │
│  Voice       │     │                  │     │   Detect Lang)    │
└─────────────┘     └──────────────────┘     └────────┬──────────┘
                                                       │
       ┌───────────────────────────────────────────────▼──────────┐
       │                                                            │
┌──────▼──────┐     ┌──────────────────┐     ┌────────────────────┐
│  🔊 Browser  │◀────│  Edge-TTS        │◀────│  Deep Translator   │
│  Plays Audio │     │  (Text-to-       │     │  (Google API)      │
│              │     │   Speech)        │     │  Translates Text   │
└─────────────┘     └──────────────────┘     └────────────────────┘
```

---

## 🚀 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 🎙️ **STT** | [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) | Speech-to-text + language detection |
| 🌍 **Translation** | [Deep Translator](https://github.com/nidhaloff/deep-translator) | Google Translate API wrapper |
| 🔊 **TTS** | [Edge-TTS](https://github.com/rany2/edge-tts) | Neural text-to-speech (30+ voices) |
| ⚙️ **Backend** | Flask + Gunicorn | REST API server |
| 🎨 **Frontend** | HTML + CSS + JavaScript | Browser recording & UI |
| ☁️ **Hosting** | Render | Cloud deployment |

---

## 🌐 Supported Languages

<details>
<summary><b>Click to expand — 30+ languages supported</b></summary>

<br/>

| Flag | Language | Code | Flag | Language | Code |
|------|----------|------|------|----------|------|
| 🇺🇸 | English | `en` | 🇯🇵 | Japanese | `ja` |
| 🇮🇳 | Hindi | `hi` | 🇨🇳 | Chinese | `zh` |
| 🇮🇳 | Telugu | `te` | 🇰🇷 | Korean | `ko` |
| 🇮🇳 | Tamil | `ta` | 🇫🇷 | French | `fr` |
| 🇮🇳 | Kannada | `kn` | 🇩🇪 | German | `de` |
| 🇮🇳 | Malayalam | `ml` | 🇪🇸 | Spanish | `es` |
| 🇮🇳 | Marathi | `mr` | 🇮🇹 | Italian | `it` |
| 🇮🇳 | Bengali | `bn` | 🇧🇷 | Portuguese | `pt` |
| 🇮🇳 | Gujarati | `gu` | 🇷🇺 | Russian | `ru` |
| 🇮🇳 | Punjabi | `pa` | 🇸🇦 | Arabic | `ar` |
| 🇵🇰 | Urdu | `ur` | 🇹🇷 | Turkish | `tr` |
| 🇮🇳 | Odia | `or` | 🇳🇱 | Dutch | `nl` |
| 🇮🇳 | Assamese | `as` | 🇵🇱 | Polish | `pl` |
| 🇸🇪 | Swedish | `sv` | 🇻🇳 | Vietnamese | `vi` |
| 🇫🇮 | Finnish | `fi` | 🇹🇭 | Thai | `th` |
| 🇮🇩 | Indonesian | `id` | | | |

</details>

---

## 📂 Project Structure

```
Smart-Voice-Translator-using-python/
│
├── 📄 app.py                  # Flask backend — routes, STT, TTS, translation
├── 📄 Procfile                # Gunicorn start command for Render
├── 📄 requirements.txt        # Python dependencies
│
├── 📁 templates/
│   └── index.html             # Frontend UI — recording & playback
├── 📁 record_videos/          # Temp folder for uploaded audio (auto-deleted)
└── 📁 speaking_video/         # Temp folder for generated audio (auto-deleted)
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10+
- A working microphone
- Internet connection (for translation API)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ramnallapati/Smart-Voice-Translator-using-python.git
cd Smart-Voice-Translator-using-python
```

### 2️⃣ Create & Activate Virtual Environment

```bash
# Create
python -m venv venv

# Activate (Linux / macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
python app.py
```

Open your browser at → **http://127.0.0.1:5000**



---

## 📡 API Reference

### `POST /process_audio`

Accepts a voice recording and returns transcription, translation, and audio URL.

**Request (multipart/form-data)**

| Field | Type | Description |
|-------|------|-------------|
| `audio` | file | `.wav` audio recording |
| `target_lang` | string | Target language code (e.g. `"hi"`) |

**Response (JSON)**

```json
{
  "detected_lang": "en",
  "detected_lang_name": "English",
  "confidence": 97.3,
  "original_text": "Hello, how are you?",
  "target_lang_name": "Hindi",
  "translated_text": "नमस्ते, आप कैसे हैं?",
  "audio_url": "/get_audio"
}
```

### `GET /get_audio`

Returns the generated `.mp3` audio file. File is deleted from server after delivery.

---

## 📈 Roadmap

- [x] Auto language detection
- [x] 30+ language support
- [x] Neural voice output
- [x] Cloud deployment
- [ ] Real-time streaming translation
- [ ] Mobile-responsive PWA
- [ ] Speaker diarization (multi-speaker)
- [ ] Faster inference with quantized models
- [ ] Offline mode support

---

## 🤝 Contributing

Contributions are welcome and appreciated!

```bash
# 1. Fork this repo
# 2. Create your feature branch
git checkout -b feature/AmazingFeature

# 3. Commit your changes
git commit -m "Add AmazingFeature"

# 4. Push to branch
git push origin feature/AmazingFeature

# 5. Open a Pull Request 🎉
```

---

## 📜 License

Distributed under the **Apache 2.0 License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by [Ram Nallapati](https://github.com/ramnallapati)**

*AI & Python Developer*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-ramnallapati-181717?style=for-the-badge&logo=github)](https://github.com/ramnallapati)

<br/>

⭐ **If this project helped you, please give it a star!** ⭐

</div>