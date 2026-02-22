import os
import sys
import asyncio
from flask import Flask, render_template, request, jsonify, send_file
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import edge_tts

# Windows asyncio fix
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = Flask(__name__)

RECORD_FOLDER = "record_videos"
SPEAK_FOLDER = "speaking_video"

os.makedirs(RECORD_FOLDER, exist_ok=True)
os.makedirs(SPEAK_FOLDER, exist_ok=True)

# -------------------------
# Load Whisper model
# -------------------------
print("Loading Whisper model...")
model = WhisperModel("base", device="auto", compute_type="auto")
print("Model Loaded Successfully")

# -------------------------
# Language name mapping (code -> display name)
# -------------------------
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali",
    "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "or": "Odia",
    "as": "Assamese", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "tr": "Turkish",
    "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "fi": "Finnish",
    "id": "Indonesian", "vi": "Vietnamese", "th": "Thai",
}

# -------------------------
# Voice Mapping (lang code -> Edge TTS voice)
# -------------------------
VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "ta": "ta-IN-PallaviNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "mr": "mr-IN-AarohiNeural",
    "bn": "bn-IN-TanishaaNeural",
    "gu": "gu-IN-DhwaniNeural",
    "pa": "pa-IN-OjasNeural",
    "ur": "ur-PK-UzmaNeural",
    "or": "or-IN-SubhasiniNeural",
    "as": "as-IN-YashicaNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ko": "ko-KR-SunHiNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-EG-SalmaNeural",
    "tr": "tr-TR-EmelNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-AgnieszkaNeural",
    "sv": "sv-SE-SofieNeural",
    "fi": "fi-FI-NooraNeural",
    "id": "id-ID-GadisNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "th": "th-TH-PremwadeeNeural",
}

# -------------------------
# Speech to Text (Whisper auto-detects language)
# -------------------------
def speech_to_text(audio_path):
    segments, info = model.transcribe(audio_path)
    full_text = " ".join(segment.text for segment in segments)
    detected_lang = info.language  # e.g. "en", "hi", "te" etc.
    confidence = round(info.language_probability * 100, 1)
    return full_text.strip(), detected_lang, confidence

# -------------------------
# Text to Speech
# -------------------------
async def text_to_audio(text, voice, output_path):
    tts = edge_tts.Communicate(text, voice=voice)
    await tts.save(output_path)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_audio", methods=["POST", "OPTIONS"])
def process_audio():
    if request.method == "OPTIONS":
        return "", 204

    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided."}), 400

        audio_file = request.files["audio"]
        target_lang = request.form.get("target_lang", "en")

        input_path = os.path.join(RECORD_FOLDER, "record.wav")
        output_path = os.path.join(SPEAK_FOLDER, "output.mp3")

        audio_file.save(input_path)

        # 1. Auto-detect language + Transcribe
        text, detected_lang, confidence = speech_to_text(input_path)
        if not text:
            return jsonify({"error": "No speech detected. Please speak clearly and try again."}), 400

        detected_lang_name = LANGUAGE_NAMES.get(detected_lang, detected_lang.upper())
        target_lang_name   = LANGUAGE_NAMES.get(target_lang, target_lang.upper())

        # 2. Skip translation if source == target
        if detected_lang == target_lang:
            translated_text = text
        else:
            try:
                translated_text = GoogleTranslator(source=detected_lang, target=target_lang).translate(text)
            except Exception:
                # Fallback: use 'auto' if detected lang code not supported by GoogleTranslator
                translated_text = GoogleTranslator(source="auto", target=target_lang).translate(text)

        # 3. Pick TTS voice (fallback to English if not mapped)
        selected_voice = VOICE_MAP.get(target_lang, "en-US-AriaNeural")

        # 4. Generate Audio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(text_to_audio(translated_text, selected_voice, output_path))
        loop.close()

        # 5. Delete input file
        if os.path.exists(input_path):
            os.remove(input_path)

        return jsonify({
            "detected_lang":      detected_lang,
            "detected_lang_name": detected_lang_name,
            "confidence":         confidence,
            "target_lang_name":   target_lang_name,
            "original_text":      text,
            "translated_text":    translated_text,
            "audio_url":          "/get_audio"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_audio", methods=["GET", "OPTIONS"])
def get_audio():
    if request.method == "OPTIONS":
        return "", 204

    output_path = os.path.join(SPEAK_FOLDER, "output.mp3")
    if not os.path.exists(output_path):
        return jsonify({"error": "Audio not found."}), 404

    response = send_file(output_path, mimetype="audio/mpeg")

    @response.call_on_close
    def cleanup():
        if os.path.exists(output_path):
            os.remove(output_path)

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))