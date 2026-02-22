import os
import sys
import asyncio
from flask import Flask, render_template, request, jsonify, send_file
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import edge_tts
from flask_cors import CORS, cross_origin

# Windows asyncio fix
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = Flask(__name__)

# -------------------------
# CORS setup
# -------------------------
CORS(app, origins=["https://relaxed-bunny-932279.netlify.app"], supports_credentials=True)

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
# Speech to Text
# -------------------------
def speech_to_text(audio_path):
    segments, info = model.transcribe(audio_path)
    full_text = " ".join(segment.text for segment in segments)
    return full_text.strip(), info.language

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

@app.route("/process_audio", methods=["POST"])
@cross_origin(origin="https://relaxed-bunny-932279.netlify.app")
def process_audio():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided."}), 400

        audio_file = request.files["audio"]
        target_lang = request.form.get("target_lang", "en")

        input_path = os.path.join(RECORD_FOLDER, "record.wav")
        output_path = os.path.join(SPEAK_FOLDER, "output.mp3")

        audio_file.save(input_path)

        # 1️⃣ Detect + Transcribe
        text, detected_lang = speech_to_text(input_path)
        if not text:
            return jsonify({"error": "No speech detected."}), 400

        # 2️⃣ Translate
        translated_text = GoogleTranslator(source=detected_lang, target=target_lang).translate(text)

        # 3️⃣ Voice Mapping
        voice_map = {
            "en": "en-US-AriaNeural",
            "te": "te-IN-ShrutiNeural",
            "hi": "hi-IN-SwaraNeural",
            "ur": "ur-PK-UzmaNeural"
        }
        selected_voice = voice_map.get(target_lang, "en-US-AriaNeural")

        # 4️⃣ Generate Audio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(text_to_audio(translated_text, selected_voice, output_path))
        loop.close()

        # 5️⃣ Delete input file immediately
        if os.path.exists(input_path):
            os.remove(input_path)

        return jsonify({
            "original_text": text,
            "translated_text": translated_text,
            "audio_url": "/get_audio"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_audio")
@cross_origin(origin="https://relaxed-bunny-932279.netlify.app")
def get_audio():
    output_path = os.path.join(SPEAK_FOLDER, "output.mp3")
    if not os.path.exists(output_path):
        return jsonify({"error": "Audio not found."}), 404

    response = send_file(output_path, mimetype="audio/mpeg")

    # Delete file after sending
    @response.call_on_close
    def cleanup():
        if os.path.exists(output_path):
            os.remove(output_path)

    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))