from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import os, uuid, numpy as np
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 200MB."}), 413

UPLOAD_FOLDER = "/tmp/reverb_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED = {".mp4", ".m4a", ".mov", ".aac", ".flac", ".ogg", ".wav", ".mp3"}

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/convert", methods=["POST"])
def convert():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, uid + ext)
    temp_wav   = os.path.join(UPLOAD_FOLDER, uid + "_temp.wav")
    output_mp3 = os.path.join(UPLOAD_FOLDER, uid + "_reverb.mp3")

    try:
        f.save(input_path)

        # Convert to WAV
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(2).set_frame_rate(44100)
        audio.export(temp_wav, format="wav")

        # Apply reverb
        board = Pedalboard([
            Reverb(
                room_size=0.65,
                damping=0.6,
                wet_level=0.45,
                dry_level=0.85,
                freeze_mode=0.0,
                width=0.9,
            )
        ])

        with AudioFile(temp_wav) as af:
            sample_rate = af.samplerate
            audio_data  = af.read(af.frames)

        processed = board(audio_data, sample_rate)

        # Normalize
        peak = np.max(np.abs(processed))
        if peak > 0:
            processed = processed / peak * 0.92

        # Export MP3
        processed_int16 = (processed * 32767).astype(np.int16)
        interleaved = processed_int16.T.flatten().tobytes()
        seg = AudioSegment(
            data=interleaved,
            sample_width=2,
            frame_rate=sample_rate,
            channels=processed.shape[0],
        )
        seg.export(output_mp3, format="mp3", bitrate="320k")

        return send_file(
            output_mp3,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name="reverb_output.mp3"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for p in [input_path, temp_wav]:
            if os.path.exists(p):
                os.remove(p)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
