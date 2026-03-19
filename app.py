from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import os, uuid, numpy as np, subprocess, json
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB limit

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 300MB."}), 413

UPLOAD_FOLDER = "/tmp/reverb_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED = {".mp4", ".m4a", ".mov", ".aac", ".flac", ".ogg", ".wav", ".mp3", ".webm"}

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400
    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, uid + ext)
    f.save(input_path)
    return jsonify({"job_id": uid, "ext": ext})

@app.route("/process/<job_id>")
def process(job_id):
    ext = request.args.get("ext", ".mp4")
    input_path = os.path.join(UPLOAD_FOLDER, job_id + ext)
    temp_wav   = os.path.join(UPLOAD_FOLDER, job_id + "_temp.wav")
    output_mp3 = os.path.join(UPLOAD_FOLDER, job_id + "_reverb.mp3")

    def generate():
        try:
            if not os.path.exists(input_path):
                yield f"data: {json.dumps({'error': 'File not found'})}\n\n"
                return

            yield f"data: {json.dumps({'step': 'Extracting audio from video...'})}\n\n"
            result = subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                temp_wav
            ], capture_output=True, text=True)

            if result.returncode != 0:
                yield f"data: {json.dumps({'error': 'Audio extraction failed'})}\n\n"
                return

            yield f"data: {json.dumps({'step': 'Applying reverb...'})}\n\n"
            board = Pedalboard([Reverb(room_size=0.65, damping=0.6, wet_level=0.45, dry_level=0.85, freeze_mode=0.0, width=0.9)])
            with AudioFile(temp_wav) as af:
                sample_rate = af.samplerate
                audio_data  = af.read(af.frames)
            processed = board(audio_data, sample_rate)

            yield f"data: {json.dumps({'step': 'Normalizing audio...'})}\n\n"
            peak = np.max(np.abs(processed))
            if peak > 0:
                processed = processed / peak * 0.92

            yield f"data: {json.dumps({'step': 'Exporting MP3...'})}\n\n"
            raw_path = os.path.join(UPLOAD_FOLDER, job_id + "_raw.pcm")
            processed_int16 = (processed * 32767).astype(np.int16)
            processed_int16.T.flatten().astype(np.int16).tofile(raw_path)
            subprocess.run([
                "ffmpeg", "-y", "-f", "s16le", "-ar", str(sample_rate), "-ac", "2",
                "-i", raw_path, "-b:a", "320k", output_mp3
            ], capture_output=True)
            if os.path.exists(raw_path):
                os.remove(raw_path)

            yield f"data: {json.dumps({'step': 'done', 'job_id': job_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            for p in [input_path, temp_wav]:
                if os.path.exists(p): os.remove(p)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/download/<job_id>")
def download(job_id):
    output_mp3 = os.path.join(UPLOAD_FOLDER, job_id + "_reverb.mp3")
    if not os.path.exists(output_mp3):
        return jsonify({"error": "File not found"}), 404
    return send_file(output_mp3, mimetype="audio/mpeg", as_attachment=True, download_name="reverb_output.mp3")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
