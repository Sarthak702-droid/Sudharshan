import os
import sys
import subprocess
import json
import flask
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
LABELS_DIR = os.path.join(CONFIGS_DIR, "labels")

# Ensure directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# Helper to find sequences
def scan_local_sequences():
    sequences = []

    # 1. Scan datasets folder inside project
    if os.path.exists(DATASETS_DIR):
        for name in os.listdir(DATASETS_DIR):
            path = os.path.join(DATASETS_DIR, name)
            if os.path.isdir(path):
                sequences.append({
                    "name": name,
                    "type": "directory",
                    "path": path
                })
            elif name.lower().endswith((".mp4", ".avi", ".mov")):
                sequences.append({
                    "name": name,
                    "type": "video",
                    "path": path
                })

    # 2. Optionally scan an external VisDrone directory. Dataset paths stay
    # outside source control and are supplied by the deployment environment.
    visdrone_seq_dir = os.environ.get("VISDRONE_SEQUENCE_DIR", "")
    if os.path.exists(visdrone_seq_dir):
        for name in os.listdir(visdrone_seq_dir):
            path = os.path.join(visdrone_seq_dir, name)
            if os.path.isdir(path):
                sequences.append({
                    "name": f"VisDrone_{name}",
                    "type": "directory",
                    "path": path
                })

    return sequences

@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "src", "dashboard"), "index.html")

@app.route("/dashboard/<path:filename>")
def serve_dashboard_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, "src", "dashboard"), filename)

@app.route("/api/sequences", methods=["GET"])
def get_sequences():
    seqs = scan_local_sequences()
    return jsonify(seqs)

@app.route("/api/scan_path", methods=["POST"])
def scan_custom_path():
    """
    Scans a custom path provided by the user (video or image sequence directory).
    """
    data = request.json or {}
    custom_path = data.get("path", "")

    if not custom_path or not os.path.exists(custom_path):
        return jsonify({"error": "Path does not exist"}), 400

    name = os.path.basename(custom_path)
    if os.path.isdir(custom_path):
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        frames = sorted([f for f in os.listdir(custom_path) if f.lower().endswith(valid_extensions)])
        return jsonify({
            "name": name,
            "type": "directory",
            "path": custom_path,
            "frameCount": len(frames)
        })
    elif custom_path.lower().endswith((".mp4", ".avi", ".mov")):
        return jsonify({
            "name": name,
            "type": "video",
            "path": custom_path,
            "frameCount": -1 # Unknown without open
        })
    else:
        return jsonify({"error": "Invalid file type. Must be a folder of images or an MP4 video file"}), 400

@app.route("/api/sequence/frames", methods=["POST"])
def get_sequence_frames():
    """
    Returns image file names inside an image sequence directory.
    """
    data = request.json or {}
    seq_path = data.get("path", "")

    if not seq_path or not os.path.exists(seq_path) or not os.path.isdir(seq_path):
        return jsonify({"error": "Invalid directory path"}), 400

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    frames = sorted([f for f in os.listdir(seq_path) if f.lower().endswith(valid_extensions)])

    return jsonify({
        "frames": frames,
        "path": seq_path
    })

@app.route("/api/sequence/frame_file", methods=["GET"])
def get_frame_file():
    """
    Serves a specific frame image from a local path.
    """
    directory = request.args.get("dir")
    filename = request.args.get("file")
    if not directory or not filename:
        return "Missing arguments", 400
    return send_from_directory(directory, filename)

@app.route("/api/labels", methods=["POST"])
def get_labels():
    """
    Loads custom labels for a given sequence.
    """
    data = request.json or {}
    name = data.get("name", "")

    # Sanitize label filename
    safe_name = "".join([c for c in name if c.isalnum() or c in ("-", "_")]).strip()
    label_file = os.path.join(LABELS_DIR, f"{safe_name}_labels.json")

    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            return jsonify(json.load(f))
    else:
        return jsonify({"frames": {}})

@app.route("/api/labels/save", methods=["POST"])
def save_labels():
    """
    Saves custom labels for a sequence.
    """
    data = request.json or {}
    name = data.get("name", "")
    labels = data.get("labels", {})

    safe_name = "".join([c for c in name if c.isalnum() or c in ("-", "_")]).strip()
    label_file = os.path.join(LABELS_DIR, f"{safe_name}_labels.json")

    # Save the labels file
    with open(label_file, "w") as f:
        json.dump(labels, f, indent=2)

    return jsonify({"success": True, "path": label_file})

@app.route("/api/process", methods=["POST"])
def process_sequence():
    """
    Triggers execution of scripts/run_pipeline.py as a subprocess.
    """
    data = request.json or {}
    input_path = data.get("input_path", "")
    label_name = data.get("label_name", "")
    expected_dir = data.get("expected_direction", "EAST")

    if not input_path or not os.path.exists(input_path):
        return jsonify({"error": f"Input path does not exist: {input_path}"}), 400

    # Build label file path if provided
    label_file_arg = []
    if label_name:
        safe_name = "".join([c for c in label_name if c.isalnum() or c in ("-", "_")]).strip()
        label_file = os.path.join(LABELS_DIR, f"{safe_name}_labels.json")
        if os.path.exists(label_file):
            label_file_arg = ["--labels", label_file]

    # Outputs paths
    video_output = os.path.join(OUTPUTS_DIR, "flow_overlay.mp4")
    json_output = os.path.join(OUTPUTS_DIR, "grid_metrics.json")
    csv_output = os.path.join(OUTPUTS_DIR, "grid_metrics.csv")

    # Construct subprocess command
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "scripts", "run_pipeline.py"),
        "--input", input_path,
        "--output_video", video_output,
        "--output_json", json_output,
        "--output_csv", csv_output,
        "--expected_direction", expected_dir
    ] + label_file_arg

    print(f"Executing: {' '.join(cmd)}")

    try:
        # Run pipeline
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Pipeline processed successfully!")

        # Load output json metrics to return to the UI
        if os.path.exists(json_output):
            with open(json_output, "r") as f:
                metrics_data = json.load(f)
        else:
            metrics_data = []

        return jsonify({
            "success": True,
            "stdout": res.stdout,
            "metrics": metrics_data,
            "video_path": "/api/outputs/flow_overlay.mp4"
        })
    except subprocess.CalledProcessError as e:
        print(f"Error executing pipeline:\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return jsonify({
            "error": "Pipeline execution failed",
            "details": e.stderr
        }), 500

@app.route("/api/outputs/flow_overlay.mp4")
def serve_output_video():
    """
    Serves the generated video file.
    """
    return send_from_directory(OUTPUTS_DIR, "flow_overlay.mp4")

@app.route("/api/outputs/grid_metrics.json")
def serve_output_json():
    """
    Serves the generated JSON metrics.
    """
    return send_from_directory(OUTPUTS_DIR, "grid_metrics.json")

@app.route("/api/outputs/frames/<filename>")
def serve_output_frame(filename):
    """
    Serves sequential rendered frames for the browser sequence player.
    """
    return send_from_directory(os.path.join(OUTPUTS_DIR, "rendered_frames"), filename)

@app.route("/api/outputs/status", methods=["GET"])
def check_outputs_status():
    """
    Returns status and frame count of existing processed outputs for startup auto-load.
    """
    json_output = os.path.join(OUTPUTS_DIR, "grid_metrics.json")
    frames_dir = os.path.join(OUTPUTS_DIR, "rendered_frames")

    exists = os.path.exists(json_output)
    frames_count = 0
    if exists and os.path.exists(frames_dir):
        frames = [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        frames_count = len(frames)

    # Read Go backend settings from config YAML
    backend_enabled = False
    backend_url = "http://localhost:8080"
    config_path = os.path.join(CONFIGS_DIR, "grid_config.yaml")
    if os.path.exists(config_path):
        import yaml
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
                b_cfg = cfg.get("backend", {})
                backend_enabled = b_cfg.get("enabled", False)
                backend_url = b_cfg.get("base_url", "http://localhost:8080")
        except Exception as e:
            print(f"Warning: Could not parse grid_config.yaml: {e}")

    return jsonify({
        "exists": exists,
        "frameCount": frames_count,
        "backendEnabled": backend_enabled,
        "backendUrl": backend_url
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Sudharshan offline dashboard server on http://localhost:{port}...")
    app.run(host="127.0.0.1", port=port, debug=True)
