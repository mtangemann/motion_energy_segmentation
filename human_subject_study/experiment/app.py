import json
import logging
from collections import defaultdict
from pathlib import Path

import coloredlogs
from flask import Flask, request, send_from_directory

app = Flask(__name__)

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
LOG_FILE = Path(__file__).parent / "logs" / "app.log"
LOG_LEVEL = logging.INFO

LOGGER = logging.getLogger(__name__)

coloredlogs.install(fmt=LOG_FORMAT, level=LOG_LEVEL)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler(LOG_FILE, mode="a")
handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter(LOG_FORMAT)
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)

DATA_PATH = Path(app.root_path) / "data"
STATIC_PATH = Path(app.root_path) / "static"
OUTPUT_PATH = Path(app.root_path) / "output"


@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def default(path: str):
    if path.startswith("data/"):
        return send_from_directory(DATA_PATH, path[5:])
    else:
        return send_from_directory(STATIC_PATH, path)


@app.route("/log", methods=["POST"])
def log():
    if not request.is_json:
        return {"error": "Request must be JSON"}, 400

    data = request.get_json()

    if "level" in data:
        level = getattr(logging, data["level"].upper(), None)
        del data["level"]
    else:
        LOGGER.warning("No log level provided. Defaulting to 'info'.")
        level = logging.INFO

    if "message" in data:
        message = data["message"]
        del data["message"]
    else:
        LOGGER.warning("No log message provided.")
        message = "<no message>"

    LOGGER.log(level, "Client: %s\n%s", message, json.dumps(data, indent=2))

    return {"status": "ok"}


@app.route("/progress/<subject_id>", methods=["GET"])
def progress(subject_id: str):
    subject_path = OUTPUT_PATH / subject_id

    if not subject_path.exists():
        return {"stage": "start", "session": -1}

    sessions = [int(path.name) for path in subject_path.iterdir() if path.is_dir()]

    if len(sessions) == 0:
        return {"stage": "start", "session": -1}
    
    session = sorted(sessions)[-1]

    if len(list(subject_path.glob("*/results.json"))) >= 1:
        return {"stage": "end", "session": session}

    recorded_trials = defaultdict(list)
    for file in subject_path.glob("*/*/*.json"):
        stage = file.parent.name
        trial = int(file.stem)
        recorded_trials[stage].append(trial)

    if len(recorded_trials["test"]) >= 1:
        trial_index = max(recorded_trials["test"])
        return {"stage": "test", "session": session, "trial_index": trial_index}

    return {"stage": "start", "session": session}


@app.route("/results", methods=["POST"])
def results():
    if not request.is_json:
        return {"error": "Request must be JSON"}, 400

    data = request.get_json()

    subject_id = data[0]["subject_id"]
    session = data[0]["session"]
    session_path = OUTPUT_PATH / subject_id / str(session)

    if len(data) == 1:
        data = data[0]

        if data["trial_type"] == "browser-check":
            response_file = session_path / "browser_check.json"

        elif data["trial_type"] == "random-dot-shape-identification-response":
            stage = data["stage"]
            trial_index = data["trial_index_in_stage"]
            response_file = session_path / stage / f"{trial_index}.json"

        else:
            LOGGER.warning("Unexpected trial type: %s", data["trial_type"])
            trial_index = data["trial_index"]
            response_file = session_path / f"unknown_{trial_index}.json"

    else:
        response_file = session_path / "results.json"

    response_file.parent.mkdir(parents=True, exist_ok=True)

    original_name = response_file.name
    index = 1
    while response_file.exists():
        next_name = original_name.replace(".json", f"_{index}.json")
        response_file = response_file.with_name(next_name)
        index += 1

    with open(response_file, "w") as f:
        json.dump(data, f)

    if len(data) > 1 and Path("prolific_return_url").exists():
        with open("prolific_return_url") as f:
            return_url = f.read().strip()
        return {"status": "ok", "prolific_return_url": return_url}
    else:
        return {"status": "ok"}
