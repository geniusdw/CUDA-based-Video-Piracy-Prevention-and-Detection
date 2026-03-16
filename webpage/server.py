from __future__ import annotations

import argparse
import cgi
import json
import re
import shutil
import subprocess
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


WEB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_ROOT.parent
RUNTIME_ROOT = WEB_ROOT / "runtime"
UPLOADS_DIR = RUNTIME_ROOT / "uploads"
GENERATED_DIR = RUNTIME_ROOT / "generated"
SECURITY_EXE = PROJECT_ROOT / "security_gpu.exe"
DETECTOR_EXE = PROJECT_ROOT / "detector.exe"
PROCESS_TIMEOUT_SECONDS = 60 * 60


def ensure_runtime_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def sanitize_filename(filename: str, fallback_stem: str = "video") -> tuple[str, str]:
    path = Path(filename or "")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", path.suffix or ".mp4")

    if not stem:
        stem = fallback_stem
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix == ".":
        suffix = ".mp4"

    return stem, suffix.lower()


def detector_summary(output: str) -> dict[str, object]:
    is_pirated = "PIRATED COPY DETECTED" in output
    is_clean = "CLEAN / NO WATERMARK DETECTED" in output

    verdict = "Unknown"
    if is_pirated:
        verdict = "Pirated"
    elif is_clean:
        verdict = "Not Pirated"

    analyzed_pairs_match = re.search(r"Analyzed (\d+) TPVM frame-pairs\.", output)
    average_score_match = re.search(r"Average Correlation Score:\s*([-+]?[0-9]*\.?[0-9]+)", output)
    detected_pairs_match = re.search(r"Detected pairs \(>[0-9.]+\):\s*(\d+)/(\d+)", output)
    min_pairs_match = re.search(r"Min detected pairs required for confirm:\s*(\d+)", output)

    detected_summary = "-"
    if detected_pairs_match:
        detected_summary = f"{detected_pairs_match.group(1)}/{detected_pairs_match.group(2)}"

    return {
        "verdict": verdict,
        "isPirated": is_pirated,
        "analyzedPairs": analyzed_pairs_match.group(1) if analyzed_pairs_match else None,
        "averageScore": average_score_match.group(1) if average_score_match else None,
        "detectedPairsSummary": detected_summary,
        "minDetectedPairs": min_pairs_match.group(1) if min_pairs_match else None,
    }


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/":
            self.path = "/index.html"
            return super().do_GET()

        return super().do_GET()

    def do_POST(self) -> None:
        route = urlparse(self.path).path

        if route == "/api/protect":
            self.handle_protect()
            return

        if route == "/api/detect":
            self.handle_detect()
            return

        self.send_json({"success": False, "error": "Endpoint not found."}, HTTPStatus.NOT_FOUND)

    def log_message(self, format_: str, *args) -> None:
        print(f"[webpage] {self.address_string()} - {format_ % args}")

    def parse_form(self) -> cgi.FieldStorage:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected a multipart/form-data upload.")

        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
            keep_blank_values=True,
        )

    def save_upload(self, form: cgi.FieldStorage) -> Path:
        if "video" not in form:
            raise ValueError("No video file was uploaded.")

        field = form["video"]
        if isinstance(field, list):
            field = field[0]

        if not getattr(field, "file", None):
            raise ValueError("The uploaded file is missing.")

        stem, suffix = sanitize_filename(getattr(field, "filename", "video.mp4"))
        destination = UPLOADS_DIR / f"{stem}_{timestamp_token()}{suffix}"

        with destination.open("wb") as upload_handle:
            shutil.copyfileobj(field.file, upload_handle)

        return destination

    def run_program(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
            check=False,
        )

    def handle_protect(self) -> None:
        try:
            if not SECURITY_EXE.exists():
                raise FileNotFoundError(f"Missing executable: {SECURITY_EXE.name}")

            form = self.parse_form()
            input_path = self.save_upload(form)
            input_stem = re.sub(r"_[0-9]{8}-[0-9]{6}-[0-9]{6}$", "", input_path.stem) or "protected_input"
            output_path = GENERATED_DIR / f"{input_stem}_protected_{timestamp_token()}.mp4"

            result = self.run_program([str(SECURITY_EXE), str(input_path), str(output_path)])
            combined_log = (result.stdout or "") + (result.stderr or "")

            if result.returncode != 0:
                raise RuntimeError(combined_log.strip() or "security_gpu.exe failed.")
            if not output_path.exists():
                raise RuntimeError("Processing finished, but no protected video file was created.")

            self.send_json(
                {
                    "success": True,
                    "outputVideo": f"/{output_path.relative_to(WEB_ROOT).as_posix()}",
                    "outputFileName": output_path.name,
                    "log": combined_log.strip() or "Processing completed successfully.",
                }
            )
        except subprocess.TimeoutExpired:
            self.send_json(
                {
                    "success": False,
                    "error": "security_gpu.exe timed out while processing the uploaded video.",
                },
                HTTPStatus.GATEWAY_TIMEOUT,
            )
        except Exception as error:  # noqa: BLE001
            self.send_json(
                {
                    "success": False,
                    "error": str(error),
                },
                HTTPStatus.BAD_REQUEST,
            )

    def handle_detect(self) -> None:
        try:
            if not DETECTOR_EXE.exists():
                raise FileNotFoundError(f"Missing executable: {DETECTOR_EXE.name}")

            form = self.parse_form()
            input_path = self.save_upload(form)
            result = self.run_program([str(DETECTOR_EXE), str(input_path)])
            combined_log = (result.stdout or "") + (result.stderr or "")

            if result.returncode != 0:
                raise RuntimeError(combined_log.strip() or "detector.exe failed.")

            summary = detector_summary(combined_log)
            if summary["verdict"] == "Unknown":
                raise RuntimeError("Detector finished, but the final verdict could not be parsed.")

            payload = {
                "success": True,
                "log": combined_log.strip() or "Detection completed successfully.",
            }
            payload.update(summary)
            self.send_json(payload)
        except subprocess.TimeoutExpired:
            self.send_json(
                {
                    "success": False,
                    "error": "detector.exe timed out while analyzing the uploaded video.",
                },
                HTTPStatus.GATEWAY_TIMEOUT,
            )
        except Exception as error:  # noqa: BLE001
            self.send_json(
                {
                    "success": False,
                    "error": str(error),
                },
                HTTPStatus.BAD_REQUEST,
            )

    def send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local CUDA Video Shield website.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Serving CUDA Video Shield at http://{args.host}:{args.port}")
    print(f"Project root: {PROJECT_ROOT}")
    print("Press Ctrl+C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
