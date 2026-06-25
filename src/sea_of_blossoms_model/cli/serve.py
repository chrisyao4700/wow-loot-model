from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sea_of_blossoms_model.lambda_handler import _load_model
from sea_of_blossoms_model.serialization import (
    prediction_input_from_api,
    prediction_output_to_api,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local loot model HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4317)
    parser.add_argument("--artifact", type=Path, help="Trained model artifact JSON.")
    args = parser.parse_args()

    if args.artifact:
        os.environ["MODEL_ARTIFACT_PATH"] = str(args.artifact)
        _load_model.cache_clear()

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"Model server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


class _Handler(BaseHTTPRequestHandler):
    server_version = "SeaOfBlossomsModel/0.1"

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return

        model = _load_model()
        self._send_json({
            "ok": True,
            "model": {
                "id": model.metadata.id,
                "version": model.metadata.version,
            },
        })

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json()
            model_input = prediction_input_from_api(payload)
            output = prediction_output_to_api(_load_model().predict(model_input))
            self._send_json(output)
        except Exception as error:
            self._send_json(
                {
                    "error": "prediction_failed",
                    "message": str(error),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("content-length")
        length = int(raw_length) if raw_length else 0
        raw_body = self.rfile.read(length)
        return json.loads(raw_body.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    main()
