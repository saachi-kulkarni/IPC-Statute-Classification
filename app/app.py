"""Statute Finder: local research demo (FastAPI backend + static law-themed frontend).

Run:  .venv/bin/python app/app.py            -> http://127.0.0.1:7860
      .venv/bin/python app/app.py --port 8000
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from common import set_seed  # noqa: E402
from predict import Predictor  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_CHARS = 200_000

set_seed()
predictor = Predictor()
app = FastAPI(title="Statute Finder (research demo)", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: str | None = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    return predictor.meta()


@app.post("/api/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if len(text.split()) < 5:
        raise HTTPException(400, "Please enter at least a few sentences of case facts.")
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"Text is too long (limit {MAX_CHARS:,} characters).")
    t0 = time.time()
    out = predictor.predict(text, req.model)
    out["latency_ms"] = round(1000 * (time.time() - t0))
    return out


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    print(f"Statute Finder on http://{args.host}:{args.port}  (models: {', '.join(predictor.models)})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
