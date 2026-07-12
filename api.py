"""
api.py — Hausa Crisis Signal Detector commercial API (FastAPI, for RapidAPI listing)

Kept separate from app.py (the Gradio demo for Hugging Face Spaces) —
different deployment targets, different audiences: app.py is a free public
demo, this is the metered paid service.

Loads the fine-tuned model from a LOCAL checkpoint directory (produced by
train.py) rather than a public Hugging Face repo — see the module-level note
in README.md for why: publishing the weights publicly would let anyone
self-host this model for free, undermining the paid-API business model.

Requires RAPIDAPI_PROXY_SECRET to be set and validates it with a
timing-safe comparison, not a plain string equality check.
"""

import hmac
import logging
import os
from pathlib import Path

import torch
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("hausa_crisis_api")

PROXY_SECRET = os.environ.get("RAPIDAPI_PROXY_SECRET")
# Local checkpoint path, produced by train.py's trainer.save_model(OUTPUT_DIR)
# — expected to be baked into the Docker image or mounted as a volume, NOT
# pulled from a public Hub repo. See module docstring.
MODEL_PATH = os.environ.get("MODEL_PATH", "./hausa_crisis_model")

app = FastAPI(
    title="Hausa Crisis Signal Detector",
    description=(
        "Classifies Hausa-language text into OCHA humanitarian crisis "
        "clusters (conflict, displacement, disease_outbreak, flood, "
        "food_insecurity, no_crisis) for MEAL systems and humanitarian "
        "data pipelines."
    ),
    version="1.0.0",
    contact={
        "name": "Sadiya Muhammad Kilgori",
        "url": "https://github.com/SKilgori/hausa-crisis-signal-detector",
    },
)

_model = None
_tokenizer = None


@app.on_event("startup")
def load_model():
    global _model, _tokenizer
    if not Path(MODEL_PATH).exists():
        # Fail loudly at startup, not on the first request — a health check
        # or deploy pipeline should catch this immediately rather than
        # surfacing as a confusing 500 to a paying customer's first call.
        raise RuntimeError(
            f"MODEL_PATH '{MODEL_PATH}' does not exist. Run train.py and "
            f"confirm the checkpoint was saved (or that MODEL_PATH points "
            f"at the right mounted/baked-in location) before starting this API."
        )
    logger.info("Loading model from %s", MODEL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.eval()
    logger.info("Model loaded. Labels: %s", _model.config.id2label)


class CrisisRequest(BaseModel):
    text: str = Field(..., description="The Hausa text to classify.", min_length=1, max_length=512)


class CrisisResponse(BaseModel):
    label: str = Field(..., description="The detected OCHA crisis cluster.")
    confidence: float = Field(..., description="Model confidence score (0.0 to 1.0).")


def verify_rapidapi_proxy(x_rapidapi_proxy_secret: str = Header(...)) -> str:
    """Validates the request was routed through RapidAPI's gateway, not sent
    directly to this server's origin URL (which would bypass billing)."""
    if not PROXY_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfiguration: PROXY_SECRET not set.")
    # Timing-safe comparison — plain `!=` leaks timing information about
    # how many leading characters matched, in theory usable to brute-force
    # the secret faster than guessing it outright. Low real-world risk here,
    # but hmac.compare_digest costs nothing and is the standard practice.
    if not hmac.compare_digest(x_rapidapi_proxy_secret, PROXY_SECRET):
        raise HTTPException(status_code=403, detail="Unauthorized: Invalid Proxy Secret")
    return x_rapidapi_proxy_secret


@app.get("/health")
def health():
    # Cheap, no-auth endpoint for your hosting provider / uptime monitor —
    # deliberately NOT behind verify_rapidapi_proxy, since monitoring
    # infrastructure won't have the RapidAPI secret.
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/classify", response_model=CrisisResponse, dependencies=[Depends(verify_rapidapi_proxy)])
async def classify_text(request: CrisisRequest):
    """Classifies Hausa text into an OCHA humanitarian crisis cluster."""
    inputs = _tokenizer(
        request.text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = _model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence_score, predicted_class_id = torch.max(probabilities, dim=-1)
    predicted_label = _model.config.id2label[predicted_class_id.item()]

    return CrisisResponse(label=predicted_label, confidence=round(confidence_score.item(), 4))
