"""
BiasProbe — Application Configuration
Loads environment variables and initializes Firebase Admin SDK + Gemini client.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from google import genai

load_dotenv()

logger = logging.getLogger("biasProbe")

# ── Environment Variables ──────────────────────────────────────────────────────

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
FIREBASE_SERVICE_ACCOUNT_JSON: str = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "")
GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")

# ── Dev Mode Detection ─────────────────────────────────────────────────────────

LOCAL_DEV: bool = os.getenv("LOCAL_DEV", "true").lower() in ("true", "1", "yes")

# Cross-platform local reports directory (project-local, not /tmp/)
REPORTS_DIR = Path(__file__).resolve().parent / "reports_out"
REPORTS_DIR.mkdir(exist_ok=True)


# ── Firebase Admin SDK ─────────────────────────────────────────────────────────

_firebase_app = None


def get_firebase_app():
    """Initialize Firebase Admin SDK (singleton)."""
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    if FIREBASE_SERVICE_ACCOUNT_JSON:
        # Could be a file path or a JSON string
        if os.path.isfile(FIREBASE_SERVICE_ACCOUNT_JSON):
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_JSON)
        else:
            try:
                cred = credentials.Certificate(json.loads(FIREBASE_SERVICE_ACCOUNT_JSON))
            except json.JSONDecodeError as e:
                logger.error("Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: %s", e)
                raise
    else:
        # Fall back to Application Default Credentials (on Cloud Run)
        cred = credentials.ApplicationDefault()

    _firebase_app = firebase_admin.initialize_app(cred, {
        "projectId": FIREBASE_PROJECT_ID,
        "storageBucket": GCS_BUCKET_NAME,
    })
    return _firebase_app


def get_firestore_client():
    """Get a Firestore client instance."""
    get_firebase_app()
    return firestore.client()


# ── Gemini AI ──────────────────────────────────────────────────────────────────

_genai_client = None


def _get_genai_client() -> genai.Client:
    """Return a singleton google.genai Client."""
    global _genai_client
    if _genai_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


class _GeminiModelWrapper:
    """Thin compatibility wrapper so callers can keep using model.generate_content(prompt)."""

    def __init__(self, client: genai.Client, model_name: str):
        self._client = client
        self._model_name = model_name

    def generate_content(self, prompt: str):
        return self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
        )


def get_gemini_model(model_name: str = "gemini-2.0-flash"):
    """Configure and return a Gemini generative model wrapper."""
    client = _get_genai_client()
    return _GeminiModelWrapper(client, model_name)


def get_gemini_embedding_model(model_name: str = "models/text-embedding-004"):
    """Return the model name string for embedding calls."""
    # Ensure the client is configured (validates API key)
    _get_genai_client()
    return model_name
