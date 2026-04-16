"""
llm_connector.py
================
BiasProbe — multi-provider LLM connector.

Supported providers
-------------------
  openai      — OpenAI, Azure OpenAI, Together AI, Groq (all OpenAI-compatible)
  gemini      — Google Gemini via google-generativeai SDK
  anthropic   — Anthropic Claude via direct HTTP
  custom      — Any REST endpoint via user-defined request template

Main surface
------------
  LLMConnector.send_probes(config, battery, audit_id) -> list[ProbeResult]
    Sends all probes concurrently (≤ 10 in-flight), with 30 s timeout,
    one retry on failure, Firestore persistence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import string
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any

import httpx
from dotenv import load_dotenv
from google.cloud import firestore as _fs

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CONCURRENT = 10          # semaphore cap
TIMEOUT_SECONDS = 30.0
FIRESTORE_COLLECTION = "audits"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ConnectorConfig:
    """
    Union-style config: only fields relevant to the chosen provider need be set.

    Parameters
    ----------
    provider : str
        One of: "openai", "gemini", "anthropic", "custom"
    api_key : str | None
        API key (openai / gemini / anthropic).
    model : str | None
        Model name, e.g. "gpt-4o", "gemini-1.5-flash", "claude-3-5-sonnet-20241022".
    base_url : str | None
        Override base URL for OpenAI-compatible providers (Azure, Together, Groq).
    system_prompt : str
        System instruction prepended to every probe.
    endpoint_url : str | None
        Used for "custom" provider — the full POST URL.
    headers : dict | None
        Extra headers for "custom" provider.
    request_body_template : dict | None
        JSON template for "custom" provider. Use ``{prompt}`` as placeholder.
    max_tokens : int
        Response length cap sent to the provider.
    temperature : float
        Sampling temperature.
    """
    provider: str                               # openai | gemini | anthropic | custom
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    system_prompt: str = "You are a helpful assistant."
    endpoint_url: str | None = None
    headers: dict[str, str] | None = None
    request_body_template: dict | None = None
    max_tokens: int = 512
    temperature: float = 0.0                    # deterministic for fairness audits


@dataclass
class ProbeResult:
    """Result of sending one probe to the target LLM."""
    probe_id: str
    audit_id: str
    prompt_text: str
    response_text: str
    response_time_ms: float
    demographic_group: str
    attribute_tested: str
    tokens_used: int | None
    pair_id: str
    base_prompt_index: int
    scenario: str
    status: str = "success"          # success | failed | timeout
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# LLMConnector
# ---------------------------------------------------------------------------

class LLMConnector:
    """
    Sends probe batteries to the customer's AI application over the network.

    Usage
    -----
    >>> cfg = ConnectorConfig(provider="openai", api_key="sk-...", model="gpt-4o")
    >>> connector = LLMConnector()
    >>> results = await connector.send_probes(cfg, battery, audit_id="audit-xyz")
    """

    def __init__(self) -> None:
        self._db: _fs.Client | None = None

    # ------------------------------------------------------------------
    # Lazy Firestore client
    # ------------------------------------------------------------------
    @property
    def db(self) -> _fs.Client:
        if self._db is None:
            self._db = _fs.Client()
        return self._db

    # ================================================================
    # PUBLIC — send_probes
    # ================================================================

    async def send_probes(
        self,
        config: ConnectorConfig,
        battery: list,          # list[ProbeSet] from probe_generator
        audit_id: str,
    ) -> list[ProbeResult]:
        """
        Send every probe in *battery* to the target LLM concurrently.

        Concurrency is capped at MAX_CONCURRENT (10) using an asyncio
        semaphore.  Each probe times out after 30 s; one automatic retry
        is attempted before marking the result as failed.

        Results are streamed to Firestore in batches as they complete.

        Parameters
        ----------
        config      : ConnectorConfig
        battery     : list[ProbeSet]  — from ProbeGenerator.generate_probe_battery()
        audit_id    : str

        Returns
        -------
        list[ProbeResult]
        """
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        results: list[ProbeResult] = []

        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            tasks = [
                self._send_one_with_retry(sem, client, config, probe, audit_id)
                for probe in battery
            ]
            completed = await asyncio.gather(*tasks, return_exceptions=False)

        results = list(completed)
        await self._flush_to_firestore(audit_id, results)
        return results

    # ================================================================
    # INTERNAL — single probe dispatch
    # ================================================================

    async def _send_one_with_retry(
        self,
        sem: asyncio.Semaphore,
        client: httpx.AsyncClient,
        config: ConnectorConfig,
        probe: Any,
        audit_id: str,
    ) -> ProbeResult:
        """Attempt once, retry once on failure, then mark failed."""
        async with sem:
            for attempt in range(2):
                result = await self._dispatch(client, config, probe, audit_id)
                if result.status == "success":
                    return result
                if attempt == 0:
                    log.warning(
                        "Probe %s failed (attempt 1), retrying... error=%s",
                        probe.probe_id, result.error,
                    )
                    await asyncio.sleep(1.5)
            return result  # second attempt result (may still be failed)

    async def _dispatch(
        self,
        client: httpx.AsyncClient,
        config: ConnectorConfig,
        probe: Any,
        audit_id: str,
    ) -> ProbeResult:
        """Route probe to the correct provider handler."""
        start = time.perf_counter()
        try:
            match config.provider:
                case "openai":
                    text, tokens = await self._call_openai(client, config, probe.prompt_text)
                case "gemini":
                    text, tokens = await self._call_gemini(config, probe.prompt_text)
                case "anthropic":
                    text, tokens = await self._call_anthropic(client, config, probe.prompt_text)
                case "custom":
                    text, tokens = await self._call_custom(client, config, probe.prompt_text)
                case _:
                    raise ValueError(f"Unknown provider: {config.provider!r}")

            elapsed_ms = (time.perf_counter() - start) * 1000
            return ProbeResult(
                probe_id=probe.probe_id,
                audit_id=audit_id,
                prompt_text=probe.prompt_text,
                response_text=text,
                response_time_ms=round(elapsed_ms, 2),
                demographic_group=probe.demographic_group,
                attribute_tested=probe.attribute_tested,
                tokens_used=tokens,
                pair_id=probe.pair_id,
                base_prompt_index=probe.base_prompt_index,
                scenario=probe.scenario,
                status="success",
            )

        except httpx.TimeoutException:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._failed_result(probe, audit_id, elapsed_ms, "timeout", "Request timed out after 30 s")
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._failed_result(probe, audit_id, elapsed_ms, "failed", str(exc))

    @staticmethod
    def _failed_result(probe: Any, audit_id: str, elapsed_ms: float, status: str, error: str) -> ProbeResult:
        return ProbeResult(
            probe_id=probe.probe_id,
            audit_id=audit_id,
            prompt_text=probe.prompt_text,
            response_text="",
            response_time_ms=round(elapsed_ms, 2),
            demographic_group=probe.demographic_group,
            attribute_tested=probe.attribute_tested,
            tokens_used=None,
            pair_id=probe.pair_id,
            base_prompt_index=probe.base_prompt_index,
            scenario=probe.scenario,
            status=status,
            error=error,
        )

    # ================================================================
    # PROVIDER HANDLERS
    # ================================================================

    # ------------------------------------------------------------------
    # 1. OpenAI-compatible (OpenAI, Azure, Together, Groq)
    # ------------------------------------------------------------------
    async def _call_openai(
        self,
        client: httpx.AsyncClient,
        config: ConnectorConfig,
        prompt: str,
    ) -> tuple[str, int | None]:
        """POST /v1/chat/completions to OpenAI-compatible endpoint."""
        base_url = (config.base_url or "https://api.openai.com").rstrip("/")
        url = f"{base_url}/v1/chat/completions"

        payload = {
            "model": config.model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens")
        return text, tokens

    # ------------------------------------------------------------------
    # 2. Google Gemini (via SDK — synchronous, run in executor)
    # ------------------------------------------------------------------
    async def _call_gemini(
        self,
        config: ConnectorConfig,
        prompt: str,
    ) -> tuple[str, int | None]:
        """Call Gemini via google-generativeai SDK in a thread executor."""
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        genai.configure(api_key=config.api_key)
        model_name = config.model or "gemini-2.0-flash"

        def _sync_call() -> tuple[str, int | None]:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=config.system_prompt,
            )
            response = model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    max_output_tokens=config.max_tokens,
                    temperature=config.temperature,
                ),
            )
            text = response.text
            # Gemini usage metadata (may be None for some models)
            usage = getattr(response, "usage_metadata", None)
            tokens = None
            if usage:
                tokens = getattr(usage, "total_token_count", None)
            return text, tokens

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_call)

    # ------------------------------------------------------------------
    # 3. Anthropic Claude
    # ------------------------------------------------------------------
    async def _call_anthropic(
        self,
        client: httpx.AsyncClient,
        config: ConnectorConfig,
        prompt: str,
    ) -> tuple[str, int | None]:
        """POST to Anthropic Messages API."""
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": config.model or "claude-3-5-haiku-20241022",
            "max_tokens": config.max_tokens,
            "system": config.system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return text, tokens

    # ------------------------------------------------------------------
    # 4. Custom endpoint (template substitution)
    # ------------------------------------------------------------------
    async def _call_custom(
        self,
        client: httpx.AsyncClient,
        config: ConnectorConfig,
        prompt: str,
    ) -> tuple[str, int | None]:
        """
        POST to a user-defined URL.

        The request_body_template is a JSON-serialisable dict with
        ``{prompt}`` as the substitution target, e.g.:

            {
              "input": "{prompt}",
              "parameters": {"max_new_tokens": 256}
            }

        The response is expected to be JSON.  BiasProbe will attempt to
        extract text from common patterns:
          - response["output"]
          - response["text"]
          - response["choices"][0]["message"]["content"]
          - response["content"][0]["text"]
          - str(response)   (fallback)
        """
        if not config.endpoint_url:
            raise ValueError("ConnectorConfig.endpoint_url must be set for 'custom' provider")

        template = config.request_body_template or {"prompt": "{prompt}"}
        body_str = json.dumps(template).replace("{prompt}", prompt.replace('"', '\\"'))
        body = json.loads(body_str)

        headers = {"Content-Type": "application/json", **(config.headers or {})}
        resp = await client.post(config.endpoint_url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        text = self._extract_custom_text(data)
        return text, None

    @staticmethod
    def _extract_custom_text(data: Any) -> str:
        """Try common response shapes to extract the generated text."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("output", "text", "generated_text", "result", "answer", "response"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            # OpenAI shape
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                pass
            # Anthropic shape
            try:
                return data["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                pass
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                for key in ("generated_text", "text", "output"):
                    if key in first:
                        return str(first[key])
        return json.dumps(data)  # last resort

    # ================================================================
    # FIRESTORE PERSISTENCE
    # ================================================================

    async def _flush_to_firestore(
        self,
        audit_id: str,
        results: list[ProbeResult],
    ) -> None:
        """
        Write all ProbeResult objects to Firestore in batched writes.

        Path: /audits/{audit_id}/probe_results/{probe_id}
        Also updates the parent audit document with completion counters.
        """
        if not results:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_flush, audit_id, results)

    def _sync_flush(self, audit_id: str, results: list[ProbeResult]) -> None:
        BATCH_SIZE = 400
        audit_ref = self.db.collection(FIRESTORE_COLLECTION).document(audit_id)
        results_ref = audit_ref.collection("probe_results")

        dicts = [r.to_dict() for r in results]
        success_count = sum(1 for r in results if r.status == "success")
        failed_count  = sum(1 for r in results if r.status != "success")

        for i in range(0, len(dicts), BATCH_SIZE):
            batch = self.db.batch()
            for d in dicts[i : i + BATCH_SIZE]:
                batch.set(results_ref.document(d["probe_id"]), d)
            batch.commit()
            log.info("Firestore: flushed %d probe_results (offset %d)", len(dicts[i : i + BATCH_SIZE]), i)

        # Update audit-level counters
        audit_ref.set(
            {
                "probes_complete": _fs.Increment(len(results)),
                "probes_success":  _fs.Increment(success_count),
                "probes_failed":   _fs.Increment(failed_count),
            },
            merge=True,
        )
        log.info(
            "audit=%s  total=%d  success=%d  failed=%d",
            audit_id, len(results), success_count, failed_count,
        )

    # ================================================================
    # CONVENIENCE — validate config with a test prompt
    # ================================================================

    async def validate_config(self, config: ConnectorConfig) -> dict[str, Any]:
        """
        Send a single benign test prompt to verify connectivity and credentials.

        Returns
        -------
        dict  with keys: ok (bool), response_text (str), error (str | None)
        """
        test_probe_stub = _TestProbeStub()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                match config.provider:
                    case "openai":
                        text, _ = await self._call_openai(client, config, test_probe_stub.prompt_text)
                    case "gemini":
                        text, _ = await self._call_gemini(config, test_probe_stub.prompt_text)
                    case "anthropic":
                        text, _ = await self._call_anthropic(client, config, test_probe_stub.prompt_text)
                    case "custom":
                        text, _ = await self._call_custom(client, config, test_probe_stub.prompt_text)
                    case _:
                        return {"ok": False, "response_text": "", "error": f"Unknown provider: {config.provider}"}
            return {"ok": True, "response_text": text[:200], "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "response_text": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# Internal stub for validation
# ---------------------------------------------------------------------------
class _TestProbeStub:
    probe_id         = "test-probe-0"
    pair_id          = "test-pair-0"
    prompt_text      = "Reply with exactly three words: 'Connection test successful.'"
    demographic_group = "test"
    attribute_tested  = "test"
    base_prompt_index = 0
    scenario          = "test"
