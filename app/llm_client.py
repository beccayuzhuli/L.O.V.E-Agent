"""
================================================================================
L.O.V.E. Relationship Support Agent — LLM Client
RSM 8430 Group 18
================================================================================

Thin wrapper around the LLM endpoint. Swap the implementation here to change
providers without touching any other file.

Currently supports:
  - Course-provided Qwen endpoint (OpenAI-compatible API)
  - Any OpenAI-compatible endpoint

Configuration via environment variables:
  LLM_API_BASE  — base URL (e.g. http://localhost:8000/v1)
  LLM_API_KEY   — API key (default: "empty" for local endpoints)
  LLM_MODEL     — model name (default: qwen3-30b-a3b-fp8)

Usage:
    from app.llm_client import generate_text
    response = generate_text("You are a helpful assistant.", "Hello!")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import requests

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

LLM_API_BASE: str = os.environ.get("LLM_API_BASE", "http://localhost:8000/v1")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "empty")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "qwen3-30b-a3b-fp8")

MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def _strip_thinking_tags(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from a response.

    Handles both properly-closed blocks and blocks that were truncated by the
    token limit (no closing tag), so the caller always gets the actual answer.
    """
    # Remove complete <think>...</think> blocks (greedy across newlines).
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    # Remove any unclosed <think> block that consumed the rest of the string.
    text = re.sub(r"<think>[\s\S]*$", "", text).strip()
    return text


def _extract_message_text(data: dict) -> str:
    """
    Extract assistant text from OpenAI-compatible responses with tolerance for
    string or list-based message content payloads.
    """
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response format: {data}") from exc

    message = choice.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    # Some compatible providers expose plain text directly on the choice.
    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return text

    raise RuntimeError(f"LLM response did not include assistant text content: {data}")


def _should_try_lowercase_model(status_code: int, response_text: str, model_name: str) -> bool:
    if model_name == model_name.lower():
        return False
    if status_code not in (400, 404):
        return False
    lowered = (response_text or "").lower()
    return "model" in lowered and ("not found" in lowered or "does not exist" in lowered)


def generate_text(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_seconds: int = 120,
    retries: int = 2,
) -> str:
    """
    Send a system + user message to the LLM and return the assistant's reply.

    Args:
        system_prompt: The system-level instruction.
        user_prompt:   The user-level message / prompt.

    Returns:
        The LLM's response text.

    Raises:
        RuntimeError: If the API call fails after the request completes.
        requests.RequestException: If the network call itself fails.
    """
    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    model_name = LLM_MODEL

    # Qwen3 models use "thinking mode" by default, spending hundreds of tokens
    # on internal reasoning before producing the actual answer.  For this app
    # we want direct, concise responses, so we disable thinking via the
    # /nothink instruction at the start of the user turn — this is a
    # training-time feature that works on every vLLM version.
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            effective_user_prompt = user_prompt
            if "qwen3" in model_name.lower():
                effective_user_prompt = f"/nothink\n{user_prompt}"

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": effective_user_prompt},
                ],
                "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
                "temperature": temperature if temperature is not None else TEMPERATURE,
            }
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
            if resp.status_code >= 400:
                response_text = (resp.text or "").strip()
                if _should_try_lowercase_model(resp.status_code, response_text, model_name):
                    fallback = model_name.lower()
                    LOGGER.warning(
                        "LLM model %s not found; retrying with lowercase alias %s",
                        model_name,
                        fallback,
                    )
                    model_name = fallback
                    continue
                raise RuntimeError(
                    f"LLM API error {resp.status_code} for model '{model_name}': {response_text[:300]}"
                )

            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"LLM API returned non-JSON response for model '{model_name}': {(resp.text or '')[:300]}"
                ) from exc

            content = _extract_message_text(data)
            cleaned = _strip_thinking_tags(content)
            if not cleaned:
                raise RuntimeError("LLM returned empty content after removing reasoning tags.")
            return cleaned
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(0.4 * (attempt + 1))

    if last_error:
        raise last_error
    raise RuntimeError("LLM request failed with unknown error.")
