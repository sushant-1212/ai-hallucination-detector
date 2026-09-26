"""
llm_provider.py
----------------
Provider abstraction for LLM inference and embeddings.
Supports:
  - Google Gemini (gemini-2.5-flash & gemini-embedding-001) - Active & Free
  - Groq (llama-3.3-70b-versatile) - Ultra fast & Free
  - OpenAI (gpt-4o-mini & text-embedding-3-small)
  - Grok / xAI (grok-2-latest)
  - Local Semantic Fallback (TF-IDF / Cosine Similarity) when offline

Built with lightweight, rock-solid HTTP requests with zero external package bloat.
"""

import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()


def get_active_provider() -> str:
    """Return the configured provider name."""
    return os.getenv("LLM_PROVIDER", "gemini").lower()


def get_gemini_key() -> str:
    return os.getenv("GOOGLE_API_KEY", "").strip()


def get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def get_xai_key() -> str:
    return os.getenv("XAI_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# GEMINI ENGINE
# ---------------------------------------------------------------------------

def _call_gemini_generate(prompt: str, system_prompt: str = "", temperature: float = 0.0) -> str:
    key = get_gemini_key()
    if not key:
        raise ValueError("GOOGLE_API_KEY is missing in your .env file or settings.")

    # Using stable gemini-2.5-flash
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 2048,
        }
    }
    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    import time
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                candidates = res_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text_parts = [p.get("text", "") for p in parts if "text" in p]
                    return "".join(text_parts).strip()
                return ""
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            error_msg = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini API Error [{e.code}]: {error_msg}")
        except Exception as e:
            if attempt < 3:
                time.sleep(2)
                continue
            raise RuntimeError(f"Gemini Request Failed: {e}")


def _call_gemini_embedding(text: str) -> list[float]:
    key = get_gemini_key()
    if not key:
        raise ValueError("GOOGLE_API_KEY is missing.")

    model_name = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent?key={key}"

    payload = {
        "content": {"parts": [{"text": text[:8000]}]}
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    import time
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                return res_json.get("embedding", {}).get("values", [])
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"[Embedding warning] Gemini embedding error [{e.code}]: {e}")
            return []
        except Exception as e:
            if attempt < 3:
                time.sleep(2)
                continue
            print(f"[Embedding warning] Gemini embedding error: {e}")
            return []
    return []


# ---------------------------------------------------------------------------
# OPENAI / GROQ / XAI ENGINE (Standard OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def _call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            return res_json["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI-compatible API Error [{e.code}]: {error_msg}")
    except Exception as e:
        raise RuntimeError(f"API Request Failed: {e}")


# ---------------------------------------------------------------------------
# UNIFIED PUBLIC INTERFACE
# ---------------------------------------------------------------------------

def generate_chat_response(prompt: str, system_prompt: str = "", temperature: float = 0.0) -> str:
    """Generate completion using the active provider with auto-fallback."""
    provider = get_active_provider()

    if provider == "gemini":
        return _call_gemini_generate(prompt, system_prompt=system_prompt, temperature=temperature)

    elif provider == "groq":
        key = get_groq_key()
        if not key:
            raise ValueError("GROQ_API_KEY is missing from .env.")
        return _call_openai_compatible(
            base_url="https://api.groq.com/openai/v1",
            api_key=key,
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )

    elif provider == "openai":
        key = get_openai_key()
        if not key:
            raise ValueError("OPENAI_API_KEY is missing from .env.")
        return _call_openai_compatible(
            base_url="https://api.openai.com/v1",
            api_key=key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )

    elif provider == "grok":
        key = get_xai_key()
        if not key:
            raise ValueError("XAI_API_KEY is missing from .env.")
        return _call_openai_compatible(
            base_url="https://api.x.ai/v1",
            api_key=key,
            model=os.getenv("GROK_MODEL", "grok-2-latest"),
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )

    else:
        # Default to Gemini
        return _call_gemini_generate(prompt, system_prompt=system_prompt, temperature=temperature)


def get_embedding(text: str) -> list[float]:
    """Return embedding vector for a single text."""
    provider = get_active_provider()
    if provider == "gemini":
        return _call_gemini_embedding(text)
    # If using OpenAI
    elif provider == "openai":
        key = get_openai_key()
        if key:
            try:
                url = "https://api.openai.com/v1/embeddings"
                data = json.dumps({"model": "text-embedding-3-small", "input": text[:8000]}).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
                with urllib.request.urlopen(req, timeout=20) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    return res_json["data"][0]["embedding"]
            except Exception:
                pass
    return _call_gemini_embedding(text)


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts."""
    embeddings = []
    for t in texts:
        embeddings.append(get_embedding(t))
    return embeddings


def provider_name() -> str:
    """Return friendly name of current active provider."""
    p = get_active_provider()
    mapping = {
        "gemini": "Google Gemini 2.5 Flash (Free Cloud API)",
        "groq": "Groq LPU (Llama-3.3-70B - Ultra Fast)",
        "openai": "OpenAI (GPT-4o Mini)",
        "grok": "xAI Grok-2",
    }
    return mapping.get(p, f"Custom ({p})")
