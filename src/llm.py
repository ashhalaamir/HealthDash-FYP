"""One function, any provider. Switch with the LLM_PROVIDER environment variable (or .env file):

    LLM_PROVIDER=gemini   GEMINI_API_KEY=...   GEMINI_MODEL=gemini-2.5-flash
    LLM_PROVIDER=openai   OPENAI_API_KEY=...   OPENAI_MODEL=gpt-4o-mini
    LLM_PROVIDER=ollama   OLLAMA_MODEL=llama3.1:8b   (OLLAMA_URL defaults to http://localhost:11434)

Every provider is asked for JSON and returns the model's reply as a plain string.
Default model names are starting points: free-tier availability changes, so if you get a 404 or a
quota error, pick another model in .env. Nothing else in the project needs to change.
"""
import os

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:      # python-dotenv is optional; plain environment variables work too
    pass

TIMEOUT = 60


class LLMError(RuntimeError):
    """The LLM call failed. Callers should fall back to the rule-based spec."""


def _need(name):
    value = os.environ.get(name)
    if not value:
        raise LLMError(f"{name} is not set. Add it to your .env file or environment.")
    return value


def _post(url, **kwargs):
    try:
        r = requests.post(url, timeout=TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        raise LLMError(f"LLM request failed: {e}") from e


def _gemini(prompt):
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    data = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                 headers={"x-goog-api-key": _need("GEMINI_API_KEY")},
                 json={"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}})
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected Gemini response: {str(data)[:200]}") from e


def _openai(prompt):
    data = _post("https://api.openai.com/v1/chat/completions",
                 headers={"Authorization": f"Bearer {_need('OPENAI_API_KEY')}"},
                 json={"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0.2,
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "user", "content": prompt}]})
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected OpenAI response: {str(data)[:200]}") from e


def _ollama(prompt):
    base = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    data = _post(f"{base}/api/generate",
                 json={"model": os.environ.get("OLLAMA_MODEL", "llama3.1:8b"), "prompt": prompt,
                       "format": "json", "stream": False, "options": {"temperature": 0.2}})
    if "response" not in data:
        raise LLMError(f"Unexpected Ollama response: {str(data)[:200]}")
    return data["response"]


PROVIDERS = {"gemini": _gemini, "openai": _openai, "ollama": _ollama}


def call_llm(prompt, provider=None):
    provider = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")
    return PROVIDERS[provider](prompt)