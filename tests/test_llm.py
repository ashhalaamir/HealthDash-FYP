import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import llm


class Resp:
    def __init__(self, data, status=200):
        self.data, self.status_code, self.text = data, status, str(data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise llm.requests.HTTPError(response=self)

    def json(self):
        return self.data


def test_404_message_explains_and_points_to_the_model_list(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123")
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: Resp({"error": {"message": "models/old is not found"}}, 404))
    with pytest.raises(llm.LLMError) as e:
        llm.call_llm("hi", "gemini")
    msg = str(e.value)
    assert "HTTP 404" in msg and "models/old is not found" in msg and "python -m src.llm models" in msg
    assert "SECRET-KEY-123" not in msg


def test_other_http_errors_keep_the_provider_message_without_the_hint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: Resp({"error": {"message": "quota exceeded"}}, 429))
    with pytest.raises(llm.LLMError) as e:
        llm.call_llm("hi", "gemini")
    assert "HTTP 429" in str(e.value) and "quota exceeded" in str(e.value) and "src.llm models" not in str(e.value)


def test_list_models_keeps_only_text_generation_models(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seen = {}
    def fake_get(url, **kw):
        seen.update(url=url, **kw)
        return Resp({"models": [
            {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/text-embedding-9", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-3.1-flash-lite", "supportedGenerationMethods": ["generateContent"]}]})
    monkeypatch.setattr(llm.requests, "get", fake_get)
    assert llm.list_models("gemini") == ["gemini-3.1-flash-lite", "gemini-3.8-flash"]
    assert seen["headers"]["x-goog-api-key"] == "k" and "key=" not in seen["url"]


def test_list_models_ollama_and_unsupported(monkeypatch):
    monkeypatch.setattr(llm.requests, "get", lambda url, **kw: Resp({"models": [{"name": "llama3.1:8b"}]}))
    assert llm.list_models("ollama") == ["llama3.1:8b"]
    with pytest.raises(llm.LLMError, match="supported for gemini and ollama"):
        llm.list_models("openai")


def test_list_models_needs_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.list_models("gemini")