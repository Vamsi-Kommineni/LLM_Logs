"""Groq. The API is OpenAI-shaped, plus timing fields and ``x_groq``."""

from __future__ import annotations

from llm_logs.adapters._openai_compat import OpenAICompatibleAdapter


class GroqAdapter(OpenAICompatibleAdapter):
    name = "groq"
    module_prefixes = ("groq.",)
