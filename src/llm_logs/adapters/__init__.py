"""Provider adapters. None of them imports a provider SDK."""

from llm_logs.adapters.anthropic import AnthropicAdapter
from llm_logs.adapters.base import Accumulator, Adapter, Extracted, by_name, detect, register
from llm_logs.adapters.groq import GroqAdapter
from llm_logs.adapters.openai import OpenAIAdapter

register(GroqAdapter())
register(OpenAIAdapter())
register(AnthropicAdapter())

__all__ = [
    "Accumulator",
    "Adapter",
    "AnthropicAdapter",
    "Extracted",
    "GroqAdapter",
    "OpenAIAdapter",
    "by_name",
    "detect",
    "register",
]
