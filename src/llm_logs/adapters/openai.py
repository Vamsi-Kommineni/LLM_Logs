"""OpenAI: Chat Completions, the Responses API, legacy completions and embeddings.

When the OpenAI SDK is pointed at another server through ``base_url``, responses
are still detected as ``openai``. Pass ``provider="..."`` to ``trace`` to record
the real provider.
"""

from __future__ import annotations

from llm_logs.adapters._openai_compat import OpenAICompatibleAdapter


class OpenAIAdapter(OpenAICompatibleAdapter):
    name = "openai"
    module_prefixes = ("openai.",)
