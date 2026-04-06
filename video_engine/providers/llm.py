"""Language model provider implementations."""

from typing import Optional


class ClaudeLLM:
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        import os
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model

    def generate(self, prompt: str, system: str = "", max_tokens: int = 4096, **kwargs) -> str:
        from video_engine.clients.claude import generate as _gen
        return _gen(prompt=prompt, system=system, model=kwargs.get("model", self.model), max_tokens=max_tokens)


class OpenAILLM:
    """OpenAI GPT LLM provider (future expansion)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def generate(self, prompt: str, system: str = "", max_tokens: int = 4096, **kwargs) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def get_llm_provider(name: str = "claude", **kwargs):
    """Factory for LLM providers."""
    providers = {
        "claude": ClaudeLLM,
        "openai": OpenAILLM,
    }
    cls = providers.get(name, ClaudeLLM)
    return cls(**kwargs)
