"""Image generation provider implementations."""

from typing import Optional


class GrokImageProvider:
    """xAI Grok image generation. Good for general scenes, cheap."""

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("XAI_API_KEY")

    def generate_image(self, prompt: str, output_path: str, size: str = "2k", **kwargs) -> str:
        from video_engine.clients.grok import generate_image as _gen
        return _gen(prompt=prompt, output_path=output_path, **kwargs)


class OpenAIImageProvider:
    """OpenAI gpt-image / DALL-E. Good for text, diagrams, precise layouts."""

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def generate_image(self, prompt: str, output_path: str, size: str = "1024x1536", **kwargs) -> str:
        from video_engine.clients.grok import generate_image_dalle as _gen
        return _gen(prompt=prompt, output_path=output_path, size=size, **kwargs)


def get_image_provider(name: str = "grok", **kwargs):
    """Factory for image providers."""
    providers = {
        "grok": GrokImageProvider,
        "openai": OpenAIImageProvider,
        "dalle": OpenAIImageProvider,
    }
    cls = providers.get(name, GrokImageProvider)
    return cls(**kwargs)
