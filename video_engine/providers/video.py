"""Video clip generation provider implementations."""

from typing import Optional, Any


class GrokVideoProvider:
    """xAI Grok video generation. Image-to-video animation."""

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("XAI_API_KEY")

    async def generate_video(self, prompt: str, output_path: str, duration: int = 8,
                              aspect_ratio: str = "9:16", image_url: str = None,
                              progress_callback: Any = None, **kwargs) -> dict:
        from video_engine.clients.grok import generate_video_async as _gen
        return await _gen(
            prompt=prompt, output_path=output_path, duration=duration,
            aspect_ratio=aspect_ratio, image_url=image_url,
            progress_callback=progress_callback, **kwargs,
        )


def get_video_provider(name: str = "grok", **kwargs):
    """Factory for video providers."""
    providers = {
        "grok": GrokVideoProvider,
    }
    cls = providers.get(name, GrokVideoProvider)
    return cls(**kwargs)
