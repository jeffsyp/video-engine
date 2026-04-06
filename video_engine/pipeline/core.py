"""Core video generation pipeline.

Extensible, provider-agnostic pipeline for generating narrated videos.
Each step is pluggable — swap TTS providers, image generators, or LLMs
without changing the pipeline logic.

Usage:
    from video_engine.pipeline import VideoPipeline, VideoConfig

    pipeline = VideoPipeline()
    result = await pipeline.generate(
        concept=concept_dict,
        config=VideoConfig(
            channel_name="My Channel",
            art_style="Clean infographic style...",
        ),
    )
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
from enum import Enum
import os


class VideoFormat(Enum):
    SHORT = "short"        # 9:16, 720x1280, <60s
    MIDFORM = "midform"    # 16:9, 1920x1080, 3-5 min
    LONGFORM = "longform"  # 16:9, 1920x1080, 10+ min


@dataclass
class VideoConfig:
    """Configuration for a video generation run. All pipeline behavior derives from this."""

    # Content identity
    channel_name: str = "Default"
    niche: str = "general"
    art_style: str = "Bold cartoon style, thick outlines, bright colors."
    category: str = "Entertainment"

    # Format
    format: VideoFormat = VideoFormat.SHORT
    width: int = 0
    height: int = 0

    # Output
    output_dir: str = "output"
    run_id: Optional[int] = None

    # Voice
    voice_id: str = "56bWURjYFHyYyVf490Dp"
    narration_speed: Optional[float] = None

    # Provider overrides (None = use defaults)
    image_provider: Optional[str] = None   # "grok", "openai", "dalle"
    video_provider: Optional[str] = None   # "grok", "runway", "kling"
    tts_provider: Optional[str] = None     # "elevenlabs", "openai"
    llm_provider: Optional[str] = None     # "claude", "openai", "gemini"

    # Quality
    subtitle_style: str = "karaoke"        # "karaoke", "standard", "none"
    background_music: Optional[str] = None  # path to music file

    # Callbacks
    on_progress: Optional[Callable] = field(default=None, repr=False)
    on_log: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self):
        if self.width == 0:
            self.width = 720 if self.format == VideoFormat.SHORT else 1920
        if self.height == 0:
            self.height = 1280 if self.format == VideoFormat.SHORT else 1080

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    @property
    def is_long_form(self) -> bool:
        return self.format in (VideoFormat.MIDFORM, VideoFormat.LONGFORM)

    @property
    def aspect_ratio(self) -> str:
        return "9:16" if self.is_portrait else "16:9"

    @property
    def resolution_label(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class AudioSegment:
    """A single narration line with its audio file."""
    index: int
    text: str
    path: str
    duration: float


@dataclass
class Visual:
    """A planned visual for one narration line."""
    index: int
    type: str              # "video", "image", "diagram"
    prompt: str
    video_prompt: str = "" # motion description for video types
    character: str = ""    # character name for consistency
    label: str = ""        # on-screen label (e.g., "#3")


@dataclass
class VisualAsset:
    """A generated visual asset (image or video file)."""
    index: int
    type: str           # "video" or "image"
    path: str
    source: str = ""    # which provider generated it


@dataclass
class Segment:
    """A rendered segment — one narration line paired with its visual."""
    index: int
    path: str
    duration: float


@dataclass
class VideoResult:
    """Final output of the pipeline."""
    video_path: str
    duration: float
    title: str = ""
    subtitle_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    narration_count: int = 0
    visual_count: int = 0
    metadata: dict = field(default_factory=dict)


# --- Provider Protocols ---
# These define the interface for swappable providers.
# Implement these to add new TTS, image, video, or LLM providers.

class TTSProvider(Protocol):
    """Text-to-speech provider interface."""
    def generate_speech(self, text: str, voice: str, output_path: str, speed: Optional[float] = None) -> str: ...


class ImageProvider(Protocol):
    """Image generation provider interface."""
    def generate_image(self, prompt: str, output_path: str, size: str = "1024x1024", **kwargs) -> str: ...


class VideoProvider(Protocol):
    """Video clip generation provider interface."""
    async def generate_video(self, prompt: str, output_path: str, duration: int = 8,
                              aspect_ratio: str = "9:16", image_url: str = None,
                              progress_callback: Any = None, **kwargs) -> dict: ...


class LLMProvider(Protocol):
    """Language model provider interface."""
    def generate(self, prompt: str, system: str = "", max_tokens: int = 4096, **kwargs) -> str: ...
