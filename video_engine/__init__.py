"""Video Engine — AI video generation pipeline.

Usage:
    from video_engine import VideoPipeline, VideoConfig, VideoFormat

    pipeline = VideoPipeline()
    result = await pipeline.generate(
        concept={"title": "My Video", "narration": ["Line 1", "Line 2"]},
        config=VideoConfig(channel_name="My Channel", art_style="..."),
    )
"""

from video_engine.pipeline.core import VideoConfig, VideoFormat, VideoResult
from video_engine.pipeline.engine import VideoPipeline

__all__ = ["VideoPipeline", "VideoConfig", "VideoFormat", "VideoResult"]
__version__ = "0.1.0"
