"""Text-to-speech provider implementations."""

from typing import Optional


class ElevenLabsTTS:
    """ElevenLabs TTS provider."""

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")

    def generate_speech(self, text: str, voice: str, output_path: str, speed: Optional[float] = None) -> str:
        from video_engine.clients.elevenlabs import generate_speech as _gen
        return _gen(text=text, voice=voice, output_path=output_path, speed=speed)


class OpenAITTS:
    """OpenAI TTS provider (future expansion)."""

    def __init__(self, api_key: Optional[str] = None, voice: str = "alloy"):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.default_voice = voice

    def generate_speech(self, text: str, voice: str, output_path: str, speed: Optional[float] = None) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice or self.default_voice,
            input=text,
            speed=speed or 1.0,
        )
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        response.stream_to_file(output_path)
        return output_path
