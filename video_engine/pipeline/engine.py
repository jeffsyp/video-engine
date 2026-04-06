"""Video generation engine — the main pipeline orchestrator.

Coordinates narration, visual planning, asset generation, segment assembly,
and final rendering. Provider-agnostic — uses whatever providers are configured.
"""

import asyncio
import json
import os
import re
import subprocess
import time
from typing import Optional

import structlog

from video_engine.pipeline.core import (
    AudioSegment, VideoConfig, VideoFormat, VideoResult, Visual, VisualAsset, Segment,
)
from video_engine.providers.tts import ElevenLabsTTS
from video_engine.providers.image import GrokImageProvider, OpenAIImageProvider, get_image_provider
from video_engine.providers.video import GrokVideoProvider, get_video_provider
from video_engine.providers.llm import ClaudeLLM, get_llm_provider

logger = structlog.get_logger()


class VideoPipeline:
    """Main video generation pipeline.

    Each step is independently cacheable — if output files exist from a
    previous run, the step is skipped. This enables resume-on-crash.
    """

    def __init__(
        self,
        tts=None,
        image_provider=None,
        portrait_image_provider=None,
        video_provider=None,
        llm=None,
    ):
        self.tts = tts or ElevenLabsTTS()
        self.image_provider = image_provider or GrokImageProvider()
        self.portrait_image_provider = portrait_image_provider or OpenAIImageProvider()
        self.video_provider = video_provider or GrokVideoProvider()
        self.llm = llm or ClaudeLLM()

    async def generate(self, concept: dict, config: VideoConfig) -> VideoResult:
        """Run the full pipeline: narration → visuals → segments → render → subtitles."""
        start_time = time.time()
        title = concept.get("title", "Untitled")
        narration_lines = concept.get("narration", [])
        voice_id = concept.get("voice_id", config.voice_id)

        if not narration_lines:
            raise ValueError("No narration lines in concept")

        # Setup directories
        narr_dir = os.path.join(config.output_dir, "narration")
        images_dir = os.path.join(config.output_dir, "images")
        clips_dir = os.path.join(config.output_dir, "clips")
        segments_dir = os.path.join(config.output_dir, "segments")
        for d in [narr_dir, images_dir, clips_dir, segments_dir]:
            os.makedirs(d, exist_ok=True)

        # Strip emojis from narration
        emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
        narration_lines = [emoji_pattern.sub("", line) for line in narration_lines]

        # --- Step 1: Narration ---
        await self._progress(config, "generating narration", f"{len(narration_lines)} lines")
        audio_segments = await self._generate_narration(
            narration_lines, voice_id, narr_dir, config,
        )
        total_dur = sum(a.duration for a in audio_segments)
        await self._progress(config, "narration done", f"{len(audio_segments)} lines, {total_dur:.0f}s")

        # --- Step 2: Visual Planning ---
        visual_plan_path = os.path.join(config.output_dir, "visual_plan.json")
        visuals = await self._plan_visuals(
            audio_segments, title, visual_plan_path, config,
        )

        # --- Step 3: Generate Visual Assets ---
        visual_assets = await self._generate_visuals(
            visuals, audio_segments, images_dir, clips_dir, config,
        )

        # --- Step 4: Create Segments ---
        segments = await self._create_segments(
            audio_segments, visual_assets, segments_dir, config,
        )

        # --- Step 5: Assemble ---
        concat_path = os.path.join(config.output_dir, "raw_concat.mp4")
        await self._assemble(segments, concat_path, config)

        # --- Step 6: Subtitles ---
        final_path = os.path.join(config.output_dir, "final.mp4")
        subtitle_path = await self._burn_subtitles(
            concat_path, final_path, audio_segments, visuals, config,
        )

        elapsed = time.time() - start_time
        await self._progress(config, "done", f"{elapsed:.0f}s total")

        return VideoResult(
            video_path=final_path,
            duration=total_dur,
            title=title,
            subtitle_path=subtitle_path,
            narration_count=len(audio_segments),
            visual_count=len(visual_assets),
            metadata={"elapsed_seconds": int(elapsed)},
        )

    # --- Step implementations ---

    async def _generate_narration(
        self, lines: list[str], voice_id: str, output_dir: str, config: VideoConfig,
    ) -> list[AudioSegment]:
        """Generate TTS audio for each narration line. Parallel, with caching."""
        total = len(lines)

        def _gen_one(i: int, text: str) -> AudioSegment:
            path = os.path.join(output_dir, f"line_{i}.mp3")
            cached = os.path.exists(path)
            if not cached:
                for attempt in range(3):
                    try:
                        self.tts.generate_speech(text=text, voice=voice_id, output_path=path, speed=config.narration_speed)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise RuntimeError(f"TTS failed for line {i}: {e}") from e
                        time.sleep(5 * (attempt + 1))
            dur = _get_duration(path)
            return AudioSegment(index=i, text=text, path=path, duration=dur)

        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, _gen_one, i, line) for i, line in enumerate(lines)]
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda a: a.index)

    async def _plan_visuals(
        self, audio: list[AudioSegment], title: str, plan_path: str, config: VideoConfig,
    ) -> list[Visual]:
        """Plan one visual per narration line. Uses cached plan if available."""

        # Check cache
        if os.path.exists(plan_path):
            try:
                with open(plan_path) as f:
                    cached = json.load(f)
                cached_visuals = cached.get("visuals", [])
                if len(cached_visuals) >= len(audio):
                    await self._progress(config, "visual plan loaded from cache", f"{len(audio)} visuals")
                    return [
                        Visual(
                            index=i,
                            type=v.get("type", "video"),
                            prompt=v.get("prompt", ""),
                            video_prompt=v.get("video_prompt", ""),
                            character=v.get("character", ""),
                            label=v.get("label", ""),
                        )
                        for i, v in enumerate(cached_visuals[:len(audio)])
                    ]
            except Exception:
                pass

        # Generate visual plan with LLM
        await self._progress(config, "planning visuals", "calling LLM")

        narr_block = "\n".join(
            f"  Line {a.index} ({a.duration:.1f}s): \"{a.text}\"" for a in audio
        )
        aspect = "16:9 landscape" if config.is_long_form else "9:16 vertical portrait"

        system = f"""You plan visuals for narrated videos. Channel: "{config.channel_name}" ({config.niche}).

One visual per narration line. Every image prompt starts with "{config.art_style}"

TYPES:
- "video": animated video clip (DEFAULT). Image is generated then animated.
  Include "video_prompt" describing the motion/animation.
- "image": static still. ONLY for charts, numbers, graphs, text displays.
- "diagram": infographic/data visualization. Uses a text-capable image generator.

Aspect ratio: {aspect}

OUTPUT — JSON:
{{
  "visuals": [
    {{"type": "video", "prompt": "{config.art_style} A visual description", "video_prompt": "camera slowly pulls back"}},
    {{"type": "image", "prompt": "{config.art_style} A chart showing data"}},
    ...
  ]
}}

Return exactly {len(audio)} visuals. Return ONLY valid JSON."""

        user = f"""Plan visuals for "{title}"

NARRATION:
{narr_block}

Total: {sum(a.duration for a in audio):.1f}s, {len(audio)} lines"""

        resp = self.llm.generate(prompt=user, system=system, max_tokens=4000)
        resp = resp.strip()
        if resp.startswith("```"):
            resp = re.sub(r"^```(?:json)?\s*", "", resp)
            resp = re.sub(r"\s*```$", "", resp)

        plan = json.loads(resp)
        raw_visuals = plan.get("visuals", [])

        # Save cache
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2)

        # Parse into Visual objects, pad if needed
        visuals = []
        for i in range(len(audio)):
            if i < len(raw_visuals):
                v = raw_visuals[i]
                visuals.append(Visual(
                    index=i, type=v.get("type", "video"),
                    prompt=v.get("prompt", ""), video_prompt=v.get("video_prompt", ""),
                    character=v.get("character", ""), label=v.get("label", ""),
                ))
            else:
                visuals.append(Visual(index=i, type="image", prompt=f"{config.art_style} Abstract scene"))

        n_vid = sum(1 for v in visuals if v.type == "video")
        n_img = sum(1 for v in visuals if v.type == "image")
        n_dia = sum(1 for v in visuals if v.type == "diagram")
        await self._progress(config, "visual plan done", f"{n_vid} videos, {n_img} images, {n_dia} diagrams")
        return visuals

    async def _generate_visuals(
        self, visuals: list[Visual], audio: list[AudioSegment],
        images_dir: str, clips_dir: str, config: VideoConfig,
    ) -> dict[int, VisualAsset]:
        """Generate all visual assets — images, diagrams, and video clips."""
        import base64

        assets: dict[int, VisualAsset] = {}

        image_visuals = [v for v in visuals if v.type == "image"]
        diagram_visuals = [v for v in visuals if v.type == "diagram"]
        video_visuals = [v for v in visuals if v.type == "video"]

        parts = []
        if image_visuals: parts.append(f"{len(image_visuals)} images")
        if diagram_visuals: parts.append(f"{len(diagram_visuals)} diagrams")
        if video_visuals: parts.append(f"{len(video_visuals)} video clips")
        await self._progress(config, "generating visuals", ", ".join(parts) if parts else "none")

        # Choose image provider based on format
        img_gen = self.portrait_image_provider if config.is_portrait else self.image_provider
        img_size = "1024x1536" if config.is_portrait else "1536x1024"

        # --- Still images ---
        if image_visuals:
            def _gen_img(v: Visual) -> VisualAsset:
                path = os.path.join(images_dir, f"line_{v.index}.png")
                if not os.path.exists(path):
                    img_gen.generate_image(prompt=v.prompt, output_path=path, size=img_size)
                return VisualAsset(index=v.index, type="image", path=path, source="image")

            loop = asyncio.get_event_loop()
            results = await asyncio.gather(*[
                loop.run_in_executor(None, _gen_img, v) for v in image_visuals
            ])
            for asset in results:
                assets[asset.index] = asset

        # --- Diagrams (always OpenAI for text rendering) ---
        if diagram_visuals:
            def _gen_diagram(v: Visual) -> VisualAsset:
                path = os.path.join(images_dir, f"line_{v.index}.png")
                if not os.path.exists(path):
                    self.portrait_image_provider.generate_image(
                        prompt=v.prompt, output_path=path,
                        size=img_size, quality="high",
                    )
                return VisualAsset(index=v.index, type="image", path=path, source="diagram")

            loop = asyncio.get_event_loop()
            results = await asyncio.gather(*[
                loop.run_in_executor(None, _gen_diagram, v) for v in diagram_visuals
            ])
            for asset in results:
                assets[asset.index] = asset

        # --- Video clips (image → animate) ---
        if video_visuals:
            async def _gen_video(v: Visual) -> VisualAsset:
                clip_path = os.path.join(clips_dir, f"line_{v.index}.mp4")
                if os.path.exists(clip_path):
                    await self._log(config, f"video clip line {v.index} [cached]")
                    return VisualAsset(index=v.index, type="video", path=clip_path, source="video")

                dur = max(1, min(int(audio[v.index].duration) + 1, 10))
                await self._log(config, f"video clip line {v.index} — generating image")

                # Step 1: Generate image
                img_path = os.path.join(images_dir, f"line_{v.index}.png")
                if not os.path.exists(img_path):
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, lambda: self.image_provider.generate_image(prompt=v.prompt, output_path=img_path)
                    )

                # Compress for video API
                img_compressed = img_path.replace(".png", "_hq.jpg")
                if not os.path.exists(img_compressed):
                    subprocess.run(["ffmpeg", "-y", "-i", img_path, "-q:v", "2", img_compressed],
                                   capture_output=True, timeout=10)
                source_img = img_compressed if os.path.exists(img_compressed) else img_path
                with open(source_img, "rb") as rf:
                    ext = "jpeg" if source_img.endswith(".jpg") else "png"
                    img_data_url = f"data:image/{ext};base64,{base64.b64encode(rf.read()).decode()}"

                # Step 2: Animate
                motion = v.video_prompt or "Slow cinematic movement"
                await self._log(config, f"video clip line {v.index} — animating ({dur}s)")

                async def _on_progress(progress, elapsed):
                    await self._log(config, f"video clip line {v.index} — {progress}% ({elapsed}s)")

                for attempt in range(3):
                    try:
                        if attempt > 0:
                            await asyncio.sleep(5 * attempt)
                            await self._log(config, f"video clip line {v.index} — retry {attempt}")
                        await self.video_provider.generate_video(
                            prompt=motion, output_path=clip_path, duration=dur,
                            aspect_ratio=config.aspect_ratio, image_url=img_data_url,
                            progress_callback=_on_progress,
                        )
                        await self._log(config, f"video clip line {v.index} — done")
                        return VisualAsset(index=v.index, type="video", path=clip_path, source="video")
                    except Exception as e:
                        if attempt == 2:
                            await self._log(config, f"video clip line {v.index} — FAILED: {str(e)[:100]}")
                            raise
                        await self._log(config, f"video clip line {v.index} — attempt {attempt} failed")

            # Launch all video clips with minimal stagger
            tasks = []
            for idx, v in enumerate(video_visuals):
                async def _launch(v=v, delay=idx * 0.3):
                    await asyncio.sleep(delay)
                    return await _gen_video(v)
                tasks.append(_launch())

            results = await asyncio.gather(*tasks)
            for asset in results:
                assets[asset.index] = asset

        await self._progress(config, "visuals done", f"{len(assets)} assets generated")
        return assets

    async def _create_segments(
        self, audio: list[AudioSegment], assets: dict[int, VisualAsset],
        segments_dir: str, config: VideoConfig,
    ) -> list[Segment]:
        """Pair each audio line with its visual and render segments."""
        await self._progress(config, "creating segments", f"{len(audio)} segments")

        if config.is_long_form:
            scale = f"scale={config.width}:{config.height}:force_original_aspect_ratio=decrease,pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2"
        else:
            scale = f"scale={config.width}:{config.height}:force_original_aspect_ratio=increase,crop={config.width}:{config.height}"

        def _render_segment(seg_audio: AudioSegment) -> Optional[Segment]:
            seg_path = os.path.join(segments_dir, f"seg_{seg_audio.index}.mp4")
            if os.path.exists(seg_path):
                return Segment(index=seg_audio.index, path=seg_path, duration=seg_audio.duration)

            asset = assets.get(seg_audio.index)
            if not asset:
                return None

            if asset.type == "video":
                # Strip audio from video clip, use narration audio instead
                video_only = asset.path.replace(".mp4", "_vidonly.mp4")
                if not os.path.exists(video_only):
                    subprocess.run([
                        "ffmpeg", "-y", "-i", asset.path,
                        "-map", "0:v:0", "-c:v", "copy", "-an", video_only,
                    ], capture_output=True, timeout=30)
                src = video_only if os.path.exists(video_only) else asset.path
                cmd = [
                    "ffmpeg", "-y",
                    "-stream_loop", "-1", "-i", src,
                    "-i", seg_audio.path,
                    "-vf", scale,
                    "-map", "0:v", "-map", "1:a",
                    "-t", str(seg_audio.duration),
                    "-r", "30", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "14",
                    "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
                    "-movflags", "+faststart", seg_path,
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", asset.path,
                    "-i", seg_audio.path,
                    "-vf", scale,
                    "-r", "30", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "14",
                    "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", seg_path,
                ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                return None
            if os.path.exists(seg_path):
                return Segment(index=seg_audio.index, path=seg_path, duration=seg_audio.duration)
            return None

        loop = asyncio.get_event_loop()
        results = await asyncio.gather(*[
            loop.run_in_executor(None, _render_segment, a) for a in audio
        ])

        segments = sorted([s for s in results if s], key=lambda s: s.index)
        await self._progress(config, "segments done", f"{len(segments)} of {len(audio)}")
        return segments

    async def _assemble(self, segments: list[Segment], output_path: str, config: VideoConfig):
        """Concatenate all segments into one video."""
        if os.path.exists(output_path):
            return

        await self._progress(config, "assembling", f"{len(segments)} segments")

        concat_list = output_path.replace(".mp4", "_list.txt")
        with open(concat_list, "w") as f:
            for seg in segments:
                f.write(f"file '{os.path.abspath(seg.path)}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "44100", "-b:a", "192k", "-movflags", "+faststart",
            output_path,
        ], capture_output=True, timeout=300)

    async def _burn_subtitles(
        self, concat_path: str, final_path: str, audio: list[AudioSegment],
        visuals: list[Visual], config: VideoConfig,
    ) -> Optional[str]:
        """Transcribe audio with Whisper and burn karaoke subtitles."""
        if config.subtitle_style == "none":
            import shutil
            shutil.copy2(concat_path, final_path)
            return None

        await self._progress(config, "adding subtitles", "transcribing with Whisper")

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segs, _ = model.transcribe(concat_path, word_timestamps=True)

            all_words = []
            for seg in segs:
                if seg.words:
                    for w in seg.words:
                        all_words.append((w.word.strip(), w.start, w.end))

            ass_path = os.path.join(config.output_dir, "subs.ass")
            _write_karaoke_ass(
                ass_path, all_words,
                width=config.width, height=config.height,
                is_long_form=config.is_long_form,
            )

            result = subprocess.run([
                "ffmpeg", "-y", "-i", concat_path,
                "-vf", f"ass='{ass_path}'",
                "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-movflags", "+faststart", final_path,
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                import shutil
                shutil.copy2(concat_path, final_path)
                return None

            return ass_path
        except Exception as e:
            logger.warning("subtitle step failed", error=str(e)[:200])
            import shutil
            shutil.copy2(concat_path, final_path)
            return None

    # --- Helpers ---

    async def _progress(self, config: VideoConfig, step: str, detail: str = ""):
        """Report progress to callbacks."""
        msg = f"{step} — {detail}" if detail else step
        if config.on_progress:
            try:
                result = config.on_progress(step, detail)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    async def _log(self, config: VideoConfig, msg: str):
        """Log a detail line."""
        if config.on_log:
            try:
                result = config.on_log(msg)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass


# --- Utility functions ---

def _get_duration(path: str) -> float:
    """Get audio/video duration via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, timeout=10,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _write_karaoke_ass(path: str, words: list, width: int = 720, height: int = 1280,
                        is_long_form: bool = False, labels: list = None):
    """Write an ASS subtitle file with word-by-word karaoke highlighting."""
    font_size = 36 if is_long_form else 52
    margin_v = 200 if is_long_form else 350
    words_per_group = 4 if is_long_form else 3

    with open(path, "w") as f:
        f.write(f"""[Script Info]
Title: Karaoke Subtitles
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,Impact,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        # Group words and create karaoke effect
        for i in range(0, len(words), words_per_group):
            group = words[i:i + words_per_group]
            if not group:
                continue

            for highlight_idx in range(len(group)):
                start_time = group[highlight_idx][1]
                end_time = group[highlight_idx][2] if highlight_idx < len(group) - 1 else group[-1][2]

                parts = []
                for j, (word, _, _) in enumerate(group):
                    if j == highlight_idx:
                        parts.append(f"{{\\1c&H00FFFF&}}{word}")
                    else:
                        parts.append(f"{{\\1c&HFFFFFF&}}{word}")

                text = " ".join(parts)
                s = f"{int(start_time // 3600)}:{int((start_time % 3600) // 60):02d}:{start_time % 60:05.2f}"
                e = f"{int(end_time // 3600)}:{int((end_time % 3600) // 60):02d}:{end_time % 60:05.2f}"
                f.write(f"Dialogue: 1,{s},{e},Word,,0,0,0,,{text}\n")
