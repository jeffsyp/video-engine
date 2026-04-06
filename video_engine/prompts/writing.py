"""Prompts for the writing phase — outline, script, critique, revision."""


def build_outline_prompt(idea: dict, niche: str) -> tuple[str, str]:
    """Return (system, user) prompts for building a video outline."""
    system = (
        "You are a YouTube scriptwriter who creates detailed video outlines. "
        "Your outlines are structured for maximum retention — strong hook, clear narrative, satisfying conclusion."
    )
    user = f"""Create a detailed outline for this YouTube video in the "{niche}" niche.

VIDEO IDEA:
- Title: {idea['title']}
- Hook: {idea['hook']}
- Angle: {idea['angle']}
- Target length: {idea['target_length_seconds']} seconds

Provide:
1. idea_title: The video title
2. sections: A list of 5-8 sections (each is a brief description of what that section covers)
3. estimated_duration_seconds: Target duration
4. key_points: 3-5 key takeaways the viewer should leave with

Return EXACTLY this JSON format (no markdown, no extra text):
{{
  "idea_title": "...",
  "sections": ["Hook: ...", "Section 2: ...", ...],
  "estimated_duration_seconds": {idea['target_length_seconds']},
  "key_points": ["point1", "point2", ...]
}}"""
    return system, user


def write_script_prompt(outline: dict, niche: str, tone: str) -> tuple[str, str]:
    """Return (system, user) prompts for writing a full script."""
    sections_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(outline["sections"]))
    key_points_text = "\n".join(f"  - {p}" for p in outline["key_points"])

    system = (
        f"You are a top YouTube scriptwriter who writes for voice performance. Your tone is: {tone}. "
        "You write scripts that sound like a real person talking — not reading. "
        "You use rhythm, pacing, and word choice to control how the narrator sounds. "
        "You know that the script IS the performance — every pause, every short sentence, "
        "every rhetorical question shapes how the AI voice delivers it."
    )
    user = f"""Write a complete YouTube video script based on this outline.

TITLE: {outline['idea_title']}
NICHE: {niche}
TARGET DURATION: {outline['estimated_duration_seconds']} seconds (~{outline['estimated_duration_seconds'] * 150 // 60} words at 150 wpm)

OUTLINE:
{sections_text}

KEY POINTS TO COVER:
{key_points_text}

VOICE PERFORMANCE RULES (critical — this will be read by an AI voice):
- Write exactly as it should be SPOKEN, not read
- Use punctuation to control pacing:
  * Short sentences = urgency and impact. Like this. Three words.
  * Ellipses (...) = dramatic pause, building tension
  * Dashes — = quick aside or pivot
  * Question marks = natural vocal uptick, engages the listener
- Break up big numbers for emphasis: "Three. Point. Two. Billion dollars."
- Add conversational asides: "And honestly?" or "Here's the thing..." or "Think about that for a second."
- Vary sentence length constantly — long flowing explanation, then BAM. Short hit.
- Use humor where natural — dry wit, unexpected comparisons, understatement
- Start sections with a pattern interrupt — something that resets attention
- Never use filler transitions like "Moving on..." or "Next up..." or "Let's talk about..."
- Instead, transition with hooks: "But that's not even the scary part." or "Now... this is where it gets interesting."

SCRIPT RULES:
- Write the FULL script, word for word as it would be narrated
- Start with a strong hook in the first 5 seconds
- Include a clear call-to-action near the end
- Aim for ~{outline['estimated_duration_seconds'] * 150 // 60} words
- Do NOT include stage directions, visual cues, or [brackets]
- No emojis

Return ONLY the script text, no JSON wrapping."""
    return system, user


def critique_script_prompt(script_content: str, idea_title: str) -> tuple[str, str]:
    """Return (system, user) prompts for critiquing a script."""
    system = (
        "You are a ruthless but constructive YouTube script editor who specializes in voice-first content. "
        "You identify weak points that would cause viewers to click away. "
        "You evaluate both the content AND how it will sound when read aloud by an AI voice. "
        "You are specific, actionable, and honest."
    )
    user = f"""Critique this YouTube script. It will be read by an AI text-to-speech voice.

TITLE: {idea_title}

SCRIPT:
{script_content}

Evaluate:

STRENGTHS:
- (2-3 things that work well)

CONTENT WEAKNESSES:
- (3-5 specific content issues)

VOICE PERFORMANCE ISSUES:
- (Does the pacing vary enough? Are there monotonous sections?)
- (Are sentences too uniform in length? Does it sound robotic when read aloud?)
- (Are there natural pauses, questions, and conversational moments?)
- (Would a listener zone out during any section? Where?)

RETENTION RISKS:
- (Moments where viewers would click away and why)

HUMOR/PERSONALITY CHECK:
- (Does it have any personality? Or does it sound like a Wikipedia article?)
- (Are there moments of wit, surprise, or human connection?)

PRIORITY FIXES:
- (Top 3 changes that would most improve this script for voice delivery)

Return ONLY the critique text, no JSON wrapping."""
    return system, user


def revise_script_prompt(
    script_content: str, critique: str, idea_title: str, tone: str
) -> tuple[str, str]:
    """Return (system, user) prompts for revising a script based on critique."""
    system = (
        f"You are a top YouTube scriptwriter. Your tone is: {tone}. "
        "You are revising a script based on editorial feedback. "
        "Address every critique point while maintaining the script's voice and flow."
    )
    user = f"""Revise this YouTube script based on the critique below. Address every weakness and priority fix.

TITLE: {idea_title}

ORIGINAL SCRIPT:
{script_content}

CRITIQUE:
{critique}

REQUIREMENTS:
- Fix every issue raised in the critique
- Maintain the original tone and voice
- Keep or improve the hook
- Ensure the script flows naturally when read aloud
- Do NOT include stage directions, visual cues, or [brackets]

Return ONLY the revised script text, no JSON wrapping or commentary."""
    return system, user


def build_short_script_from_topic(
    channel_name: str,
    niche: str,
    voice_id: str,
    channel_id: int,
    title: str,
    brief: str,
    key_facts: str = "",
) -> tuple[str, str]:
    """Write a 20-30 second short script from a topic."""

    system = f"""You are a scriptwriter for "{channel_name}" — an educational channel about {niche}.

Write a 20-30 second narration that answers the topic's question quickly and clearly. The viewer should feel smarter after watching.

RULES:
- 5-8 narration lines. Under 30 seconds. Every word earns its place.
- Line 1: Ask the question directly to the viewer.
- Build from simple to surprising. Start with what they know, reveal what they don't.
- Use analogies to make complex things click.
- End with the "aha" moment.
- Be accurate. Don't oversimplify to the point of being wrong.
- Conversational — like explaining to a curious friend.
- No jargon without immediately explaining it.
- No emojis.

OUTPUT — return a JSON object:
{{
  "title": "{title}",
  "narration": ["Line 1 — the hook", "Line 2", "...", "Final line — the aha"],
  "caption": "YouTube description + 5-8 hashtags",
  "tags": ["specific_tag", "broader_tag", "education", "shorts"],
  "voice_id": "{voice_id}",
  "channel_id": {channel_id},
  "format_version": 2
}}

Return ONLY valid JSON, no markdown."""

    user = f"""Write a 20-30 second short script for:

TITLE: {title}
BRIEF: {brief}
KEY FACTS: {key_facts}

Teach ONE thing clearly. End with the moment that makes it click."""

    return system, user


def build_midform_script_from_topic(
    channel_name: str,
    niche: str,
    voice_id: str,
    channel_id: int,
    title: str,
    brief: str,
    key_facts: str = "",
) -> tuple[str, str]:
    """Write a 3-5 minute mid-form explainer script from a topic."""

    system = f"""You are a scriptwriter for "{channel_name}" — an educational channel about {niche}.

Write a 3-5 minute narration that fully explains the topic. Build understanding step by step so the viewer truly GETS IT.

RULES:
- 25-40 narration lines. Each line = one visual on screen.
- Line 1: Hook — ask the question or state something surprising.
- Build understanding progressively — each line adds a new piece.
- Include at least one "wait, really?" moment that reframes what the viewer thought they knew.
- Use analogies and comparisons to make abstract concepts concrete.
- Include specific numbers, names, and real details.
- End with a satisfying conclusion that ties everything together.
- Conversational tone — like explaining to a smart friend over coffee.
- No jargon without explaining it. No filler. Every line must add value.
- No emojis.

OUTPUT — return a JSON object:
{{
  "title": "{title}",
  "narration": ["Line 1 — the hook", "Line 2", "...", "Final line — the conclusion"],
  "caption": "YouTube description + 5-8 hashtags",
  "tags": ["specific_tag", "broader_tag", "education"],
  "voice_id": "{voice_id}",
  "channel_id": {channel_id},
  "format_version": 2,
  "long_form": true
}}

Return ONLY valid JSON, no markdown."""

    user = f"""Write a 3-5 minute explainer script for:

TITLE: {title}
BRIEF: {brief}
KEY FACTS: {key_facts}

Build understanding step by step. Make the viewer truly GET IT by the end."""

    return system, user
