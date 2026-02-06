# prompts.py
# ----------
# Builds a clear, consistent prompt for summarization.
# This version is ASCII-only to avoid hidden encoding issues.

from typing import Literal

LengthOpt = Literal["short", "medium", "detailed"]
ToneOpt = Literal["neutral", "professional", "casual"]


def build_summarization_prompt(
    doc_text: str,
    length: LengthOpt = "short",
    tone: ToneOpt = "neutral",
    bullet_points: bool = True,
    include_title: bool = True,
    max_chars: int = 12000,
) -> str:
    """
    Construct a structured summarization prompt.
    Truncates very long input for demo performance.
    """

    # Normalize and trim input text
    text = (doc_text or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated for demo...]"

    # Target length guidance (kept as plain ASCII text)
    length_map = {
        "short": "about 120-150 words",
        "medium": "about 200-300 words",
        "detailed": "about 400-600 words",
    }

    # Tone guidance
    tone_map = {
        "neutral": "neutral, objective",
        "professional": "concise, business-professional",
        "casual": "friendly, plain-language",
    }

    # Formatting instructions
    format_instructions = []
    if include_title:
        format_instructions.append(
            "Start with a single-line Title that captures the main topic."
        )
    if bullet_points:
        format_instructions.append(
            "Use bullet points for key takeaways."
        )
    format_instructions.append("Avoid speculation. Preserve the original meaning.")
    format_instructions.append("If the input is not summarizable, say so briefly.")

    # Build the final prompt as a single triple-quoted literal
    prompt = f"""
You are a helpful assistant that produces accurate, faithful summaries.

Goal: Summarize the user's document.
Target length: {length_map.get(length, 'about 150 words')}
Tone: {tone_map.get(tone, 'neutral')}
Formatting:
- {' '.join(format_instructions)}

Document:
{text}
""".strip()

    return prompt
