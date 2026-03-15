from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Filler words/phrases to remove (case-insensitive)
FILLER_WORDS = {
    "um", "uh", "er", "ah", "hmm", "mm",
}
FILLER_PHRASES = [
    "you know", "i mean", "sort of", "kind of",
]
# "like" only when it appears as filler (not in "I like", "looks like", etc.)
# We handle it as a standalone filler if surrounded by commas or at utterance start/end.

# Known Whisper hallucinations
HALLUCINATION_PATTERNS = [
    r"(?i)thank you for watching",
    r"(?i)thanks for watching",
    r"(?i)please subscribe",
    r"(?i)subscribe to",
    r"(?i)like and subscribe",
    r"(?i)see you in the next",
    r"(?i)don'?t forget to subscribe",
    r"(?i)please like",
    r"(?i)if you enjoyed this",
]

# Consecutive repeat threshold
REPEAT_THRESHOLD = 3


def post_process(result: dict, audio_duration: float | None = None) -> dict:
    """Post-process a transcription result from the model service.

    Modifies the result dict in place and returns it.
    Steps:
      1. Hallucination detection and filtering
      2. Filler word removal
      3. Punctuation / whitespace cleanup
    """
    text = result.get("text", "")
    words = result.get("words", [])

    if not text.strip():
        return result

    # 1. Hallucination detection
    if _is_hallucination(text, words, audio_duration):
        logger.info("Filtered hallucination: %r", text)
        result["text"] = ""
        result["words"] = []
        return result

    # 2. Remove filler words from the word list
    filtered_words = _remove_fillers(words)

    # 3. Rebuild text from remaining words
    text = " ".join(w["word"] for w in filtered_words)

    # 4. Cleanup
    text = _cleanup_text(text)

    result["text"] = text
    result["words"] = filtered_words
    return result


def _is_hallucination(
    text: str, words: list[dict], audio_duration: float | None
) -> bool:
    # Check known hallucination patterns
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, text):
            return True

    # Check for repeated phrases (3+ consecutive identical words)
    if words and len(words) >= REPEAT_THRESHOLD:
        for i in range(len(words) - REPEAT_THRESHOLD + 1):
            chunk = [w["word"].lower().strip() for w in words[i : i + REPEAT_THRESHOLD]]
            if len(set(chunk)) == 1:
                return True

    # Anomalous text length vs audio duration
    if audio_duration and audio_duration > 0:
        word_count = len(words) if words else len(text.split())
        words_per_second = word_count / audio_duration
        # Normal speech: 2-4 words/sec. Hallucinations can produce 10+ w/s
        if words_per_second > 8.0:
            logger.debug(
                "Suspicious word rate: %.1f words/sec (duration=%.1fs, words=%d)",
                words_per_second,
                audio_duration,
                word_count,
            )
            return True

    return False


def _remove_fillers(words: list[dict]) -> list[dict]:
    """Remove filler words from the word list."""
    filtered = []
    i = 0
    while i < len(words):
        word_lower = words[i]["word"].lower().strip().strip(".,!?")

        # Check multi-word filler phrases
        matched_phrase = False
        for phrase in FILLER_PHRASES:
            phrase_words = phrase.split()
            phrase_len = len(phrase_words)
            if i + phrase_len <= len(words):
                candidate = [
                    words[i + j]["word"].lower().strip().strip(".,!?")
                    for j in range(phrase_len)
                ]
                if candidate == phrase_words:
                    i += phrase_len
                    matched_phrase = True
                    break

        if matched_phrase:
            continue

        # Check single-word fillers
        if word_lower in FILLER_WORDS:
            i += 1
            continue

        filtered.append(words[i])
        i += 1

    return filtered


def _cleanup_text(text: str) -> str:
    """Fix whitespace and capitalization after filler removal."""
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    if not text:
        return text

    # Re-capitalize after sentence-ending punctuation
    def _cap_after_period(m: re.Match) -> str:
        return m.group(1) + " " + m.group(2).upper()

    text = re.sub(r"([.!?])\s+([a-z])", _cap_after_period, text)

    # Capitalize first letter
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    return text
