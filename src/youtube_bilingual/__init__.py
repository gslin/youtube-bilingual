#!/usr/bin/env python3
"""Download a YouTube video and mux bilingual (original + zh-Hant) subtitles.

Pipeline:
  1. yt-dlp downloads the video
  2. ffmpeg extracts compressed audio
  3. OpenAI ASR (whisper-1) transcribes with word timestamps
  4. Local VAD finds the first real speech so intro music is not captioned
  5. An OpenAI text model translates cues and splits them so only one short original line and one Chinese line show at a time
  6. ffmpeg muxes an ASS subtitle track into an MKV

Timestamped captions require whisper-1 (or gpt-4o-transcribe-diarize).
gpt-transcribe / gpt-4o-transcribe do not return timestamps.

Requires:
  - ffmpeg / ffprobe
  - OPENAI_API_KEY in .env (or the environment)
  - uv run youtube-bilingual ...  (or uvx --from . youtube-bilingual ...)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

MAX_ASR_BYTES = 24 * 1024 * 1024
MAX_DESCRIPTION_CHARS = 4000
DEFAULT_CHUNK_SECONDS = 10 * 60
DEFAULT_ASR_MODEL = "whisper-1"
DEFAULT_TRANSLATE_MODEL = "gpt-4.1-mini"
DEFAULT_BATCH_SIZE = 12
DEFAULT_MAX_LINE_CHARS_CJK = 40
DEFAULT_MAX_LINE_CHARS_LATIN = 84
MAX_LINES_PER_CUE = 2
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_MIN_RUN_MS = 270
TIMESTAMP_ASR_MODELS = ("whisper-1", "gpt-4o-transcribe-diarize")
CJK_LANGUAGES = {"zh", "ja", "ko"}
_STRONG_BREAKS = set("。．.！？!?♪…")
_WEAK_BREAKS = set("、，,；;：: ")
LANGUAGE_ALIASES = {
    "jp": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "kr": "ko",
    "kor": "ko",
    "korean": "ko",
    "cn": "zh",
    "chi": "zh",
    "zho": "zh",
    "zh-tw": "zh",
    "zh-cn": "zh",
    "zh-hk": "zh",
    "chinese": "zh",
    "en-us": "en",
    "en-gb": "en",
    "english": "en",
    "fr-fr": "fr",
    "de-de": "de",
    "es-es": "es",
    "pt-br": "pt",
    "pt-pt": "pt",
}

T = TypeVar("T")


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Cue:
    id: int
    start: float
    end: float
    original: str
    zh_hant: str = ""
    words: list[Word] = field(default_factory=list)


@dataclass
class VideoSource:
    path: Path
    title: str
    description: str = ""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def which_or_exit(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Required command not found: {name}")
    return path


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    log("+ " + " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return (name[:120] or "video")


def clip_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n..."


def default_max_line_chars(language: str) -> int:
    return DEFAULT_MAX_LINE_CHARS_CJK if language in CJK_LANGUAGES else DEFAULT_MAX_LINE_CHARS_LATIN


def is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF
        or 0x3400 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF66 <= code <= 0xFF9D
    )


def mainly_cjk(text: str) -> bool:
    letters = [ch for ch in text if not ch.isspace()]
    if not letters:
        return False
    cjk = sum(1 for ch in letters if is_cjk_char(ch))
    return cjk / len(letters) >= 0.3


def split_text(text: str, max_chars: int) -> list[str]:
    text = text.replace("\r\n", " ").replace("\n", " ").strip()
    if not text:
        return []
    if max_chars < 4:
        max_chars = 4
    if len(text) <= max_chars:
        return [text]
    cjk = mainly_cjk(text)
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        end = min(i + max_chars, n)
        if end < n:
            window = text[i:end]
            cut = None
            for idx in range(len(window) - 1, 0, -1):
                if window[idx] in _STRONG_BREAKS:
                    cut = i + idx + 1
                    break
            if cut is None:
                for idx in range(len(window) - 1, 0, -1):
                    ch = window[idx]
                    if ch in _WEAK_BREAKS or (not cjk and ch.isspace()):
                        cut = i + idx + 1
                        break
            if cut is not None:
                end = cut
        part = text[i:end].strip()
        if part:
            parts.append(part)
        if end <= i:
            end = min(i + max_chars, n)
        i = end
    if (
        len(parts) >= 2
        and len(parts[-1]) <= 3
        and parts[-2][-1] not in _STRONG_BREAKS
        and parts[-2][-1] not in _WEAK_BREAKS
    ):
        parts[-2] += parts[-1]
        parts.pop()
    return parts or [text]


def split_long_cues(cues: list[Cue], max_line_chars: int, max_lines: int = MAX_LINES_PER_CUE) -> list[Cue]:
    limit = max_line_chars * max_lines
    out: list[Cue] = []
    for cue in cues:
        parts = split_text(cue.original, limit)
        if len(parts) <= 1:
            out.append(cue)
            continue
        weights = [max(len(part), 1) for part in parts]
        total = sum(weights)
        span = max(cue.end - cue.start, 0.3 * len(parts))
        t = cue.start
        for index, part in enumerate(parts):
            if index == len(parts) - 1:
                end = cue.end
            else:
                end = cue.start + span * (sum(weights[: index + 1]) / total)
            if end <= t:
                end = t + 0.3
            out.append(Cue(id=0, start=t, end=end, original=part))
            t = end
        if out[-1].end < cue.end:
            out[-1].end = cue.end
    return [
        Cue(id=index, start=item.start, end=item.end, original=item.original, zh_hant=item.zh_hant)
        for index, item in enumerate(out)
    ]


def wrap_ass_text(text: str, max_chars: int) -> str:
    return "\\N".join(ass_escape(part) for part in split_text(text, max_chars))


def join_words(words: list[Word]) -> str:
    parts = [item.word.strip() for item in words if item.word.strip()]
    if not parts:
        return ""
    if mainly_cjk("".join(parts)):
        return "".join(parts)
    text = parts[0]
    for part in parts[1:]:
        if part.startswith("'") or part.startswith("’"):
            text += part
        else:
            text += " " + part
    return text


def words_to_cues(words: list[Word], max_line_chars: int, max_lines: int = MAX_LINES_PER_CUE) -> list[Cue]:
    limit = max(max_line_chars * max_lines, 4)
    batch: list[Word] = []
    cues: list[Cue] = []

    def flush() -> None:
        if not batch:
            return
        text = join_words(batch)
        if text:
            start = batch[0].start
            end = max(batch[-1].end, start + 0.4)
            cues.append(
                Cue(id=len(cues), start=start, end=end, original=text, words=list(batch))
            )
        batch.clear()

    for word in words:
        if not word.word.strip():
            continue
        trial = batch + [word]
        if batch and len(join_words(trial)) > limit:
            flush()
        batch.append(word)
        text = join_words(batch)
        if text and text[-1] in _STRONG_BREAKS and len(text) >= max_line_chars:
            flush()
    flush()
    return cues


def assign_pieces_with_words(cue: Cue, pieces: list[tuple[str, str]]) -> list[Cue] | None:
    remaining = list(cue.words)
    if not remaining:
        return None
    out: list[Cue] = []
    for index, (original, zh_hant) in enumerate(pieces):
        if index == len(pieces) - 1:
            chunk = remaining
            remaining = []
        else:
            target = max(len(original), 1)
            chunk = []
            while remaining and len(join_words(chunk)) < target:
                chunk.append(remaining.pop(0))
            if not chunk:
                return None
        if not chunk:
            return None
        start = chunk[0].start
        end = max(chunk[-1].end, start + 0.3)
        out.append(
            Cue(
                id=0,
                start=start,
                end=end,
                original=original or join_words(chunk),
                zh_hant=zh_hant,
                words=chunk,
            )
        )
    return out


def expand_translated_pieces(cue: Cue, pieces: list[tuple[str, str]]) -> list[Cue]:
    cleaned = [(original.strip(), zh.strip()) for original, zh in pieces if original.strip() or zh.strip()]
    if not cleaned:
        return [cue]
    if len(cleaned) == 1:
        original, zh_hant = cleaned[0]
        return [
            Cue(
                id=cue.id,
                start=cue.start,
                end=cue.end,
                original=original or cue.original,
                zh_hant=zh_hant,
                words=list(cue.words),
            )
        ]
    assigned = assign_pieces_with_words(cue, cleaned)
    if assigned:
        return assigned
    weights = [max(len(original), 1) for original, _zh in cleaned]
    total = sum(weights)
    span = max(cue.end - cue.start, 0.35 * len(cleaned))
    t = cue.start
    out: list[Cue] = []
    for index, (original, zh_hant) in enumerate(cleaned):
        if index == len(cleaned) - 1:
            end = cue.end
        else:
            end = cue.start + span * (sum(weights[: index + 1]) / total)
        if end <= t:
            end = t + 0.3
        out.append(
            Cue(
                id=0,
                start=t,
                end=end,
                original=original or cue.original,
                zh_hant=zh_hant,
            )
        )
        t = end
    if out[-1].end < cue.end:
        out[-1].end = cue.end
    return out


def voiced_runs(voiced: list[bool], frame_ms: int, min_run_ms: int = VAD_MIN_RUN_MS) -> list[tuple[float, float]]:
    min_frames = max(1, min_run_ms // frame_ms)
    runs: list[tuple[float, float]] = []
    index = 0
    count = len(voiced)
    while index < count:
        if not voiced[index]:
            index += 1
            continue
        end = index
        while end < count and voiced[end]:
            end += 1
        if end - index >= min_frames:
            runs.append((index * frame_ms / 1000.0, end * frame_ms / 1000.0))
        index = end
    return runs


def pick_speech_onset(runs: list[tuple[float, float]]) -> float:
    if not runs:
        return 0.0
    start, end = runs[0]
    if start <= 0.2 and (end - start) >= 3.0 and len(runs) >= 2:
        gap = runs[1][0] - end
        if gap >= 0.25:
            return runs[1][0]
    return start


def apply_speech_onset(words: list[Word], onset: float, *, min_intro: float = 1.0) -> list[Word]:
    if not words or onset < min_intro:
        return words
    before = [item for item in words if item.start < onset]
    after = [item for item in words if item.start >= onset]
    if not before:
        return words
    if after and after[0].start <= onset + 1.0:
        clamped: list[Word] = []
        for item in after:
            start = max(item.start, onset)
            clamped.append(Word(item.word, start, max(item.end, start + 0.02)))
        return clamped
    delta = onset - before[0].start
    shifted = [Word(item.word, item.start + delta, item.end + delta) for item in before]
    if not after:
        return shifted
    limit = after[0].start
    trimmed: list[Word] = []
    for item in shifted:
        if item.start >= limit:
            continue
        end = min(item.end, limit)
        if end > item.start:
            trimmed.append(Word(item.word, item.start, end))
    return trimmed + after


def read_pcm16_16k(path: Path) -> bytes:
    log(f"+ ffmpeg pcm 16k from {path.name}")
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(VAD_SAMPLE_RATE),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"ffmpeg pcm extract failed: {detail}")
    return result.stdout


def detect_speech_onset(audio_path: Path, *, aggressiveness: int = 3) -> float:
    import webrtcvad

    pcm = read_pcm16_16k(audio_path)
    vad = webrtcvad.Vad(aggressiveness)
    frame_samples = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)
    frame_bytes = frame_samples * 2
    if len(pcm) < frame_bytes:
        return 0.0
    voiced: list[bool] = []
    for index in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        frame = pcm[index : index + frame_bytes]
        try:
            voiced.append(bool(vad.is_speech(frame, VAD_SAMPLE_RATE)))
        except Exception:
            voiced.append(False)
    return pick_speech_onset(voiced_runs(voiced, VAD_FRAME_MS))


def normalize_language(value: str) -> str:
    raw = value.strip().lower().replace("_", "-")
    if raw in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[raw]
    if re.fullmatch(r"[a-z]{2}", raw):
        return raw
    raise SystemExit(
        f"Unsupported language '{value}'. Use an ISO-639-1 code such as en, ja, ko, zh."
    )


def to_ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    h, rem = divmod(total_cs, 3600 * 100)
    m, rem = divmod(rem, 60 * 100)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\r\n", "\\N")
        .replace("\n", "\\N")
    )


def probe_duration(path: Path) -> float:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    text = result.stdout.strip()
    if not text or text == "N/A":
        raise SystemExit(f"Could not read duration: {path}")
    return float(text)


def download_video(
    url: str,
    work: Path,
    *,
    cookies: Path | None,
    cookies_from_browser: str | None,
) -> VideoSource:
    outtmpl = str(work / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mkv",
        "-o",
        outtmpl,
        "--write-info-json",
        "--print",
        "after_move:%(filepath)s",
        "--print",
        "after_move:%(title)s",
        "--no-warnings",
        url,
    ]
    if cookies:
        cmd[1:1] = ["--cookies", str(cookies)]
    if cookies_from_browser:
        cmd[1:1] = ["--cookies-from-browser", cookies_from_browser]
    result = run(cmd)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"yt-dlp did not return filepath/title:\n{result.stdout}")
    video_path = Path(lines[-2])
    printed_title = lines[-1]
    if not video_path.is_file():
        candidates = sorted(
            path for path in work.glob("source.*") if path.suffix.lower() != ".json"
        )
        if not candidates:
            raise SystemExit("yt-dlp finished but the video file was not found")
        video_path = candidates[0]
    info = load_video_info(work)
    title = str(info.get("title") or printed_title).strip() or "video"
    description = str(info.get("description") or "").strip()
    return VideoSource(path=video_path, title=title, description=description)


def load_video_info(work: Path) -> dict[str, Any]:
    files = sorted(work.glob("*.info.json"))
    if not files:
        return {}
    try:
        data = json.loads(files[-1].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def extract_audio(video_path: Path, audio_path: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(audio_path),
        ]
    )


def split_audio(audio_path: Path, chunk_dir: Path, chunk_seconds: int) -> list[tuple[Path, float]]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    size = audio_path.stat().st_size
    if size <= MAX_ASR_BYTES:
        return [(audio_path, 0.0)]

    log(f"Audio is {size} bytes; splitting into {chunk_seconds}s chunks (25 MB ASR limit)")
    pattern = str(chunk_dir / "chunk_%03d.m4a")
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ac",
            "1",
            "-ar",
            "16000",
            pattern,
        ]
    )
    chunks = sorted(chunk_dir.glob("chunk_*.m4a"))
    if not chunks:
        raise SystemExit("ffmpeg produced no audio chunks")
    offset = 0.0
    result: list[tuple[Path, float]] = []
    for chunk in chunks:
        if chunk.stat().st_size > MAX_ASR_BYTES:
            raise SystemExit(f"Chunk still exceeds 25 MB: {chunk}")
        result.append((chunk, offset))
        offset += probe_duration(chunk)
    return result


def field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def openai_call(fn: Callable[[], T], *, retries: int = 5) -> T:
    from openai import APIStatusError, RateLimitError

    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except RateLimitError as exc:
            last_error = exc
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code not in (408, 409, 429, 500, 502, 503, 504):
                raise
        log(f"OpenAI request failed (attempt {attempt + 1}/{retries}): {last_error}")
        if attempt + 1 >= retries:
            break
        time.sleep(delay)
        delay = min(delay * 2, 30)
    assert last_error is not None
    raise last_error


def transcribe_chunks(
    client: Any,
    chunks: list[tuple[Path, float]],
    *,
    asr_model: str,
    language: str,
) -> list[Word]:
    words: list[Word] = []
    prompt = ""
    for index, (chunk, offset) in enumerate(chunks, start=1):
        log(f"Transcribing chunk {index}/{len(chunks)}: {chunk.name} (offset {offset:.2f}s)")
        chunk_words = transcribe_file(
            client,
            chunk,
            asr_model=asr_model,
            language=language,
            offset=offset,
            prompt=prompt,
        )
        words.extend(chunk_words)
        if chunk_words:
            prompt = " ".join(item.word for item in chunk_words[-20:])[-220:]
    return words


def transcribe_file(
    client: Any,
    path: Path,
    *,
    asr_model: str,
    language: str,
    offset: float,
    prompt: str,
) -> list[Word]:
    def _create() -> Any:
        with path.open("rb") as audio_file:
            kwargs: dict[str, Any] = {
                "model": asr_model,
                "file": audio_file,
                "language": language,
            }
            if asr_model == "gpt-4o-transcribe-diarize":
                kwargs["response_format"] = "diarized_json"
                kwargs["chunking_strategy"] = "auto"
            else:
                kwargs["response_format"] = "verbose_json"
                kwargs["timestamp_granularities"] = ["word", "segment"]
                if prompt:
                    kwargs["prompt"] = prompt
            return client.audio.transcriptions.create(**kwargs)

    result = openai_call(_create)
    words_raw = field(result, "words") or []
    words: list[Word] = []
    for item in words_raw:
        token = str(field(item, "word") or "").strip()
        if not token:
            continue
        start = float(field(item, "start") or 0.0) + offset
        end = float(field(item, "end") or start) + offset
        if end <= start:
            end = start + 0.05
        words.append(Word(token, start, end))
    if words:
        return words
    segments = field(result, "segments") or []
    for seg in segments:
        text = str(field(seg, "text") or "").strip()
        if not text:
            continue
        start = float(field(seg, "start") or 0.0) + offset
        end = float(field(seg, "end") or start) + offset
        if end <= start:
            end = start + 0.5
        words.append(Word(text, start, end))
    return words


def translate_cues(
    client: Any,
    cues: list[Cue],
    *,
    model: str,
    language: str,
    batch_size: int,
    title: str,
    description: str,
    max_line_chars: int,
) -> list[Cue]:
    from pydantic import BaseModel, Field

    class SubtitlePiece(BaseModel):
        original: str
        zh_hant: str

    class TranslatedCue(BaseModel):
        id: int
        pieces: list[SubtitlePiece] = Field(min_length=1)

    class TranslationBatch(BaseModel):
        cues: list[TranslatedCue]

    video_meta = {
        "title": title.strip(),
        "description": clip_text(description, MAX_DESCRIPTION_CHARS),
    }
    instructions = (
        "You convert ASR cues into bilingual subtitles that fit on one TV-sized screen.\n"
        "For each input cue, return one or more pieces in speaking order.\n"
        "Each piece is shown by itself: one original line above one Traditional Chinese line.\n"
        f"Hard limits for every piece:\n"
        f"- original: at most {max_line_chars} characters, a single line, no newline.\n"
        f"- zh_hant: at most {max_line_chars} characters, Taiwan Traditional Chinese "
        "(zh-Hant-TW), a single line, no newline.\n"
        "Split at natural phrase boundaries when the source is too long to fit.\n"
        "Cover the whole source text; do not drop spoken words or add extra meaning.\n"
        "Keep original wording except for obvious ASR typos, spacing, and punctuation.\n"
        "If a cue already fits, return exactly one piece.\n"
        "Do not merge different input ids. Return every input id once, same order.\n"
        "Do not add notes, brackets, or speaker labels unless they are in the source.\n"
        "Keep well-known names, brands, and code in the original script when that is natural.\n"
        "The user message includes the YouTube title and description; use them as "
        "terminology context and keep that wording consistent.\n"
        f"Source spoken language code: {language}."
    )

    def request_batch(chunk: list[Cue]) -> dict[int, Any]:
        payload = {
            "video": video_meta,
            "limits": {
                "max_chars_per_line": max_line_chars,
                "lines_per_piece": 1,
            },
            "cues": [{"id": cue.id, "text": cue.original} for cue in chunk],
        }
        log(
            f"Translating cues {chunk[0].id + 1}-{chunk[-1].id + 1} "
            f"({len(chunk)} items) / {len(cues)}"
        )

        def _parse() -> Any:
            return client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text_format=TranslationBatch,
            )

        parsed = openai_call(_parse).output_parsed
        if parsed is None:
            return {}
        return {item.id: item for item in parsed.cues}

    out: list[Cue] = []
    for start in range(0, len(cues), batch_size):
        leftover = cues[start : start + batch_size]
        by_id: dict[int, Any] = {}
        attempt_size = len(leftover)
        while leftover:
            chunk = leftover[:attempt_size]
            got = request_batch(chunk)
            missing = [cue for cue in chunk if cue.id not in got]
            by_id.update({cue.id: got[cue.id] for cue in chunk if cue.id in got})
            unsent = leftover[attempt_size:]
            if missing:
                log(f"Incomplete translation, retrying {len(missing)} cues")
                leftover = missing + unsent
                if attempt_size == 1 and len(chunk) == 1:
                    raise SystemExit(f"Translation response missing cue id {chunk[0].id}")
                attempt_size = 1 if attempt_size <= 2 else max(1, attempt_size // 2)
            else:
                leftover = unsent
                attempt_size = batch_size
        for cue in cues[start : start + batch_size]:
            item = by_id[cue.id]
            pieces = [(piece.original, piece.zh_hant) for piece in item.pieces]
            out.extend(expand_translated_pieces(cue, pieces))
    return [
        Cue(
            id=index,
            start=item.start,
            end=item.end,
            original=item.original,
            zh_hant=item.zh_hant,
            words=list(item.words),
        )
        for index, item in enumerate(out)
    ]


def build_ass(cues: Iterable[Cue], title: str, max_line_chars: int = DEFAULT_MAX_LINE_CHARS_LATIN) -> str:
    lines = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "WrapStyle: 1",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Original,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2,0,2,60,60,190,1",
        "Style: Chinese,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2.4,0,2,60,60,40,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in cues:
        start = to_ass_time(cue.start)
        end = to_ass_time(cue.end)
        original = wrap_ass_text(cue.original, max_line_chars)
        zh = wrap_ass_text(cue.zh_hant, max_line_chars)
        lines.append(
            f"Dialogue: 0,{start},{end},Original,,0,0,0,,{original}"
        )
        if zh:
            lines.append(f"Dialogue: 0,{start},{end},Chinese,,0,0,0,,{zh}")
    lines.append("")
    return "\n".join(lines)


def mux_mkv(video_path: Path, ass_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(ass_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-map",
            "1:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-c:s",
            "ass",
            "-metadata:s:s:0",
            "language=zho",
            "-metadata:s:s:0",
            "title=Original + zh-Hant",
            "-disposition:s:0",
            "default",
            str(output_path),
        ]
    )


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    here = Path(__file__).resolve().parent
    for directory in [here, *here.parents]:
        env_path = directory / ".env"
        if env_path.is_file():
            load_dotenv(env_path)
        if (directory / "pyproject.toml").is_file():
            break


def get_client() -> Any:
    from openai import OpenAI

    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Put it in .env or export it.")
    return OpenAI()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def self_test() -> None:
    assert normalize_language("JP") == "ja"
    assert normalize_language("zh-TW") == "zh"
    assert to_ass_time(0) == "0:00:00.00"
    assert to_ass_time(3661.237) == "1:01:01.24"
    assert to_ass_time(1.005) == "0:00:01.00" or to_ass_time(1.005) == "0:00:01.01"
    assert ass_escape("a{b}\\c\nd") == "a\\{b\\}\\\\c\\Nd"
    cues = [
        Cue(id=0, start=1.0, end=3.5, original="Hello, world.", zh_hant="你好，世界。"),
    ]
    ass = build_ass(cues, "test")
    assert "Dialogue: 0,0:00:01.00,0:00:03.50,Original,,0,0,0,,Hello, world." in ass
    assert "Dialogue: 0,0:00:01.00,0:00:03.50,Chinese,,0,0,0,,你好，世界。" in ass
    assert clip_text("short", 10) == "short"
    assert clip_text("abcdefghij", 8) == "abcdefgh\n..."
    assert split_text("你好。世界。測試", 6) == ["你好。世界。", "測試"]
    assert split_text("Hello there, friend", 12) == ["Hello there,", "friend"]
    long_cue = Cue(
        id=0,
        start=0.0,
        end=4.0,
        original="你好。這是一段比較長的字幕內容需要被切開。",
    )
    split = split_long_cues([long_cue], max_line_chars=8, max_lines=1)
    assert len(split) >= 2
    assert all(len(item.original) <= 11 for item in split)
    wrapped = wrap_ass_text("一二三四五六七八九十一二三四五六七八九十", 8)
    assert wrapped == "一二三四五六七八\\N九十一二三四五六\\N七八九十"
    short_wrap = wrap_ass_text("今日はとてもいい天気なので散歩に行きましょう。", 20)
    assert "\\Nょう。" not in short_wrap
    assert pick_speech_onset([]) == 0.0
    assert pick_speech_onset([(0.0, 8.0), (9.0, 12.0)]) == 9.0
    assert pick_speech_onset([(0.0, 1.0), (2.0, 4.0)]) == 0.0
    shifted = apply_speech_onset(
        [Word("a", 0.0, 0.4), Word("b", 0.4, 0.8), Word("c", 12.0, 12.4)],
        12.0,
    )
    assert [item.word for item in shifted] == ["c"]
    shifted = apply_speech_onset(
        [Word("a", 0.0, 2.0), Word("b", 2.0, 4.0)],
        12.0,
    )
    assert shifted[0].start == 12.0
    packed = words_to_cues(
        [Word("Hello", 1.0, 1.2), Word("there", 1.2, 1.5), Word("friend.", 1.5, 2.0)],
        max_line_chars=20,
        max_lines=1,
    )
    assert packed[0].original == "Hello there friend."
    assert packed[0].start == 1.0
    assert packed[0].end == 2.0
    source = Cue(
        id=0,
        start=1.0,
        end=5.0,
        original="Hello there my friend today",
        words=[
            Word("Hello", 1.0, 1.4),
            Word("there", 1.4, 1.8),
            Word("my", 1.8, 2.2),
            Word("friend", 2.2, 3.0),
            Word("today", 3.0, 4.0),
        ],
    )
    split_cues = expand_translated_pieces(
        source,
        [("Hello there", "你好啊"), ("my friend today", "我的朋友今天")],
    )
    assert len(split_cues) == 2
    assert split_cues[0].zh_hant == "你好啊"
    assert split_cues[0].end <= split_cues[1].start + 1e-6
    print("self-test ok")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a YouTube URL with yt-dlp, transcribe it with OpenAI ASR, "
            "and mux original + Traditional Chinese subtitles into an MKV."
        )
    )
    parser.add_argument("url", nargs="?", help="YouTube URL")
    parser.add_argument(
        "-l",
        "--language",
        help="Spoken language as ISO-639-1, e.g. en, ja, ko, zh",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output MKV path")
    parser.add_argument(
        "--asr-model",
        default=DEFAULT_ASR_MODEL,
        help="OpenAI ASR model with timestamps (whisper-1 or gpt-4o-transcribe-diarize)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TRANSLATE_MODEL,
        help="OpenAI model used to produce Traditional Chinese lines",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Cues per translation request",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Audio chunk length when the file exceeds the 25 MB ASR limit",
    )
    parser.add_argument(
        "--max-line-chars",
        type=int,
        default=0,
        help="Max characters per subtitle line (0 = 40 for ja/zh/ko, 84 otherwise)",
    )
    parser.add_argument("--work-dir", type=Path, help="Keep intermediate files in this directory")
    parser.add_argument("--keep-work", action="store_true", help="Do not delete the work directory")
    parser.add_argument("--cookies", type=Path, help="Netscape cookies.txt for yt-dlp")
    parser.add_argument(
        "--cookies-from-browser",
        help="Pass through to yt-dlp, e.g. chrome or firefox",
    )
    parser.add_argument("--self-test", action="store_true", help="Run local helper tests and exit")
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Do not detect speech onset; keep Whisper timestamps as-is",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    if not args.url or not args.language:
        raise SystemExit("url and --language are required (or pass --self-test)")
    if args.asr_model not in TIMESTAMP_ASR_MODELS:
        raise SystemExit(
            "Subtitles need timestamps. Use --asr-model whisper-1 "
            "(default) or gpt-4o-transcribe-diarize. "
            "gpt-transcribe / gpt-4o-transcribe do not return timestamps."
        )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.chunk_seconds < 30:
        raise SystemExit("--chunk-seconds must be >= 30")
    if args.max_line_chars < 0:
        raise SystemExit("--max-line-chars must be >= 0")

    which_or_exit("yt-dlp")
    which_or_exit("ffmpeg")
    which_or_exit("ffprobe")
    language = normalize_language(args.language)
    max_line_chars = args.max_line_chars or default_max_line_chars(language)
    client = get_client()

    if args.work_dir:
        work = args.work_dir
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix="yt-bilingual-"))
        cleanup = not args.keep_work

    try:
        log(f"Work directory: {work}")
        video = download_video(
            args.url,
            work,
            cookies=args.cookies,
            cookies_from_browser=args.cookies_from_browser,
        )
        log(f"Downloaded: {video.title}")
        audio_path = work / "audio.m4a"
        extract_audio(video.path, audio_path)
        chunks = split_audio(audio_path, work / "chunks", args.chunk_seconds)
        words = transcribe_chunks(
            client,
            chunks,
            asr_model=args.asr_model,
            language=language,
        )
        if not words:
            raise SystemExit("ASR returned no words")
        if not args.no_vad:
            onset = detect_speech_onset(audio_path)
            log(f"Speech onset at {onset:.2f}s")
            words = apply_speech_onset(words, onset)
            if not words:
                raise SystemExit("No words remain after speech-onset alignment")
        cues = words_to_cues(words, max_line_chars)
        before = len(cues)
        cues = split_long_cues(cues, max_line_chars)
        if len(cues) != before:
            log(f"Split {before} ASR cues into {len(cues)} for {max_line_chars} chars/line")
        if not cues:
            raise SystemExit("ASR returned no subtitle cues")
        write_json(work / "transcript.json", [asdict(cue) for cue in cues])
        before = len(cues)
        cues = translate_cues(
            client,
            cues,
            model=args.model,
            language=language,
            batch_size=args.batch_size,
            title=video.title,
            description=video.description,
            max_line_chars=max_line_chars,
        )
        if len(cues) != before:
            log(f"Translation split {before} cues into {len(cues)} screen-sized pieces")
        write_json(work / "bilingual.json", [asdict(cue) for cue in cues])
        ass_text = build_ass(cues, video.title, max_line_chars=max_line_chars)
        ass_path = work / "bilingual.ass"
        ass_path.write_text(ass_text, encoding="utf-8")

        output = args.output
        if output is None:
            output = Path(f"{sanitize_filename(video.title)}.mkv")
        if output.suffix.lower() != ".mkv":
            output = output.with_suffix(".mkv")
        mux_mkv(video.path, ass_path, output)
        sidecar = output.with_suffix(".ass")
        sidecar.write_text(ass_text, encoding="utf-8")
        log(f"Wrote {output}")
        log(f"Wrote {sidecar}")
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)
        elif not args.work_dir:
            log(f"Keeping work directory: {work}")


if __name__ == "__main__":
    main()
