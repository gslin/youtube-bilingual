#!/usr/bin/env python3
"""Download a YouTube video and mux bilingual (original + zh-Hant) subtitles.

Pipeline:
  1. yt-dlp downloads the video
  2. ffmpeg extracts compressed audio
  3. OpenAI ASR (whisper-1) transcribes with segment timestamps
  4. An OpenAI text model translates each cue into Traditional Chinese
  5. ffmpeg muxes an ASS subtitle track into an MKV

Timestamped captions require whisper-1 (or gpt-4o-transcribe-diarize).
gpt-transcribe / gpt-4o-transcribe do not return timestamps.

Requires:
  - ffmpeg / ffprobe
  - yt-dlp
  - OPENAI_API_KEY in .env (or the environment)
  - pip install -r requirements.txt
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

MAX_ASR_BYTES = 24 * 1024 * 1024
DEFAULT_CHUNK_SECONDS = 10 * 60
DEFAULT_ASR_MODEL = "whisper-1"
DEFAULT_TRANSLATE_MODEL = "gpt-4.1-mini"
TIMESTAMP_ASR_MODELS = ("whisper-1", "gpt-4o-transcribe-diarize")
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
class Cue:
    id: int
    start: float
    end: float
    original: str
    zh_hant: str = ""


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
) -> tuple[Path, str]:
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
    title = lines[-1]
    if not video_path.is_file():
        candidates = sorted(work.glob("source.*"))
        if not candidates:
            raise SystemExit("yt-dlp finished but the video file was not found")
        video_path = candidates[0]
    return video_path, title


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
) -> list[Cue]:
    cues: list[Cue] = []
    prompt = ""
    for index, (chunk, offset) in enumerate(chunks, start=1):
        log(f"Transcribing chunk {index}/{len(chunks)}: {chunk.name} (offset {offset:.2f}s)")
        cues.extend(
            transcribe_file(
                client,
                chunk,
                asr_model=asr_model,
                language=language,
                offset=offset,
                prompt=prompt,
            )
        )
        if cues:
            prompt = " ".join(cue.original for cue in cues[-8:])[-220:]
    numbered = []
    for i, cue in enumerate(cues):
        numbered.append(Cue(id=i, start=cue.start, end=cue.end, original=cue.original))
    return numbered


def transcribe_file(
    client: Any,
    path: Path,
    *,
    asr_model: str,
    language: str,
    offset: float,
    prompt: str,
) -> list[Cue]:
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
                kwargs["timestamp_granularities"] = ["segment"]
                if prompt:
                    kwargs["prompt"] = prompt
            return client.audio.transcriptions.create(**kwargs)

    result = openai_call(_create)
    segments = field(result, "segments") or []
    cues: list[Cue] = []
    for seg in segments:
        text = str(field(seg, "text") or "").strip()
        if not text:
            continue
        start = float(field(seg, "start") or 0.0) + offset
        end = float(field(seg, "end") or start) + offset
        if end <= start:
            end = start + 0.5
        cues.append(Cue(id=len(cues), start=start, end=end, original=text))
    return cues


def translate_cues(
    client: Any,
    cues: list[Cue],
    *,
    model: str,
    language: str,
    batch_size: int,
) -> None:
    from pydantic import BaseModel

    class TranslatedCue(BaseModel):
        id: int
        original: str
        zh_hant: str

    class TranslationBatch(BaseModel):
        cues: list[TranslatedCue]

    instructions = (
        "You convert subtitle cues into bilingual captions.\n"
        "For each cue:\n"
        "- original: keep the spoken wording. Only fix obvious ASR typos, spacing, "
        "and punctuation. Do not paraphrase.\n"
        "- zh_hant: natural Traditional Chinese used in Taiwan (zh-Hant-TW).\n"
        "Rules:\n"
        "- Return the same cue ids, same count, same order.\n"
        "- Do not merge or split cues.\n"
        "- Do not add notes, brackets, or speaker labels unless they are in the source.\n"
        "- Keep well-known names, brands, and code in the original script when that is natural.\n"
        "- Use Traditional Chinese punctuation for zh_hant.\n"
        f"- Source spoken language code: {language}."
    )

    for start in range(0, len(cues), batch_size):
        batch = cues[start : start + batch_size]
        payload = {
            "cues": [{"id": cue.id, "text": cue.original} for cue in batch],
        }
        log(f"Translating cues {batch[0].id + 1}-{batch[-1].id + 1} / {len(cues)}")

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
            raise SystemExit("Translation model returned no structured output")
        by_id = {item.id: item for item in parsed.cues}
        missing = [cue.id for cue in batch if cue.id not in by_id]
        if missing:
            raise SystemExit(f"Translation response missing cue ids: {missing[:10]}")
        for cue in batch:
            item = by_id[cue.id]
            cue.original = (item.original or cue.original).strip()
            cue.zh_hant = item.zh_hant.strip()


def build_ass(cues: Iterable[Cue], title: str) -> str:
    lines = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
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
        "0,0,0,0,100,100,0,0,1,2,0,2,60,60,110,1",
        "Style: Chinese,Arial,56,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2.4,0,2,60,60,40,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in cues:
        start = to_ass_time(cue.start)
        end = to_ass_time(cue.end)
        original = ass_escape(cue.original)
        zh = ass_escape(cue.zh_hant)
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

    cwd_env = Path.cwd() / ".env"
    script_env = Path(__file__).resolve().parent / ".env"
    load_dotenv(cwd_env)
    if script_env.resolve() != cwd_env.resolve():
        load_dotenv(script_env)


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
    parser.add_argument("--batch-size", type=int, default=40, help="Cues per translation request")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Audio chunk length when the file exceeds the 25 MB ASR limit",
    )
    parser.add_argument("--work-dir", type=Path, help="Keep intermediate files in this directory")
    parser.add_argument("--keep-work", action="store_true", help="Do not delete the work directory")
    parser.add_argument("--cookies", type=Path, help="Netscape cookies.txt for yt-dlp")
    parser.add_argument(
        "--cookies-from-browser",
        help="Pass through to yt-dlp, e.g. chrome or firefox",
    )
    parser.add_argument("--self-test", action="store_true", help="Run local helper tests and exit")
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

    which_or_exit("yt-dlp")
    which_or_exit("ffmpeg")
    which_or_exit("ffprobe")
    language = normalize_language(args.language)
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
        video_path, title = download_video(
            args.url,
            work,
            cookies=args.cookies,
            cookies_from_browser=args.cookies_from_browser,
        )
        log(f"Downloaded: {title}")
        audio_path = work / "audio.m4a"
        extract_audio(video_path, audio_path)
        chunks = split_audio(audio_path, work / "chunks", args.chunk_seconds)
        cues = transcribe_chunks(
            client,
            chunks,
            asr_model=args.asr_model,
            language=language,
        )
        if not cues:
            raise SystemExit("ASR returned no subtitle cues")
        write_json(work / "transcript.json", [asdict(cue) for cue in cues])
        translate_cues(
            client,
            cues,
            model=args.model,
            language=language,
            batch_size=args.batch_size,
        )
        write_json(work / "bilingual.json", [asdict(cue) for cue in cues])
        ass_text = build_ass(cues, title)
        ass_path = work / "bilingual.ass"
        ass_path.write_text(ass_text, encoding="utf-8")

        output = args.output
        if output is None:
            output = Path(f"{sanitize_filename(title)}.mkv")
        if output.suffix.lower() != ".mkv":
            output = output.with_suffix(".mkv")
        mux_mkv(video_path, ass_path, output)
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
