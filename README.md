# youtube-bilingual

Download a YouTube video, transcribe the spoken audio with OpenAI ASR, translate each cue into Traditional Chinese (Taiwan), and mux bilingual subtitles into an `.mkv`.

The subtitle track shows the original line above the Traditional Chinese line.

## Pipeline

1. `yt-dlp` downloads the video.
2. `ffmpeg` extracts 16 kHz mono AAC audio.
3. OpenAI ASR (`whisper-1` by default) transcribes the audio with word timestamps.
4. Local VAD finds the first real speech so intro music is not captioned. Words are grouped into longer translation units on pauses and sentence punctuation.
5. An OpenAI text model (`gpt-5.6-luna` by default) runs in two phases: first it splits original speech into one-line cards by meaning, then it translates each card into Traditional Chinese. CJK defaults to 40 characters per line; other languages default to 84. Title and description are used as terminology context.
6. `ffmpeg` muxes a soft ASS subtitle track into an MKV (video and audio are copied). ASS lines are hard-wrapped with `\\N`.

Timed captions need timestamps. Use `whisper-1` or `gpt-4o-transcribe-diarize`. `gpt-transcribe` and `gpt-4o-transcribe` do not return timestamps, so they cannot be used as the ASR model here.

OpenAI file transcription accepts uploads up to 25 MB. Longer audio is split into 10-minute chunks.

## Requirements

- `uv` (provides `uv` and `uvx`)
- `ffmpeg` and `ffprobe`
- Python 3.10+ (installed automatically by uv if needed)

`yt-dlp` is a Python dependency and is installed by uv.

## Setup

Copy the example env file and set your OpenAI API key:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...
```

The script loads `.env` from the current working directory, then from this directory. An already-exported `OPENAI_API_KEY` is not overwritten.

## Usage

From this directory:

```bash
uv run youtube-bilingual 'https://www.youtube.com/watch?v=XXXX' -l ja
```

Without installing into the current environment:

```bash
uvx --from . youtube-bilingual 'https://www.youtube.com/watch?v=XXXX' -l ja
```

From a git remote:

```bash
uvx --from git+https://example.com/youtube-bilingual.git youtube-bilingual 'https://www.youtube.com/watch?v=XXXX' -l ja
```

`-l` is the spoken language as an ISO-639-1 code (`en`, `ja`, `ko`, `zh`, ...).

Output (in the current directory, unless `-o` is set):

- `<title>.mkv` — video/audio plus a default bilingual ASS track (`zho`, title `Original + zh-Hant`)
- `<title>.ass` — the same subtitles as a sidecar file

Re-run translation only (skips download, ASR, VAD, and segmentation). Requires a previous full run that kept `segmented.json` and `source.mkv`:

```bash
uv run youtube-bilingual --translate-only --work-dir /tmp/yt-bilingual-xxxx -l ja
```

Helper check that does not call the network:

```bash
uv run youtube-bilingual --self-test
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `-l`, `--language` | required | Spoken language (ISO-639-1) |
| `-o`, `--output` | `<title>.mkv` | Output MKV path |
| `--asr-model` | `whisper-1` | ASR model with timestamps (`whisper-1` or `gpt-4o-transcribe-diarize`) |
| `--model` | `gpt-5.6-luna` | OpenAI model used to split original lines and translate them |
| `--batch-size` | `100` | Cues per translation request |
| `--jobs` | `4` | Concurrent segment/translate API requests |
| `--chunk-seconds` | `600` | Audio chunk length when the file exceeds the 25 MB ASR limit |
| `--max-line-chars` | auto | Max characters per subtitle line (`40` for `ja`/`zh`/`ko`, `84` otherwise) |
| `--work-dir` | temp dir | Keep intermediate files in this directory |
| `--keep-work` | off | Do not delete the temp work directory |
| `--cookies` | none | Netscape `cookies.txt` for yt-dlp |
| `--cookies-from-browser` | none | Passed through to yt-dlp (`chrome`, `firefox`, ...) |
| `--self-test` | off | Run local helper tests and exit |
| `--no-vad` | off | Keep Whisper timestamps; do not detect speech onset |
| `--translate-only` | off | Reuse `--work-dir` `segmented.json` and only re-run translation + mux |

Age-gated or login-walled videos:

```bash
uv run youtube-bilingual 'https://www.youtube.com/watch?v=XXXX' -l en --cookies-from-browser chrome
```
