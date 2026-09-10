#!/usr/bin/env python3
"""Remux sidecar ASS files into MKVs and embed subtitle fonts.

The original mux ran in a sandbox where fontconfig could not see the
installed Noto CJK files. Run this outside a sandbox so fc-match can
resolve the fonts, then rewrite each MKV in place:

    uv run python scripts/remux_ass.py
    uv run python scripts/remux_ass.py --dry-run output/kotodama-potori
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from youtube_bilingual import (
    DEFAULT_CHINESE_FONT,
    DEFAULT_ORIGINAL_FONT,
    log,
    mux_mkv,
    subtitle_font_files,
    which_or_exit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "output" / "kotodama-potori"


def mkv_ass_pairs(directory: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for mkv in sorted(directory.glob("*.mkv")):
        if mkv.name.startswith("."):
            continue
        ass = mkv.with_suffix(".ass")
        if not ass.is_file():
            log(f"Skip {mkv.name}: no sidecar ASS")
            continue
        pairs.append((mkv, ass))
    return pairs


def remux_one(mkv: Path, ass: Path) -> None:
    fd, raw = tempfile.mkstemp(
        prefix=f".{mkv.name}.",
        suffix=".mkv",
        dir=mkv.parent,
    )
    os.close(fd)
    tmp = Path(raw)
    try:
        mux_mkv(mkv, ass, tmp)
        os.replace(tmp, mkv)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace each MKV's subtitle track with its sidecar ASS "
            "and embed Noto Sans CJK fonts from the host fontconfig."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Directory of MKV+ASS pairs (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pairs and fonts without writing files",
    )
    args = parser.parse_args(argv)

    which_or_exit("ffmpeg")
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    fonts = subtitle_font_files()
    if not fonts:
        raise SystemExit(
            "Subtitle fonts not found. Install Noto Sans CJK "
            f"({DEFAULT_ORIGINAL_FONT} / {DEFAULT_CHINESE_FONT}) "
            "and run this script outside a sandbox."
        )
    for font in fonts:
        size_kib = max(1, font.stat().st_size // 1024)
        log(f"Host font: {font} ({size_kib} KiB)")

    pairs = mkv_ass_pairs(directory)
    if not pairs:
        raise SystemExit(f"No MKV+ASS pairs in {directory}")

    log(f"{len(pairs)} file(s) in {directory}")
    for index, (mkv, ass) in enumerate(pairs, start=1):
        log(f"[{index}/{len(pairs)}] {mkv.name}")
        if args.dry_run:
            continue
        remux_one(mkv, ass)

    log("Dry run done" if args.dry_run else "Done")


if __name__ == "__main__":
    main(sys.argv[1:])
