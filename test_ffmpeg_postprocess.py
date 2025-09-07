#!/usr/bin/env python3
"""
Standalone smoke test for MoviePy + ffmpeg + Pillow.

Creates a short teaser.mp4, then appends an optional poster image
and a black end card ("Coming soon to theatres near you…") to produce
teaser_with_credits.mp4 in the current directory.

Run:
  python3 test_ffmpeg_postprocess.py

Prereqs (in your venv):
  python3 -m pip install --upgrade moviepy Pillow imageio-ffmpeg
System:
  ffmpeg must be installed (brew install ffmpeg | apt-get install -y ffmpeg)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Fixed imports for MoviePy 2.0
from moviepy import ColorClip, VideoFileClip, ImageClip, concatenate_videoclips
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from PIL import Image, ImageDraw, ImageFont


def ensure_ffmpeg() -> Optional[str]:
    """Wire ffmpeg for MoviePy using imageio-ffmpeg if not on PATH."""
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", exe)
        try:
            from moviepy.config import change_settings  # type: ignore

            change_settings({"FFMPEG_BINARY": exe})
        except Exception:
            pass
        return exe
    except Exception:
        return None


def make_teaser(output: str = "teaser.mp4", w: int = 960, h: int = 540, dur: float = 3.0, color: tuple[int, int, int] = (20, 24, 40)) -> Path:
    """Creates a basic video file for testing."""
    # Fixed: MoviePy 2.0 API changes - use with_duration instead of duration parameter
    clip = ColorClip((w, h), color=color).with_duration(dur)
    clip = clip.with_fps(24)
    clip.write_videofile(output, codec="libx264", audio_codec="aac", fps=24)
    return Path(output)


def append_poster_and_endcard(teaser_path: Path, poster_path: Optional[Path], out_path: Path) -> None:
    """Appends a poster and end card to a video clip."""
    base = VideoFileClip(str(teaser_path))
    w, h = base.w, base.h
    clips = [base]

    # Poster ~2s (letterboxed on black)
    if poster_path and poster_path.exists():
        poster_clip = ImageClip(str(poster_path)).with_duration(2.0)
        # Fixed: MoviePy 2.0 API - use with_size instead of resize, and updated method names
        poster_clip = poster_clip.with_size(width=w).on_color(size=(w, h), color=(0, 0, 0), pos="center")
        # Fixed: Use new effects API with with_effects()
        poster_clip = poster_clip.with_effects([FadeIn(0.4), FadeOut(0.3)])
        clips.append(poster_clip)

    # End card ~2s
    msg = "Coming soon to theatres near you…"
    txt_img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(txt_img)
    # Choose a font safely
    font = None
    for name in ("Arial.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, size=max(16, int(h * 0.06)))
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    
    # Size text robustly across Pillow versions
    try:
        bbox = draw.textbbox((0, 0), msg, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            tw, th = draw.textsize(msg, font=font)
        except Exception:
            tw, th = (len(msg) * 10, int(h * 0.06))
    
    x, y = max(0, (w - tw) // 2), max(0, int(h * 0.45))
    try:
        draw.text((x, y), msg, font=font, fill=(255, 255, 255))
    except Exception:
        pass

    tmp_end = teaser_path.parent / "_endcard.png"
    try:
        txt_img.save(tmp_end)
    except Exception:
        # If saving fails, skip end card gracefully
        tmp_end = None

    if tmp_end and tmp_end.exists():
        end_clip = ImageClip(str(tmp_end)).with_duration(2.0)
        # Fixed: Use new effects API with with_effects()
        end_clip = end_clip.with_effects([FadeIn(0.6)])
        clips.append(end_clip)

    final = concatenate_videoclips(clips, method="compose")
    tmp_out = teaser_path.parent / "_tmp_out.mp4"
    final.write_videofile(str(tmp_out), codec="libx264", audio_codec="aac", fps=base.fps or 24)
    
    try:
        if out_path.exists():
            out_path.unlink()
        tmp_out.rename(out_path)
    finally:
        try:
            base.close()
        except Exception:
            pass
        try:
            final.close()
        except Exception:
            pass
        try:
            if tmp_end:
                tmp_end.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    print("=== MoviePy + ffmpeg end-card test ===")
    # Show interpreter
    import sys
    print("Interpreter:", sys.executable)

    # Wire ffmpeg
    ffmpeg_exe = ensure_ffmpeg()
    print("ffmpeg exe (imageio-ffmpeg):", ffmpeg_exe or "Not wired (PATH will be used)")

    # Verify imports (updated for MoviePy 2.0+)
    try:
        from moviepy import VideoFileClip  # A simple check for a core module
        print("moviepy:", "OK")
    except Exception as e:
        print("moviepy import FAILED:", e)
        return

    try:
        from PIL import Image  # type: ignore
        import PIL  # type: ignore
        print("Pillow:", "OK", getattr(PIL, "__version__", ""))
    except Exception as e:
        print("Pillow import FAILED:", e)
        return

    out_dir = Path(".").resolve()
    teaser = make_teaser(output=str(out_dir / "teaser.mp4"))
    print("Wrote:", teaser)

    # Set to a real poster path if you have one in this folder
    poster_path: Optional[Path] = None  # e.g., Path("first_look.jpg")
    credited = out_dir / "teaser_with_credits.mp4"
    append_poster_and_endcard(teaser, poster_path, credited)
    print("Wrote:", credited)
    print("Exists:", credited.exists(), "Size:", credited.stat().st_size if credited.exists() else 0)


if __name__ == "__main__":
    main()