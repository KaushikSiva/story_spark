from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional


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
        logging.getLogger(__name__).info("video_postprocess: using ffmpeg exe=%s", exe)
        return exe
    except Exception:
        return None


def _import_moviepy_components():
    """Import MoviePy components compatibly across versions (>=2 preferred).

    Tries MoviePy 2.x top-level imports first, then 1.x editor shim.
    Avoids deprecated submodules like moviepy.video.compositing.concatenate.
    """
    try:
        # MoviePy 2.x flattened API
        from moviepy import (  # type: ignore
            VideoFileClip,
            ImageClip,
            ColorClip,
            concatenate_videoclips,
        )
        return VideoFileClip, ImageClip, ColorClip, concatenate_videoclips
    except Exception:
        # Fallback for older MoviePy 1.x
        from moviepy.editor import (  # type: ignore
            VideoFileClip,
            ImageClip,
            ColorClip,
            concatenate_videoclips,
        )
        return VideoFileClip, ImageClip, ColorClip, concatenate_videoclips


def append_poster_and_endcard(*, teaser_path: Path, poster_path: Optional[Path], out_path: Path) -> None:
    """Compose a new video with poster then end-card appended.

    - teaser_path: existing video
    - poster_path: optional poster image to insert for ~2s
    - out_path: final output path (will overwrite)
    """
    ff = ensure_ffmpeg()
    logging.getLogger(__name__).info(
        "video_postprocess: start teaser=%s poster=%s out=%s ffmpeg=%s",
        str(teaser_path), str(poster_path) if poster_path else None, str(out_path), ff
    )

    # Imports deferred so ensure_ffmpeg can set env first
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    # Prefer MoviePy 2.x API
    VideoFileClip, ImageClip, _ColorClip, concatenate_videoclips = _import_moviepy_components()

    base = VideoFileClip(str(teaser_path))
    w, h = base.w, base.h
    clips = [base]

    # Poster ~2s (letterboxed using a black background instead of on_color)
    if poster_path and poster_path.exists():
        logging.getLogger(__name__).info("video_postprocess: appending poster segment")
        poster_clip = ImageClip(str(poster_path))

        # Determine target size preserving aspect ratio to fit within (w, h)
        try:
            pw, ph = poster_clip.w, poster_clip.h
            scale = min(w / float(pw or 1), h / float(ph or 1))
            target_size = (max(1, int((pw or w) * scale)), max(1, int((ph or h) * scale)))
        except Exception:
            target_size = (w, h)

        # Resize poster to target_size with API-compat
        if hasattr(poster_clip, "with_size"):
            try:
                poster_clip = poster_clip.with_size(target_size)
            except Exception:
                try:
                    poster_clip = poster_clip.with_size(width=target_size[0])
                except Exception:
                    pass
        elif hasattr(poster_clip, "resize"):
            try:
                poster_clip = poster_clip.resize(newsize=target_size)
            except Exception:
                poster_clip = poster_clip.resize(width=target_size[0])

        # Set duration on poster
        if hasattr(poster_clip, "with_duration"):
            poster_clip = poster_clip.with_duration(2.0)
        else:
            poster_clip = poster_clip.set_duration(2.0)

        # Create black background
        try:
            from moviepy import ColorClip as _ColorClip  # type: ignore
        except Exception:
            from moviepy.editor import ColorClip as _ColorClip  # type: ignore

        bg = _ColorClip((w, h), color=(0, 0, 0))
        if hasattr(bg, "with_duration"):
            bg = bg.with_duration(2.0)
        else:
            bg = bg.set_duration(2.0)

        # Center the poster on background
        if hasattr(poster_clip, "with_position"):
            poster_on_bg = poster_clip.with_position("center")
        elif hasattr(poster_clip, "set_position"):
            poster_on_bg = poster_clip.set_position("center")
        else:
            poster_on_bg = poster_clip

        # Composite
        try:
            from moviepy import CompositeVideoClip  # type: ignore
        except Exception:
            from moviepy.editor import CompositeVideoClip  # type: ignore

        comp = CompositeVideoClip([bg, poster_on_bg])

        # Fade in/out effects if available
        if hasattr(comp, "with_effects"):
            try:
                from moviepy.video.fx.FadeIn import FadeIn  # type: ignore
                from moviepy.video.fx.FadeOut import FadeOut  # type: ignore
                comp = comp.with_effects([FadeIn(0.4), FadeOut(0.3)])
            except Exception:
                pass
        else:
            try:
                comp = comp.fadein(0.4).fadeout(0.3)
            except Exception:
                pass

        clips.append(comp)

    # End-card ~2s via PIL
    msg = "Coming soon to theatres near you…"
    txt_img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(txt_img)
    # Font selection
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
    # Text size
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

    tmp_end = out_path.parent / "_endcard.png"
    try:
        txt_img.save(tmp_end)
    except Exception:
        tmp_end = None

    if tmp_end and tmp_end.exists():
        logging.getLogger(__name__).info("video_postprocess: appending end card")
        end_clip = ImageClip(str(tmp_end))
        if hasattr(end_clip, "with_duration"):
            try:
                end_clip = end_clip.with_duration(2.0)
            except Exception:
                if hasattr(end_clip, "set_duration"):
                    end_clip = end_clip.set_duration(2.0)
            if hasattr(end_clip, "with_effects"):
                try:
                    from moviepy.video.fx.FadeIn import FadeIn  # type: ignore
                    end_clip = end_clip.with_effects([FadeIn(0.6)])
                except Exception:
                    try:
                        import moviepy.video.fx.all as vfx  # type: ignore
                        end_clip = vfx.fadein(end_clip, 0.6)
                    except Exception:
                        pass
        else:
            end_clip = end_clip.set_duration(2.0)
            try:
                end_clip = end_clip.fadein(0.6)
            except Exception:
                pass
        clips.append(end_clip)

    # Write out
    final = concatenate_videoclips(clips, method="compose")
    tmp_out = out_path.parent / "_tmp_out.mp4"
    logging.getLogger(__name__).info("video_postprocess: writing final video to %s", str(tmp_out))

    # Compat: MoviePy 1.x vs 2.x write_videofile signature (verbose/logger removed in 2.x)
    try:
        import inspect
        sig = inspect.signature(final.write_videofile)  # type: ignore[attr-defined]
        params = sig.parameters
        if "verbose" in params or "logger" in params:
            final.write_videofile(
                str(tmp_out), codec="libx264", audio_codec="aac", fps=base.fps or 24, verbose=False, logger=None
            )
        else:
            final.write_videofile(str(tmp_out), codec="libx264", audio_codec="aac", fps=base.fps or 24)
    except Exception:
        # Fallback: safest minimal arguments
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


def concat_videos(input1: str | Path, input2: str | Path, out_path: Path) -> str:
    """Concatenate two videos back-to-back and write to out_path.

    Uses MoviePy's concatenate_videoclips with method="compose" to handle
    differing sizes/codecs robustly.
    """
    ensure_ffmpeg()
    VideoFileClip, _ImageClip, _ColorClip, concatenate_videoclips = _import_moviepy_components()
    c1 = None
    c2 = None
    try:
        c1 = VideoFileClip(str(input1))
        c2 = VideoFileClip(str(input2))
        final = concatenate_videoclips([c1, c2], method="compose")
        # Write out with compatibility for MoviePy versions
        tmp_out = out_path.parent / "_tmp_concat.mp4"
        try:
            import inspect
            sig = inspect.signature(final.write_videofile)  # type: ignore[attr-defined]
            params = sig.parameters
            if "verbose" in params or "logger" in params:
                final.write_videofile(
                    str(tmp_out), codec="libx264", audio_codec="aac", fps=c1.fps or 24, verbose=False, logger=None
                )
            else:
                final.write_videofile(str(tmp_out), codec="libx264", audio_codec="aac", fps=c1.fps or 24)
        except Exception:
            final.write_videofile(str(tmp_out), codec="libx264", audio_codec="aac", fps=c1.fps or 24)
        try:
            if out_path.exists():
                out_path.unlink()
            tmp_out.rename(out_path)
        finally:
            try:
                final.close()
            except Exception:
                pass
        return str(out_path)
    finally:
        try:
            if c1:
                c1.close()
        except Exception:
            pass
        try:
            if c2:
                c2.close()
        except Exception:
            pass


def concat_videos_many(inputs: list[str | Path], out_path: Path) -> str:
    """Concatenate N videos in order and write to out_path.

    Uses MoviePy concatenate_videoclips(method="compose") for robustness.
    """
    ensure_ffmpeg()
    VideoFileClip, _ImageClip, _ColorClip, concatenate_videoclips = _import_moviepy_components()
    clips = []
    try:
        for p in inputs:
            clips.append(VideoFileClip(str(p)))
        if not clips:
            raise ValueError("No input videos provided")
        final = concatenate_videoclips(clips, method="compose")
        tmp_out = out_path.parent / "_tmp_concat_many.mp4"
        try:
            import inspect
            sig = inspect.signature(final.write_videofile)  # type: ignore[attr-defined]
            params = sig.parameters
            fps = getattr(clips[0], 'fps', None) or 24
            if "verbose" in params or "logger" in params:
                final.write_videofile(str(tmp_out), codec="libx264", audio_codec="aac", fps=fps, verbose=False, logger=None)
            else:
                final.write_videofile(str(tmp_out), codec="libx264", audio_codec="aac", fps=fps)
        except Exception:
            final.write_videofile(str(tmp_out), codec="libx264", audio_codec="aac", fps=getattr(clips[0], 'fps', None) or 24)
        try:
            if out_path.exists():
                out_path.unlink()
            tmp_out.rename(out_path)
        finally:
            try:
                final.close()
            except Exception:
                pass
        return str(out_path)
    finally:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
