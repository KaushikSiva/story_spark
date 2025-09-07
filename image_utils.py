import base64
import mimetypes
import uuid
from pathlib import Path
from typing import List, Optional

from PIL import Image

try:  # pragma: no cover
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover
    genai_types = None


def extract_images_from_genai_response(response) -> list[dict]:
    parts_out: list[dict] = []
    images = getattr(response, "images", None)
    if images:
        for img in images:
            data = getattr(img, "data", None) or getattr(img, "image_bytes", None) or getattr(img, "_image_bytes", None)
            mime = getattr(img, "mime_type", None) or getattr(img, "mime", None) or "image/png"
            if data is not None:
                parts_out.append({"data": data, "mime": mime})

    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for p in parts:
            inline = getattr(p, "inline_data", None)
            if inline is not None:
                data = getattr(inline, "data", None)
                mime = getattr(inline, "mime_type", None) or "image/png"
                if data is not None:
                    parts_out.append({"data": data, "mime": mime})
                    continue
            img = getattr(p, "image", None)
            if img is not None:
                data = getattr(img, "data", None)
                mime = getattr(img, "mime_type", None) or "image/png"
                if data is not None:
                    parts_out.append({"data": data, "mime": mime})
                    continue
            data = getattr(p, "data", None)
            mime = getattr(p, "mime_type", None)
            if data is not None and mime is not None:
                parts_out.append({"data": data, "mime": mime})

    return parts_out


def save_images_parts(parts: list[dict], out_dir: str) -> list[str]:
    saved: list[str] = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for p in parts:
        data = p.get("data")
        mime = p.get("mime") or "image/png"
        if isinstance(data, str):
            try:
                data = base64.b64decode(data)
            except Exception:
                data = data.encode("latin-1", errors="ignore")
        if not isinstance(data, (bytes, bytearray)):
            continue
        ext = "png" if "png" in mime else ("jpg" if "jpeg" in mime or "jpg" in mime else "png")
        filename = f"{uuid.uuid4().hex}.{ext}"
        path = Path(out_dir) / filename
        with open(path, "wb") as f:
            f.write(data)
        saved.append(str(path))
    return saved


def save_image_part_fixed(part: dict, *, out_dir: str, base_name: str) -> str:
    data = part.get("data")
    mime = (part.get("mime") or "image/jpeg").lower()
    if isinstance(data, str):
        try:
            data = base64.b64decode(data)
        except Exception:
            data = data.encode("latin-1", errors="ignore")
    if not isinstance(data, (bytes, bytearray)):
        raise RuntimeError("Invalid image data for fixed save")

    bn = Path(base_name).stem
    ext = "png" if "png" in mime else "jpg"
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    for e in ("jpg", "png"):
        try:
            (out_dir_path / f"{bn}.{e}").unlink(missing_ok=True)
        except Exception:
            pass
    out_path = out_dir_path / f"{bn}.{ext}"
    with open(out_path, "wb") as f:
        f.write(data)
    return str(out_path)


def stitch_images_grid(files: list[str], *, out_dir: str, cols: Optional[int] = None, rows: Optional[int] = None, tile_size: int = 512, bg_color: tuple[int, int, int] = (0, 0, 0), fixed_name: Optional[str] = None) -> Optional[str]:
    images: list[Image.Image] = []
    for fp in files:
        try:
            im = Image.open(fp).convert("RGB")
            images.append(im)
        except Exception:
            continue
    if not images:
        return None
    n = len(images)
    if cols is None or rows is None:
        grid_map = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (3, 2), 6: (3, 2), 7: (4, 2), 8: (4, 2), 9: (3, 3), 10: (4, 3), 11: (4, 3), 12: (4, 3)}
        if n in grid_map:
            cols_calc, rows_calc = grid_map[n]
        else:
            import math
            cols_calc = int(math.ceil(math.sqrt(n)))
            rows_calc = int(math.ceil(n / max(1, cols_calc)))
        cols, rows = cols_calc, rows_calc
    cols = max(1, int(cols))
    rows = max(1, int(rows))

    thumbs: list[Image.Image] = []
    for im in images[: cols * rows]:
        tw = th = int(tile_size)
        im2 = im.copy()
        im2.thumbnail((tw, th), Image.LANCZOS)
        canvas = Image.new("RGB", (tw, th), bg_color)
        x = (tw - im2.width) // 2
        y = (th - im2.height) // 2
        canvas.paste(im2, (x, y))
        thumbs.append(canvas)

    width = cols * tile_size
    height = rows * tile_size
    collage = Image.new("RGB", (width, height), bg_color)
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(thumbs):
                break
            collage.paste(thumbs[i], (c * tile_size, r * tile_size))
            i += 1

    out_name = fixed_name or f"{uuid.uuid4().hex}.jpg"
    out_path = str(Path(out_dir) / out_name)
    if fixed_name:
        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass
    try:
        collage.save(out_path, format="JPEG", quality=90)
        return out_path
    except Exception:
        return None


def load_reference_image_parts(*, people_dir: str, k: int) -> list:
    try:
        p = Path(people_dir)
        if not p.exists() or not p.is_dir():
            return []
        all_files = [
            f for f in p.iterdir() if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if not all_files:
            return []
        import random as _rnd
        chosen = _rnd.sample(all_files, k=min(len(all_files), max(1, int(k))))
        parts = []
        for f in chosen:
            try:
                data = f.read_bytes()
                mime = mimetypes.guess_type(f.name)[0] or ("image/png" if f.suffix.lower() == ".png" else "image/jpeg")
                if genai_types and hasattr(genai_types, "Part") and hasattr(genai_types.Part, "from_bytes"):
                    parts.append(genai_types.Part.from_bytes(data=data, mime_type=mime))
                else:
                    parts.append({"inline_data": {"data": base64.b64encode(data).decode("ascii"), "mime_type": mime}})
            except Exception:
                continue
        return parts
    except Exception:
        return []


def file_to_genai_part(path: str):
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        data = p.read_bytes()
        mime = mimetypes.guess_type(p.name)[0] or ("image/png" if p.suffix.lower() == ".png" else "image/jpeg")
        if genai_types and hasattr(genai_types, "Part") and hasattr(genai_types.Part, "from_bytes"):
            return genai_types.Part.from_bytes(data=data, mime_type=mime)
        else:
            return {"inline_data": {"data": base64.b64encode(data).decode("ascii"), "mime_type": mime}}
    except Exception:
        return None

