import os
import uuid
import base64
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from dotenv import load_dotenv
import threading
import time

load_dotenv()  # Load env vars from .env if present

try:
    # Prefer the new google-genai client if available
    from google import genai  # type: ignore
    _GENAI_MODE = "google-genai"
except Exception:  # pragma: no cover
    genai = None
    _GENAI_MODE = None

try:
    # Types for config (GenerateContentConfig)
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover
    genai_types = None

try:
    # Fallback to legacy google.generativeai if installed
    import google.generativeai as legacy_genai  # type: ignore
except Exception:  # pragma: no cover
    legacy_genai = None


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    # Basic config
    app.config.update(
        OUTPUT_DIR=os.environ.get("OUTPUT_DIR", "static/generated"),
        DEFAULT_IMAGE_COUNT=int(os.environ.get("IMAGE_COUNT", "5")),
        IMAGE_SIZE=os.environ.get("IMAGE_SIZE", "1024x1024"),
        MODEL=os.environ.get("GENERATION_MODEL", "models/gemini-2.5-flash-image-preview"),
        OUTPUT_TTL_SECONDS=int(os.environ.get("OUTPUT_TTL_SECONDS", str(60 * 60))),  # default 1 hour
        OUTPUT_SWEEP_INTERVAL_SECONDS=int(os.environ.get("OUTPUT_SWEEP_INTERVAL_SECONDS", str(10 * 60))),  # default 10 min
        # CORS: allow all by default; tighten in production
        CORS_ALLOW_ORIGINS=os.environ.get("CORS_ALLOW_ORIGINS", "*"),
    )

    # Ensure output directory exists
    Path(app.config["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    # Initial sweep and start periodic sweeper once per process
    _sweep_output_dir(app.config["OUTPUT_DIR"], app.config["OUTPUT_TTL_SECONDS"])  # best-effort
    _start_periodic_sweeper(
        app.config["OUTPUT_DIR"],
        app.config["OUTPUT_TTL_SECONDS"],
        app.config["OUTPUT_SWEEP_INTERVAL_SECONDS"],
    )

    # Light, dependency-free CORS headers
    @app.after_request
    def add_cors_headers(resp):  # pragma: no cover
        resp.headers["Access-Control-Allow-Origin"] = app.config["CORS_ALLOW_ORIGINS"]
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp

    @app.route("/health", methods=["GET"])  # simple health check
    def health():
        return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})

    @app.route("/api/generate", methods=["GET", "OPTIONS"])
    def generate_images():
        if request.method == "OPTIONS":  # CORS preflight
            return ("", 204)

        # Read from query parameters for GET
        name = (request.args.get("name") or "").strip()
        year_str = str(request.args.get("year") or "").strip()
        style = (request.args.get("style") or "").strip()
        # Parse n safely
        n_param = request.args.get("n")
        try:
            n = int(n_param) if n_param is not None else int(app.config["DEFAULT_IMAGE_COUNT"])
        except Exception:
            n = int(app.config["DEFAULT_IMAGE_COUNT"])
        # Parse ttl
        ttl_param = request.args.get("ttl_seconds")
        try:
            ttl_seconds = int(ttl_param) if ttl_param is not None else int(app.config["OUTPUT_TTL_SECONDS"])
        except Exception:
            ttl_seconds = int(app.config["OUTPUT_TTL_SECONDS"])
        ttl_seconds = max(10, ttl_seconds)  # minimum 10s

        if not name or not year_str.isdigit():
            return (
                jsonify({
                    "error": "Invalid input",
                    "details": "Provide 'name' (string) and 'year' (number)",
                }),
                400,
            )
        year = int(year_str)

        try:
            files = _generate_and_save(
                prompt=_build_prompt(name, year, style),
                n=n,
                image_size=app.config["IMAGE_SIZE"],
                model_name=app.config["MODEL"],
                out_dir=app.config["OUTPUT_DIR"],
            )
        except Exception as e:  # pragma: no cover
            return (
                jsonify({
                    "error": "Image generation failed",
                    "details": str(e),
                }),
                500,
            )

        base_url = request.url_root.rstrip("/")
        images = [
            {
                "url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{Path(f).name}",
                "filename": Path(f).name,
            }
            for f in files
        ]

        # Schedule cleanup of generated images after TTL
        _schedule_cleanup(files, ttl_seconds)

        return jsonify({
            "name": name,
            "year": int(year),
            "count": len(images),
            "images": images,
            "model": app.config["MODEL"],
            "size": app.config["IMAGE_SIZE"],
            "ttl_seconds": ttl_seconds,
        })

    return app


def _build_prompt(name: str, year: int, style: str) -> str:
    realism_rules = _historical_realism_rules(year)
    development_rules = _development_scaling_guidance(year)
    parts = [
        f"Create a realistic, historically accurate scene featuring {name} in the year {year}.",
        "Adhere strictly to the technology, clothing, architecture, signage, vehicles, and materials available in that year.",
        "Avoid anachronisms or future tech beyond the specified year.",
        realism_rules,
        development_rules,
        "High fidelity, coherent perspective, natural lighting, detailed textures.",
    ]
    if style:
        parts.append(f"Visual style: {style}.")
    return " ".join([p for p in parts if p])


def _historical_realism_rules(year: int) -> str:
    flying_rule = (
        "Before 2040 do not include any flying objects other than airplanes or helicopters; no drones, rockets, flying cars, blimps, balloons, or UFOs."
        if year < 2040
        else ""
    )

    if year <= 1850:
        era = (
            "No airplanes or motor vehicles. No rockets, electronics, neon, plastics, or skyscrapers. "
            "Transportation is horse-drawn carts and carriages; steam trains only where plausible. Lighting is daylight or gaslight."
        )
    elif year <= 1900:
        era = (
            "No airplanes or modern automobiles. No rockets, electronics, neon, plastics, or skyscrapers. "
            "Transportation includes horse-drawn vehicles and steam trains; very early bicycles and carriages are fine."
        )
    elif year <= 1945:
        era = (
            "Allow early automobiles and propeller aircraft. No jet aircraft, rockets, digital screens, or post‑war fashion/architecture."
        )
    elif year <= 1969:
        era = (
            "Allow jet aircraft and mid‑century tech. No personal computers, smartphones, LCD/LED screens, or ultra‑tall modern skylines."
        )
    elif year <= 1999:
        era = (
            "Allow CRT monitors, pagers, early mobile phones, and period‑appropriate cars. No smartphones, flat panels, or 21st‑century vehicles."
        )
    elif year <= 2039:
        era = (
            "Contemporary technology is acceptable; avoid widespread autonomous vehicles, mass flying devices, or space travel scenes."
        )
    else:
        era = "Use technology and fashion authentic to the specified year; avoid items introduced after it."

    return " ".join([p for p in [flying_rule, era] if p])


def _development_scaling_guidance(year: int) -> str:
    if year <= 1850:
        return (
            "Urban development is sparse. Small, low‑rise buildings (1–3 stories), narrow or dirt roads, limited signage, no cars."
        )
    if year <= 1900:
        return (
            "Low‑rise towns (2–4 stories), cobblestone or early paved roads, few industrial structures, minimal signage and infrastructure."
        )
    if year <= 1945:
        return (
            "Mostly low to mid‑rise (2–6 stories), early automobiles and trams, modest road widths, limited advertising, simpler materials."
        )
    if year <= 1969:
        return (
            "Mid‑century scale: mid‑rise blocks, boxy cars, wider roads but not modern mega‑highways; restrained skylines."
        )
    if year <= 1999:
        return (
            "Growing density with some high‑rises, but overall smaller buildings and cars than modern day; CRT signage, fewer glass towers."
        )
    if year <= 2039:
        return (
            "Modern development with realistic density; avoid ultra‑tall futuristic megastructures or over‑dense sci‑fi skylines."
        )
    return "Scale density plausibly for the year and location."


def _schedule_cleanup(paths, delay_seconds: int):
    def _worker(ps, delay):
        try:
            time.sleep(delay)
            for p in ps:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass

    t = threading.Thread(target=_worker, args=(list(paths), int(delay_seconds)), daemon=True)
    t.start()


# Guard to avoid multiple sweepers in dev reload scenarios
_SWEEPER_STARTED = False


def _start_periodic_sweeper(out_dir: str, ttl_seconds: int, interval_seconds: int):
    global _SWEEPER_STARTED
    if _SWEEPER_STARTED:
        return

    def _loop():
        while True:
            try:
                _sweep_output_dir(out_dir, ttl_seconds)
            except Exception:
                pass
            time.sleep(max(30, int(interval_seconds)))

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    _SWEEPER_STARTED = True


def _sweep_output_dir(out_dir: str, ttl_seconds: int):
    now = time.time()
    p = Path(out_dir)
    if not p.exists():
        return
    for fp in p.iterdir():
        try:
            if not fp.is_file():
                continue
            age = now - fp.stat().st_mtime
            if age > ttl_seconds:
                fp.unlink(missing_ok=True)
        except Exception:
            # ignore and continue
            pass


def _generate_and_save(prompt: str, n: int, image_size: str, model_name: str, out_dir: str):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    # Prefer new google-genai client
    if _GENAI_MODE == "google-genai" and genai is not None:
        client = genai.Client(api_key=api_key)
        saved_files = []
        # Generate one image per call for reliability across SDK versions
        cfg = genai_types.GenerateContentConfig(
            response_modalities=["Text", "Image"]
        ) if genai_types is not None else None

        for _ in range(max(1, int(n))):
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=cfg,
            )
            parts = _extract_images_from_genai_response(response)
            if not parts:
                continue
            saved_files.extend(_save_images_parts(parts[:1], out_dir))

        if not saved_files:
            raise RuntimeError("No images returned from model")
        return saved_files

    # Fallback to legacy google.generativeai
    if legacy_genai is not None:
        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(model_name=model_name)
        result = model.generate_images(
            prompt=prompt,
            number_of_images=n,
            size=image_size,
        )
        images = getattr(result, "images", [])
        if not images:
            raise RuntimeError("No images returned from model")
        return _save_images_blob(images, out_dir)

    raise RuntimeError(
        "Neither 'google-genai' nor 'google.generativeai' is available. Install one to proceed."
    )


def _save_images_blob(images, out_dir: str):
    saved = []
    for img in images:
        # Support both new and legacy clients
        data = getattr(img, "data", None) or getattr(img, "_image_bytes", None) or getattr(img, "image_bytes", None)
        if data is None:
            raise RuntimeError("Unrecognized image object: missing bytes")
        # Decide extension from mime if available
        mime = getattr(img, "mime_type", None) or getattr(img, "mime", None) or "image/png"
        ext = "png" if "png" in mime else ("jpg" if "jpeg" in mime or "jpg" in mime else "png")
        filename = f"{uuid.uuid4().hex}.{ext}"
        path = Path(out_dir) / filename
        with open(path, "wb") as f:
            f.write(data)
        saved.append(str(path))
    return saved


def _save_images_parts(parts, out_dir: str):
    saved = []
    for p in parts:
        data = p.get("data")
        mime = p.get("mime") or "image/png"
        if isinstance(data, str):
            # Some SDKs return base64-encoded strings for inline_data
            try:
                data = base64.b64decode(data)
            except Exception:
                # If not base64, assume raw bytes in latin-1
                data = data.encode("latin-1", errors="ignore")
        if not isinstance(data, (bytes, bytearray)):
            raise RuntimeError("Image bytes are not in a supported format")

        ext = "png" if "png" in mime else ("jpg" if "jpeg" in mime or "jpg" in mime else "png")
        filename = f"{uuid.uuid4().hex}.{ext}"
        path = Path(out_dir) / filename
        with open(path, "wb") as f:
            f.write(data)
        saved.append(str(path))
    return saved


def _extract_images_from_genai_response(response):
    parts_out = []
    # Direct images attribute if present
    images = getattr(response, "images", None)
    if images:
        for img in images:
            data = getattr(img, "data", None) or getattr(img, "image_bytes", None) or getattr(img, "_image_bytes", None)
            mime = getattr(img, "mime_type", None) or getattr(img, "mime", None) or "image/png"
            if data:
                parts_out.append({"data": data, "mime": mime})

    # Candidates -> content -> parts
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for p in parts:
            # Try inline_data
            inline = getattr(p, "inline_data", None)
            if inline is not None:
                data = getattr(inline, "data", None)
                mime = getattr(inline, "mime_type", None) or "image/png"
                if data is not None:
                    parts_out.append({"data": data, "mime": mime})
                    continue
            # Try image field
            img = getattr(p, "image", None)
            if img is not None:
                data = getattr(img, "data", None)
                mime = getattr(img, "mime_type", None) or "image/png"
                if data is not None:
                    parts_out.append({"data": data, "mime": mime})
                    continue
            # Fallback: part itself has data/mime
            data = getattr(p, "data", None)
            mime = getattr(p, "mime_type", None)
            if data is not None and mime is not None:
                parts_out.append({"data": data, "mime": mime})

    return parts_out


if __name__ == "__main__":  # pragma: no cover
    app = create_app()
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))
