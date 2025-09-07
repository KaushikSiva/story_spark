import os
import json
import logging
import random
import argparse
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request
from dotenv import load_dotenv
import requests
from pathlib import Path
from PIL import Image
import uuid
import mimetypes


load_dotenv()


try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
    _GENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    genai = None
    genai_types = None
    _GENAI_AVAILABLE = False


def create_app(*, enable_veo: bool = False) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    
    # Cache for last motion poster response to avoid re-passing synopsis
    _last_motion_poster = {}

    # Reuse the same LLM config style as app2.py and add image-gen settings
    app.config.update(
        CORS_ALLOW_ORIGINS=os.environ.get("CORS_ALLOW_ORIGINS", "*"),
        LLM_BASE_URL=os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"),
        LLM_MODEL=os.environ.get("LLM_MODEL", "openai/gpt-oss-20b"),
        LLM_TIMEOUT_SECONDS=int(os.environ.get("LLM_TIMEOUT_SECONDS", "45")),
        LLM_API_KEY=os.environ.get("LLM_API_KEY", ""),
        LLM_API_STYLE=os.environ.get("LLM_API_STYLE", "auto"),  # auto|chat|completions
        LLM_CHAT_PATH=os.environ.get("LLM_CHAT_PATH", "/chat/completions"),
        LLM_COMPLETIONS_PATH=os.environ.get("LLM_COMPLETIONS_PATH", "/completions"),
        OUTPUT_DIR=os.environ.get("OUTPUT_DIR", "static/generated"),
        IMAGE_STORY_MODEL=os.environ.get("IMAGE_STORY_MODEL", "models/gemini-2.5-flash-image-preview"),
        FALLBACK_IMAGE_MODEL=os.environ.get("FALLBACK_IMAGE_MODEL", "models/gemini-2.5-flash-image-preview"),
        FAL_API_KEY=os.environ.get("FAL_API_KEY", ""),
        FAL_VEO3_MODEL=os.environ.get("FAL_VEO3_MODEL", "fal-ai/veo3/image-to-video"),
        VIDEO_TIMEOUT_SECONDS=int(os.environ.get("VIDEO_TIMEOUT_SECONDS", "300")),  # 5 minutes for video generation
        PEOPLE_DIR=os.environ.get("PEOPLE_DIR", "static/people"),
    )

    # Ensure output path exists
    Path(app.config["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    @app.after_request
    def add_cors_headers(resp):  # pragma: no cover
        resp.headers["Access-Control-Allow-Origin"] = app.config["CORS_ALLOW_ORIGINS"]
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp

    @app.route("/health3", methods=["GET"])  # health for app3
    def health():
        return jsonify({"status": "ok"})

    # Simple UI routes
    @app.route("/", methods=["GET"])  # serve UI
    @app.route("/ui", methods=["GET"])  # alias
    def ui_root():  # pragma: no cover
        try:
            return app.send_static_file("ui/index.html")
        except Exception:
            return "UI not found. Ensure static/ui/index.html exists.", 404

    @app.route("/api/motion_poster", methods=["GET", "OPTIONS"])
    def motion_poster():
        if request.method == "OPTIONS":
            return ("", 204)

        title = (request.args.get("title") or request.args.get("movie") or "").strip() or None
        genre = (request.args.get("genre") or "").strip() or None
        # Prefer 'synopsis'; accept legacy 'scene' for compatibility
        synopsis = (request.args.get("synopsis") or request.args.get("scene") or "").strip() or None
        seed_param = request.args.get("seed")
        try:
            seed = int(seed_param) if seed_param is not None else None
        except Exception:
            seed = None

        require_local = (request.args.get("require_local") or "").lower() in {"1", "true", "yes"}
        api_style = (request.args.get("api_style") or app.config["LLM_API_STYLE"]).lower()

        result, source, llm_error = generate_motion_poster_script(
            title=title,
            genre=genre,
            synopsis=synopsis,
            seed=seed,
            base_url=app.config["LLM_BASE_URL"],
            model=app.config["LLM_MODEL"],
            timeout=app.config["LLM_TIMEOUT_SECONDS"],
            api_key=app.config["LLM_API_KEY"],
            require_local=require_local,
            api_style=api_style,
            chat_path=app.config["LLM_CHAT_PATH"],
            completions_path=app.config["LLM_COMPLETIONS_PATH"],
        )

        # Use the finalized title for logging (request param may be None)
        final_title_for_log = (result.get("title") or title or "Untitled")
        log.info(
            "motion_poster source=%s title=%s genre=%s synopsis=%s",
            source,
            final_title_for_log,
            result.get("genre"),
            result.get("synopsis"),
        )
        if llm_error:
            log.warning("motion_poster llm_error=%s", llm_error)

        final_title = (result.get("title") or title or "Untitled")
        payload = {
            "title": final_title,
            "synopsis": result.get("synopsis", ""),
            "num_characters": int(result.get("num_characters", 1) or 1),
            "genre": result.get("genre", ""),
            "source": source,
        }
        # Optional: generate images from synopsis (default ON). Disable with generate_images=0/false/no
        flag_raw = (request.args.get("generate_images") or request.args.get("images"))
        gen_images_flag = True if flag_raw is None else (str(flag_raw).lower() not in {"0", "false", "no"})
        if gen_images_flag:
            try:
                n_param = request.args.get("n")
                n = int(n_param) if n_param is not None else 8
                n = max(1, min(12, n))
            except Exception:
                n = 8
            # Character reference handling: derive from generated JSON only
            character_target = int(payload.get("num_characters", 1) or 1)
            log.info(
                "character_target=%s; preparing reference images from %s (k=%s)",
                character_target,
                app.config["PEOPLE_DIR"],
                character_target + 4,
            )
            reference_parts = load_reference_image_parts(
                people_dir=app.config["PEOPLE_DIR"],
                k=character_target + 4,
            )
            try:
                # Log the effective count we will attach to Gemini
                log.info("gemini reference count=%s", len(reference_parts))
            except Exception:
                pass
            try:
                files = generate_images_from_synopsis(
                    synopsis=payload["synopsis"],
                    model_name=app.config["IMAGE_STORY_MODEL"],
                    out_dir=app.config["OUTPUT_DIR"],
                    n=n,
                    reference_parts=reference_parts,
                    character_target=character_target,
                )
                base_url = request.url_root.rstrip("/")
                payload["images"] = [
                    {
                        "url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{Path(f).name}",
                        "filename": Path(f).name,
                    }
                    for f in files
                ]
                # Also create a stitched collage image combining all parts into one
                stitched = stitch_images_grid(files, out_dir=app.config["OUTPUT_DIR"], tile_size=512)
                if stitched:
                    payload["stitched_image"] = {
                        "url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{Path(stitched).name}",
                        "filename": Path(stitched).name,
                    }
                # Create a single first-look poster as well
                safe_title = (payload.get("title") or title or "Untitled").strip() or "Untitled"
                poster_file = generate_poster_from_synopsis(
                    title=safe_title,
                    synopsis=payload["synopsis"],
                    director="Bruno",
                    model_name=app.config["IMAGE_STORY_MODEL"],
                    out_dir=app.config["OUTPUT_DIR"],
                    reference_parts=reference_parts,
                    character_target=character_target,
                )
                if poster_file:
                    payload["poster_image"] = {
                        "url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{Path(poster_file).name}",
                        "filename": Path(poster_file).name,
                    }
            except Exception as e:  # pragma: no cover
                # Do not fail the whole request; just omit images
                logging.getLogger(__name__).warning("image generation failed: %s", e)
        
        # Cache this response for video teaser generation
        _last_motion_poster.clear()
        _last_motion_poster.update(payload)
        
        return jsonify(payload)

    @app.route("/api/video_teaser", methods=["GET", "OPTIONS"])
    def video_teaser():
        if request.method == "OPTIONS":
            return ("", 204)

        # Check if Veo model is enabled
        if not enable_veo:
            return jsonify({
                "error": "Video teaser generation is disabled", 
                "details": "Start the application with --model veo to enable video generation"
            }), 503

        # Get parameters from query string
        title = (request.args.get("title") or "").strip() or None
        synopsis = (request.args.get("synopsis") or "").strip() or None
        stitched_image_url = (request.args.get("stitched_image_url") or "").strip() or None
        duration = int(request.args.get("duration", 8))  # Default 8 seconds (FAL default)
        
        # Try to get synopsis from last motion_poster response if not provided
        if not synopsis and _last_motion_poster:
            synopsis = _last_motion_poster.get("synopsis")
            title = title or _last_motion_poster.get("title")
            logging.getLogger(__name__).info("Using cached synopsis from last motion_poster response")
        
        # Try to get stitched_image_url from last motion_poster response if not provided
        if not stitched_image_url and _last_motion_poster and "stitched_image" in _last_motion_poster:
            stitched_image_url = _last_motion_poster["stitched_image"]["url"]
            logging.getLogger(__name__).info("Using cached stitched_image_url from last motion_poster response")
        
        if not synopsis:
            return jsonify({"error": "synopsis parameter is required or call /api/motion_poster first"}), 400
        if not stitched_image_url:
            return jsonify({"error": "stitched_image_url parameter is required or call /api/motion_poster first"}), 400

        try:
            logging.getLogger(__name__).info("Starting video teaser generation...")
            video_result = generate_video_teaser(
                title=title,
                synopsis=synopsis,
                image_url=stitched_image_url,
                duration=duration,
                fal_api_key=app.config["FAL_API_KEY"],
                model=app.config["FAL_VEO3_MODEL"],
                timeout=app.config["VIDEO_TIMEOUT_SECONDS"]
            )
            logging.getLogger(__name__).info("Video teaser generation completed")
            
            # Add image info to response
            video_result["stitched_image_url"] = stitched_image_url
            
            base_url = request.url_root.rstrip("/")
            if video_result.get("local_file"):
                video_result["url"] = f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{Path(video_result['local_file']).name}"
            
            return jsonify(video_result)
            
        except Exception as e:
            logging.getLogger(__name__).error("Video teaser generation failed: %s", e)
            return jsonify({
                "error": "Video generation failed", 
                "details": str(e)
            }), 500

    return app


def generate_motion_poster_script(*, title: str | None, genre: str | None, synopsis: str | None, seed: int | None, base_url: str, model: str, timeout: int, api_key: str, require_local: bool, api_style: str, chat_path: str, completions_path: str) -> Tuple[Dict[str, Any], str, str | None]:
    last_err: str | None = None
    try:
        data = _poster_from_local_gpt(
            title=title,
            genre=genre,
            synopsis=synopsis,
            seed=seed,
            base_url=base_url,
            model=model,
            timeout=timeout,
            api_key=api_key,
            api_style=api_style,
            chat_path=chat_path,
            completions_path=completions_path,
        )
        return data, "local_gpt", None
    except Exception as e:
        last_err = str(e)
        if require_local:
            raise
    fallback = _fallback_poster(title=title, genre=genre, synopsis=synopsis, seed=seed)
    return fallback, "fallback", last_err


def _poster_from_local_gpt(*, title: str | None, genre: str | None, synopsis: str | None, seed: int | None, base_url: str, model: str, timeout: int, api_key: str, api_style: str, chat_path: str, completions_path: str) -> Dict[str, Any]:
    chat_url = base_url.rstrip("/") + chat_path
    comp_url = base_url.rstrip("/") + completions_path

    sys_prompt = (
        "You are a film marketing copywriter. Generate a movie synopsis and related metadata. "
        "Return ONLY a valid JSON object with EXACT keys and types: "
        "{\"title\":\"...\",\"synopsis\":\"...\",\"num_characters\":<integer>,\"genre\":\"...\"}. "
        "Rules: 'synopsis' is 7–8 lines (each line a separate sentence), summarizing the film (setup, characters, inciting event, rising stakes, climax hint, resolution tone) without major spoilers. "
        "'num_characters' is the count of distinct principal characters implied on screen (estimate). "
        "'genre' is 1–3 words (e.g., Action, Sci‑Fi Thriller). No extra text outside JSON."
    )

    user_bits = []
    if title:
        user_bits.append(f"Title: {title}")
    if genre:
        user_bits.append(f"Genre: {genre}")
    if synopsis:
        user_bits.append(f"Synopsis cue: {synopsis}")
    if not genre:
        user_bits.append("Choose a fitting primary genre.")
    if not synopsis:
        user_bits.append("Invent a 7–8 line, PG-friendly movie synopsis with no major spoilers.")
    if not title:
        user_bits.append("Invent a compelling 1–4 word title and include it in the JSON 'title' field.")
    user_bits.append("Keep it spoiler-light and cinematic.")
    user_prompt = " \n".join(user_bits) if user_bits else "Create a 7–8 line, PG-friendly movie synopsis with title and metadata."

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 1,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
        "stop": ["<|end|>", "<|start|>", "<|channel|>"],
    }
    if seed is not None:
        payload["seed"] = seed

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def call_chat(full: bool):
        pl = payload if full else {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        }
        if seed is not None:
            pl["seed"] = seed
        r = requests.post(chat_url, json=pl, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def call_completions():
        prompt = f"System: {sys_prompt}\nUser: {user_prompt}"
        pl = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 300,
        }
        if seed is not None:
            pl["seed"] = seed
        r = requests.post(comp_url, json=pl, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    data = None
    attempts = []
    style = (api_style or "auto").lower()
    try_order = []
    if style == "chat":
        try_order = ["chat_min", "chat_full"]
    elif style == "completions":
        try_order = ["comp"]
    else:
        try_order = ["chat_min", "chat_full", "comp"]

    for mode in try_order:
        try:
            if mode == "chat_full":
                data = call_chat(full=True)
            elif mode == "chat_min":
                data = call_chat(full=False)
            elif mode == "comp":
                data = call_completions()
            attempts.append({"mode": mode, "ok": True})
            break
        except Exception as e:
            attempts.append({"mode": mode, "ok": False, "err": str(e)})
            data = None
            continue
    if data is None:
        raise RuntimeError(f"LLM call failed: {attempts}")

    # Extract text content from either chat or completions style response
    content = (
        data.get("choices", [{}])[0].get("message", {}).get("content")
        if isinstance(data.get("choices", [{}])[0], dict) and data.get("choices", [{}])[0].get("message") is not None
        else data.get("choices", [{}])[0].get("text", "")
    ) or ""

    cleaned = _strip_llm_artifacts(content)
    json_text = _extract_json_object(cleaned)
    if json_text is None:
        # Try direct parse in case server enforced JSON mode and returned raw object
        if isinstance(data, dict):
            try:
                # Some servers return a parsed JSON object directly
                if ("synopsis" in data and "num_characters" in data) or ("title" in data and "synopsis" in data):
                    return _normalize_poster_fields(data)
            except Exception:
                pass
        # Try to recover simple fields from raw text heuristically
        raise ValueError("Unable to parse motion poster JSON from LLM output")

    try:
        parsed = json.loads(json_text)
        return _normalize_poster_fields(parsed)
    except Exception as e:
        raise ValueError(f"Invalid JSON from LLM: {e}")


def _normalize_poster_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    title = obj.get("title")
    synopsis = obj.get("synopsis")
    genre = obj.get("genre")
    numc = obj.get("num_characters")
    # Back-compat: accept 'scene' if 'synopsis' is missing
    if synopsis is None and isinstance(obj, dict):
        synopsis = obj.get("scene")

    if not isinstance(title, str) or not title.strip():
        title = "Untitled"
    if not isinstance(synopsis, str):
        synopsis = str(synopsis) if synopsis is not None else "First look reveal"
    if not isinstance(genre, str):
        genre = str(genre) if genre is not None else "Drama"
    try:
        # Infer from synopsis instead of deprecated script
        numc_int = int(numc) if numc is not None else _infer_num_characters(synopsis)
    except Exception:
        numc_int = _infer_num_characters(synopsis)
    numc_int = max(0, min(10, numc_int))

    final_synopsis = _enforce_synopsis_lines((synopsis or "").strip(), genre=str(genre))

    return {
        "title": title.strip() or "Untitled",
        "synopsis": final_synopsis,
        "num_characters": numc_int,
        "genre": genre.strip() or "Drama",
    }


def _enforce_synopsis_lines(text: str, *, genre: str | None) -> str:
    import re
    if not text:
        text = "A mysterious event sets quiet lives in motion."
    # Normalize line breaks; split to lines if present, else by sentences
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        # Split by sentence enders
        parts = re.split(r"(?<=[.!?])\s+", text)
        lines = [p.strip() for p in parts if p.strip()]
    # Truncate to at most 8 lines
    lines = lines[:8]
    # Pad to at least 7 lines with genre-aware beats
    g = (genre or "Drama").lower()
    fillers = [
        "A reluctant protagonist steps forward, facing an unexpected choice.",
        "Allies emerge, but trust comes with a cost.",
        "A discovery upends what they believe is true.",
        f"Stakes rise as forces of {g} close in.",
        "Bonds are tested; a hidden weakness surfaces.",
        "A risky plan forms, promising hope and danger.",
        "Their world will not be the same again.",
    ]
    i = 0
    while len(lines) < 7 and i < len(fillers):
        lines.append(fillers[i])
        i += 1
    # Ensure 7–8 lines; if exactly 7, add a closing tone line
    if len(lines) == 7:
        lines.append("The path forward is clear—and costly.")
    # Join back
    return "\n".join(lines[:8])


def _infer_num_characters(script: str) -> int:
    if not script:
        return 1
    s = script.lower()
    # very rough heuristics
    if any(w in s for w in ["crowd", "soldiers", "villagers", "people"]):
        return 5
    if any(w in s for w in ["two ", "duo", "pair", "both "]):
        return 2
    if any(w in s for w in ["three", "trio"]):
        return 3
    return 1


def _fallback_poster(*, title: str | None, genre: str | None, synopsis: str | None, seed: int | None) -> Dict[str, Any]:
    rnd = random.Random(seed)
    genres = [
        "Action Thriller",
        "Sci‑Fi",
        "Fantasy",
        "Drama",
        "Mystery",
        "Horror",
        "Adventure",
    ]
    synopses = [
        "A lone figure in rain turns toward distant sirens",
        "Sunlit alley, dust drifting as a shadow passes",
        "Storm gathers over a silent, evacuated city",
        "Neon-lit street at midnight, footsteps approach",
        "Windswept desert ridge, a beacon flickers on",
        "Tide crashes against cliffs; an emblem emerges",
    ]

    g = genre or rnd.choice(genres)
    sc = synopsis or rnd.choice(synopses)
    t = title or "Untitled"

    # Build a 7–8 line synopsis
    lines = [
        f"In {sc.lower()}, tensions simmer as a mystery takes shape.",
        "A reluctant protagonist steps forward, haunted by a past choice.",
        "Allies gather, each with a flaw that complicates trust.",
        "An inciting discovery pushes them onto a perilous path.",
        f"Stakes escalate; forces of {g.lower()} menace close in.",
        "Betrayal cuts deep, but an unexpected bond offers hope.",
        "A risky plan forms, setting the stage for a decisive clash.",
        "The outcome will change their world—no turning back now.",
    ]
    synopsis_text = "\n".join(lines[:8])
    return {
        "title": t,
        "synopsis": synopsis_text,
        "num_characters": 1,
        "genre": g,
    }


def build_images_prompt_from_synopsis(synopsis: str, n: int, *, character_target: int | None) -> str:
    n = max(1, n)
    ct = int(character_target) if character_target is not None else None
    ref_note = (
        f"You may be given up to {ct + 4 if ct is not None else 8} candidate human reference photos from a 'people' folder. "
        f"If any match the story's characters, use whichever {ct if ct is not None else 'few'} are most suitable. "
        f"If characters are non-human or no references fit, ignore them and invent consistent characters. "
    )
    return (
        f"Create a beautifully entertaining {n} part story with {n} narrative beats inspired by the following movie synopsis. "
        f"Tell the story purely through imagery with no words or text on the images. "
        f"Keep characters, setting, and styling consistent across all parts. {ref_note}\n\n"
        f"Synopsis:\n{synopsis.strip()}\n\n"
        f"Output: Create a single, coherent image that visually stitches together all {n} parts into one cinematic collage."
    )


def generate_images_from_synopsis(*, synopsis: str, model_name: str, out_dir: str, n: int = 8, reference_parts: list | None = None, character_target: int | None = None) -> list[str]:
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google-genai is not installed")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)
    cfg = genai_types.GenerateContentConfig(response_modalities=["Text", "Image"]) if genai_types else None

    prompt = build_images_prompt_from_synopsis(_enforce_synopsis_lines(synopsis, genre=None), n, character_target=character_target)

    def _call(model: str):
        contents = (list(reference_parts) + [prompt]) if reference_parts else prompt
        return client.models.generate_content(model=model, contents=contents, config=cfg)

    response = None
    try:
        response = _call(model_name)
    except Exception as e:
        msg = str(e).lower()
        if any(tok in msg for tok in ["not_found", "404", "not found", "is not supported"]):
            fb = os.environ.get("FALLBACK_IMAGE_MODEL", "models/gemini-2.5-flash-image-preview")
            if model_name != fb:
                response = _call(fb)
            else:
                raise
        else:
            raise

    parts = _extract_images_from_genai_response(response)
    saved = _save_images_parts(parts[:n], out_dir) if parts else []

    # If batch call returned fewer than requested, try per-line generation
    if len(saved) < 1:
        lines = [ln.strip() for ln in _enforce_synopsis_lines(synopsis, genre=None).splitlines() if ln.strip()]
        for ln in lines:
            if len(saved) >= n:
                break
            single_prompt = (
                "Create one cinematic image illustrating the following story beat. "
                "Do not include any words or text on the image. Maintain consistent characters and setting.\n"
                f"Beat: {ln}"
            )
            try:
                contents = (list(reference_parts) + [single_prompt]) if reference_parts else single_prompt
                r = client.models.generate_content(model=model_name, contents=contents, config=cfg)
            except Exception:
                # fallback once
                fb = os.environ.get("FALLBACK_IMAGE_MODEL", "models/gemini-2.5-flash-image-preview")
                contents = (list(reference_parts) + [single_prompt]) if reference_parts else single_prompt
                r = client.models.generate_content(model=fb, contents=contents, config=cfg)
            more = _extract_images_from_genai_response(r)
            if more:
                saved.extend(_save_images_parts(more[:1], out_dir))

    if not saved:
        raise RuntimeError("No images returned for synopsis prompt")
    return saved[:n]


def build_poster_prompt(title: str, synopsis: str, director: str = "Bruno", *, character_target: int | None = None) -> str:
    return (
        f"Design a cinematic FIRST LOOK MOVIE POSTER for the film titled '{title}'. "
        f"Director credit: Directed by {director}. "
        f"Use the following synopsis to guide the imagery, characters, tone, and setting. "
        f"If candidate human reference photos are provided, choose whichever best match up to {character_target if character_target is not None else 4} characters; "
        f"if characters are non-human or references do not fit, ignore them and invent consistent characters. "
        f"Include tasteful, legible on-poster text: the film title '{title}', the credit 'Directed by {director}', and 2–4 thematically appropriate character names you invent (avoid real IP). "
        f"Style: high fidelity, cohesive layout, strong typography, filmic color grading, premium key-art quality. "
        f"Avoid watermarks or logos.\n\n"
        f"Synopsis:\n{synopsis.strip()}"
    )


def generate_poster_from_synopsis(*, title: str, synopsis: str, director: str, model_name: str, out_dir: str, reference_parts: list | None = None, character_target: int | None = None) -> str | None:
    if not _GENAI_AVAILABLE:
        return None
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)
    cfg = genai_types.GenerateContentConfig(response_modalities=["Text", "Image"]) if genai_types else None

    prompt = build_poster_prompt(title, _enforce_synopsis_lines(synopsis, genre=None), director, character_target=character_target)

    def _call(model: str):
        contents = (list(reference_parts) + [prompt]) if reference_parts else prompt
        return client.models.generate_content(model=model, contents=contents, config=cfg)

    try:
        response = _call(model_name)
    except Exception:
        fb = os.environ.get("FALLBACK_IMAGE_MODEL", "models/gemini-2.5-flash-image-preview")
        try:
            response = _call(fb)
        except Exception:
            return None

    # response handled in try/except above

    parts = _extract_images_from_genai_response(response)
    if not parts:
        return None
    saved = _save_images_parts(parts[:1], out_dir)
    return saved[0] if saved else None


def load_reference_image_parts(*, people_dir: str, k: int) -> list:
    # Collect candidate image file paths
    try:
        p = Path(people_dir)
        if not p.exists() or not p.is_dir():
            try:
                logging.getLogger(__name__).info(
                    "PEOPLE_DIR missing or not a directory: %s (no references)", people_dir
                )
            except Exception:
                pass
            return []
        all_files = [
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if not all_files:
            try:
                logging.getLogger(__name__).info(
                    "PEOPLE_DIR has no supported image files: %s", people_dir
                )
            except Exception:
                pass
            return []
        import random as _rnd
        chosen = _rnd.sample(all_files, k=min(len(all_files), max(1, int(k))))
        try:
            logging.getLogger(__name__).info(
                "gemini reference images=%s", [str(f) for f in chosen]
            )
        except Exception:
            pass
        parts = []
        for f in chosen:
            try:
                data = f.read_bytes()
                mime = mimetypes.guess_type(f.name)[0] or ("image/png" if f.suffix.lower() == ".png" else "image/jpeg")
                if genai_types and hasattr(genai_types, "Part") and hasattr(genai_types.Part, "from_bytes"):
                    parts.append(genai_types.Part.from_bytes(data=data, mime_type=mime))
                else:
                    import base64
                    parts.append({"inline_data": {"data": base64.b64encode(data).decode("ascii"), "mime_type": mime}})
            except Exception:
                continue
        return parts
    except Exception:
        return []


def _extract_images_from_genai_response(response):
    parts_out = []
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


def _save_images_parts(parts, out_dir: str):
    from pathlib import Path
    import base64
    import uuid
    saved = []
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


def stitch_images_grid(files: list[str], *, out_dir: str, cols: int | None = None, rows: int | None = None, tile_size: int = 512, bg_color: tuple[int, int, int] = (0, 0, 0)) -> str | None:
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
        # Choose a pleasant grid for common counts
        grid_map = {
            1: (1, 1),
            2: (2, 1),
            3: (3, 1),
            4: (2, 2),
            5: (3, 2),
            6: (3, 2),
            7: (4, 2),
            8: (4, 2),
            9: (3, 3),
            10: (4, 3),
            11: (4, 3),
            12: (4, 3),
        }
        if n in grid_map:
            cols_calc, rows_calc = grid_map[n]
        else:
            import math
            cols_calc = int(math.ceil(math.sqrt(n)))
            rows_calc = int(math.ceil(n / max(1, cols_calc)))
        cols, rows = cols_calc, rows_calc
    cols = max(1, int(cols))
    rows = max(1, int(rows))

    # Normalize to uniform thumbnails
    thumbs: list[Image.Image] = []
    for im in images[: cols * rows]:
        # Preserve aspect ratio within tile bounds, then letterbox on background
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

    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = str(Path(out_dir) / out_name)
    try:
        collage.save(out_path, format="JPEG", quality=90)
        return out_path
    except Exception:
        return None


def _strip_llm_artifacts(text: str) -> str:
    s = text.strip()
    # Remove code fences
    if s.startswith("```"):
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
        s = "\n".join(lines).strip()
    # Remove common channel tags
    for tag in ["<|channel|>analysis", "<|channel|>final", "<|start|>assistant", "<|end|>"]:
        s = s.replace(tag, "")
    return s.strip()


def _extract_json_object(text: str) -> str | None:
    # Find first '{' and scan to matching '}' accounting for quotes and escapes
    try:
        start = text.index('{')
    except ValueError:
        return None
    i = start
    depth = 0
    in_str = False
    esc = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        i += 1
    return None


def generate_video_teaser(*, title: str | None, synopsis: str, image_url: str, duration: int, fal_api_key: str, model: str, timeout: int) -> Dict[str, Any]:
    """Generate a video teaser using FAL.ai's Veo3 image-to-video model"""
    import time
    
    log = logging.getLogger(__name__)
    
    if not fal_api_key:
        raise RuntimeError("FAL_API_KEY not set")
    
    # Handle image URL - convert localhost URLs to base64 data URIs
    final_image_url = image_url
    if image_url.startswith(('http://localhost', 'http://127.0.0.1')):
        # For localhost URLs, convert to base64 data URI since FAL.ai can't access localhost
        try:
            img_response = requests.get(image_url, timeout=10)
            img_response.raise_for_status()
            
            # Determine MIME type from response headers or URL extension
            content_type = img_response.headers.get('content-type', 'image/jpeg')
            if not content_type.startswith('image/'):
                # Fallback based on URL extension
                if image_url.lower().endswith('.png'):
                    content_type = 'image/png'
                else:
                    content_type = 'image/jpeg'
            
            # Convert to base64 data URI
            import base64
            image_data = base64.b64encode(img_response.content).decode('utf-8')
            final_image_url = f"data:{content_type};base64,{image_data}"
            logging.getLogger(__name__).info(f"Converted localhost URL to base64 data URI")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot access local image URL {image_url}: {str(e)}")
    else:
        # For external URLs, validate they're accessible
        if not image_url.startswith(('http://', 'https://')):
            raise RuntimeError("image_url must be a valid HTTP/HTTPS URL")
        
        try:
            img_response = requests.head(image_url, timeout=10)
            if img_response.status_code != 200:
                raise RuntimeError(f"Image URL not accessible: HTTP {img_response.status_code}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot access image URL {image_url}: {str(e)}")
    
    # Build the prompt for video generation
    video_prompt = build_video_prompt(title=title, synopsis=synopsis, duration=duration)
    
    # Prepare the request payload according to FAL.ai Veo3 API
    payload = {
        "prompt": video_prompt,
        "image_url": final_image_url,
        "duration": f"{duration}s",  # FAL expects format like "8s"
        "resolution": "720p",  # FAL default
        "generate_audio": True
    }
    
    headers = {
        "Authorization": f"Key {fal_api_key}",
        "Content-Type": "application/json"
    }
    
    # Submit the video generation request
    submit_url = f"https://fal.run/{model}"
    
    # Log the request for debugging
    logging.getLogger(__name__).info(f"FAL.ai request to {submit_url} with payload: {payload}")
    
    try:
        response = requests.post(submit_url, json=payload, headers=headers, timeout=timeout)
        if response.status_code != 200:
            # Get detailed error info for debugging
            try:
                error_details = response.json()
                log.error(f"FAL.ai API error details: {error_details}")
                raise RuntimeError(f"FAL.ai API error ({response.status_code}): {error_details}")
            except:
                log.error(f"FAL.ai API error response: {response.text}")
                raise RuntimeError(f"FAL.ai API error ({response.status_code}): {response.text}")
        result = response.json()
        
        # Check if we have a direct video URL or need to poll for results
        if "video" in result and "url" in result["video"]:
            video_url = result["video"]["url"]
            
            # Download the video file locally
            local_file = download_video_file(video_url, "static/generated")
            
            return {
                "video_url": video_url,
                "local_file": local_file,
                "duration": duration,
                "prompt": video_prompt,
                "model": model,
                "status": "completed"
            }
        elif "request_id" in result:
            # Poll for completion if async
            return poll_video_completion(result["request_id"], fal_api_key, model, timeout)
        else:
            raise RuntimeError(f"Unexpected response format: {result}")
            
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Video generation timed out after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"FAL.ai API request failed: {str(e)}")


def build_video_prompt(*, title: str | None, synopsis: str, duration: int) -> str:
    """Build an effective prompt for video generation"""
    prompt_parts = [
        f"Create a cinematic {duration}-second movie teaser trailer.",
        "High production value with dramatic lighting and professional cinematography.",
        "Include dynamic camera movements, engaging visual storytelling.",
        "No text overlays or credits - pure visual narrative.",
        "Maintain consistent characters and setting throughout."
    ]
    
    if title:
        prompt_parts.append(f"Movie title context: {title}")
    
    # Add synopsis context
    prompt_parts.append(f"Story context: {synopsis}")
    
    # Add teaser-specific direction
    prompt_parts.extend([
        "Focus on establishing mood, key characters, and central conflict.",
        "Build tension and intrigue without revealing major plot points.",
        "End with a compelling hook that makes viewers want to see more.",
        "Professional movie trailer quality with cinematic color grading."
    ])
    
    return " ".join(prompt_parts)


def poll_video_completion(request_id: str, fal_api_key: str, model: str, timeout: int) -> Dict[str, Any]:
    """Poll FAL.ai for video completion status"""
    import time
    
    headers = {
        "Authorization": f"Key {fal_api_key}",
    }
    
    status_url = f"https://fal.run/{model}/requests/{request_id}"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(status_url, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status", "unknown")
            
            if status == "completed":
                if "video" in result and "url" in result["video"]:
                    video_url = result["video"]["url"]
                    local_file = download_video_file(video_url, "static/generated")
                    
                    return {
                        "video_url": video_url,
                        "local_file": local_file,
                        "duration": result.get("duration", 8),
                        "model": model,
                        "status": "completed",
                        "request_id": request_id
                    }
                else:
                    raise RuntimeError("Video completed but no URL found in response")
            
            elif status in ["failed", "error"]:
                error_msg = result.get("error", "Unknown error occurred")
                raise RuntimeError(f"Video generation failed: {error_msg}")
            
            elif status in ["queued", "in_progress"]:
                # Wait before polling again
                time.sleep(5)
                continue
            else:
                raise RuntimeError(f"Unknown status: {status}")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error polling video status: {str(e)}")
    
    raise RuntimeError(f"Video generation timed out after {timeout} seconds")


def download_video_file(video_url: str, output_dir: str) -> str:
    """Download video file from URL and save locally"""
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    filename = f"teaser_{uuid.uuid4().hex}.mp4"
    filepath = Path(output_dir) / filename
    
    try:
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return str(filepath)
        
    except Exception as e:
        raise RuntimeError(f"Failed to download video: {str(e)}")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="AI Story Generator with optional video teaser generation")
    parser.add_argument(
        "--model", 
        choices=["veo"], 
        help="Enable specific models: 'veo' enables FAL.ai Veo3 video teaser generation"
    )
    args = parser.parse_args()
    
    enable_veo = args.model == "veo"
    if enable_veo:
        print("🎬 Video teaser generation enabled with FAL.ai Veo3 model")
    else:
        print("📸 Running in image-only mode. Use --model veo to enable video generation")
    
    app = create_app(enable_veo=enable_veo)
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8002")))
