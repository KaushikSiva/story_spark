import os
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple, List

import requests

from image_utils import (
    extract_images_from_genai_response,
    save_images_parts_with_pattern,
    save_image_part_fixed,
)


def generate_motion_poster_script(*, title: str | None, genre: str | None, synopsis: str | None, seed: int | None, base_url: str, model: str, timeout: int, api_key: str, require_local: bool, api_style: str, chat_path: str, completions_path: str) -> tuple[dict[str, Any], str, str | None]:
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
        "temperature": 0.7,
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

    content = (
        data.get("choices", [{}])[0].get("message", {}).get("content")
        if isinstance(data.get("choices", [{}])[0], dict) and data.get("choices", [{}])[0].get("message") is not None
        else data.get("choices", [{}])[0].get("text", "")
    ) or ""

    cleaned = _strip_llm_artifacts(content)
    json_text = _extract_json_object(cleaned)
    if json_text is None:
        # Some servers may return the object directly
        if isinstance(data, dict):
            try:
                if "synopsis" in data and "num_characters" in data:
                    return _normalize_poster_fields(data)
            except Exception:
                pass
        raise ValueError("Unable to parse motion poster JSON from LLM output")

    parsed = json.loads(json_text)
    return _normalize_poster_fields(parsed)


def _normalize_poster_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    title = obj.get("title")
    synopsis = obj.get("synopsis")
    genre = obj.get("genre")
    numc = obj.get("num_characters")
    if synopsis is None and isinstance(obj, dict):
        synopsis = obj.get("scene")

    if not isinstance(title, str) or not title.strip():
        title = "Untitled"
    if not isinstance(synopsis, str):
        synopsis = str(synopsis) if synopsis is not None else "First look reveal"
    if not isinstance(genre, str):
        genre = str(genre) if genre is not None else "Drama"
    try:
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


def _strip_llm_artifacts(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
        s = "\n".join(lines).strip()
    for tag in ["<|channel|>analysis", "<|channel|>final", "<|start|>assistant", "<|end|>"]:
        s = s.replace(tag, "")
    return s.strip()


def _extract_json_object(text: str) -> str | None:
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


def _enforce_synopsis_lines(text: str, *, genre: str | None) -> str:
    import re
    if not text:
        text = "A mysterious event sets quiet lives in motion."
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        parts = re.split(r"(?<=[.!?])\s+", text)
        lines = [p.strip() for p in parts if p.strip()]
    lines = lines[:8]
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
    if len(lines) == 7:
        lines.append("The path forward is clear—and costly.")
    return "\n".join(lines[:8])


def _infer_num_characters(text: str) -> int:
    if not text:
        return 1
    s = text.lower()
    if any(w in s for w in ["crowd", "soldiers", "villagers", "people"]):
        return 5
    if any(w in s for w in ["two ", "duo", "pair", "both "]):
        return 2
    if any(w in s for w in ["three", "trio"]):
        return 3
    return 1


def _fallback_poster(*, title: str | None, genre: str | None, synopsis: str | None, seed: int | None) -> Dict[str, Any]:
    rnd = random.Random(seed)
    genres = ["Action Thriller", "Sci‑Fi", "Fantasy", "Drama", "Mystery", "Horror", "Adventure"]
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
    return {"title": t, "synopsis": synopsis_text, "num_characters": 1, "genre": g}


def generate_images_from_synopsis(*, synopsis: str, model_name: str, out_dir: str, n: int = 8, reference_parts: list | None = None, character_target: int | None = None) -> list[str]:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)
    cfg = genai_types.GenerateContentConfig(response_modalities=["Text", "Image"]) if genai_types else None

    # Prompt (template from motion_poster if provided)
    try:
        from motion_poster import build_images_prompt_from_synopsis as _prompt_tpl
        prompt = _prompt_tpl(_enforce_synopsis_lines(synopsis, genre=None), n, character_target=character_target)
    except Exception:
        prompt = f"Create a {n}-part story collage from: {_enforce_synopsis_lines(synopsis, genre=None)}"

    def _call(model: str, contents):
        return client.models.generate_content(model=model, contents=contents, config=cfg)

    contents = (list(reference_parts) + [prompt]) if reference_parts else prompt
    try:
        response = _call(model_name, contents)
    except Exception as e:
        msg = str(e).lower()
        if any(tok in msg for tok in ["not_found", "404", "not found", "is not supported"]):
            fb = os.environ.get("FALLBACK_IMAGE_MODEL", "models/gemini-2.5-flash-image-preview")
            response = _call(fb, contents)
        else:
            raise

    parts = extract_images_from_genai_response(response)
    saved = []
    next_idx = 1
    if parts:
        batch = save_images_parts_with_pattern(parts[:n], out_dir, pattern="scene_{i}.{ext}", start_index=next_idx)
        saved.extend(batch)
        next_idx += len(batch)
    # If fewer than requested, fall back per-line to reach n
    if len(saved) < n:
        # Per-line retry to salvage at least one
        lines = [ln.strip() for ln in _enforce_synopsis_lines(synopsis, genre=None).splitlines() if ln.strip()]
        for ln in lines:
            if len(saved) >= n:
                break
            single_prompt = (
                "Create one cinematic image illustrating the following story beat. "
                "Do not include any words or text on the image. Maintain consistent characters and setting.\n"
                f"Beat: {ln}"
            )
            contents2 = (list(reference_parts) + [single_prompt]) if reference_parts else single_prompt
            try:
                r = _call(model_name, contents2)
            except Exception:
                fb = os.environ.get("FALLBACK_IMAGE_MODEL", "models/gemini-2.5-flash-image-preview")
                r = _call(fb, contents2)
            more = extract_images_from_genai_response(r)
            if more:
                batch = save_images_parts_with_pattern(more[:1], out_dir, pattern="scene_{i}.{ext}", start_index=next_idx)
                saved.extend(batch)
                next_idx += len(batch)
    if not saved:
        raise RuntimeError("No images returned for synopsis prompt")
    return saved[:n]


def generate_poster_from_synopsis(*, title: str, synopsis: str, director: str, model_name: str, out_dir: str, reference_parts: list | None = None, character_target: int | None = None, fixed_name: str | None = None) -> str | None:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)
    cfg = genai_types.GenerateContentConfig(response_modalities=["Text", "Image"]) if genai_types else None

    try:
        from motion_poster import build_poster_prompt as _poster_tpl
        prompt = _poster_tpl(title, _enforce_synopsis_lines(synopsis, genre=None), director, character_target=character_target)
    except Exception:
        prompt = f"Design a first-look poster for '{title}'.\nSynopsis: {_enforce_synopsis_lines(synopsis, genre=None)}"

    contents = (list(reference_parts) + [prompt]) if reference_parts else prompt

    def _call(model: str, contents):
        return client.models.generate_content(model=model, contents=contents, config=cfg)

    try:
        response = _call(model_name, contents)
    except Exception:
        fb = os.environ.get("FALLBACK_IMAGE_MODEL", "models/gemini-2.5-flash-image-preview")
        try:
            response = _call(fb, contents)
        except Exception:
            return None

    parts = extract_images_from_genai_response(response)
    if not parts:
        return None
    if fixed_name:
        return save_image_part_fixed(parts[0], out_dir=out_dir, base_name=fixed_name)
    saved = save_images_parts(parts[:1], out_dir)
    return saved[0] if saved else None


def get_first_look_path(out_dir: str) -> str | None:
    p = Path(out_dir)
    for name in ("first_look.jpg", "first_look.png"):
        fp = p / name
        if fp.exists() and fp.is_file():
            return str(fp)
    return None
