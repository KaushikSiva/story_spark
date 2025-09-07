import os
import json
import logging
import random
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request
from dotenv import load_dotenv
import requests


load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Reuse the same LLM config style as app2.py
    app.config.update(
        CORS_ALLOW_ORIGINS=os.environ.get("CORS_ALLOW_ORIGINS", "*"),
        LLM_BASE_URL=os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"),
        LLM_MODEL=os.environ.get("LLM_MODEL", "openai/gpt-oss-20b"),
        LLM_TIMEOUT_SECONDS=int(os.environ.get("LLM_TIMEOUT_SECONDS", "45")),
        LLM_API_KEY=os.environ.get("LLM_API_KEY", ""),
        LLM_API_STYLE=os.environ.get("LLM_API_STYLE", "auto"),  # auto|chat|completions
        LLM_CHAT_PATH=os.environ.get("LLM_CHAT_PATH", "/chat/completions"),
        LLM_COMPLETIONS_PATH=os.environ.get("LLM_COMPLETIONS_PATH", "/completions"),
    )

    @app.after_request
    def add_cors_headers(resp):  # pragma: no cover
        resp.headers["Access-Control-Allow-Origin"] = app.config["CORS_ALLOW_ORIGINS"]
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp

    @app.route("/health3", methods=["GET"])  # health for app3
    def health():
        return jsonify({"status": "ok"})

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

        log.info(
            "motion_poster source=%s title=%s genre=%s synopsis=%s",
            source,
            title,
            result.get("genre"),
            result.get("synopsis"),
        )
        if llm_error:
            log.warning("motion_poster llm_error=%s", llm_error)

        payload = {
            "title": result.get("title", title or "Untitled"),
            "synopsis": result.get("synopsis", ""),
            "num_characters": int(result.get("num_characters", 1) or 1),
            "genre": result.get("genre", ""),
            "source": source,
        }
        if llm_error:
            payload["llm_error"] = llm_error
        return jsonify(payload)

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

    return {
        "title": title.strip() or "Untitled",
        "synopsis": synopsis.strip() or "First look reveal",
        "num_characters": numc_int,
        "genre": genre.strip() or "Drama",
    }


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


if __name__ == "__main__":  # pragma: no cover
    app = create_app()
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8002")))
