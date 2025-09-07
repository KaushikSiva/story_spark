import os
import uuid
import base64
import random
import threading
import time
import json
import logging
from pathlib import Path
from typing import List, Dict

from flask import Flask, jsonify, request
from dotenv import load_dotenv
import requests

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
    _GENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    genai = None
    genai_types = None
    _GENAI_AVAILABLE = False


load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    app.config.update(
        OUTPUT_DIR=os.environ.get("OUTPUT_DIR", "static/generated"),
        MODEL=os.environ.get("GENERATION_MODEL", "models/gemini-2.5-flash-image-preview"),
        IMAGE_SIZE=os.environ.get("IMAGE_SIZE", "1024x1024"),
        OUTPUT_TTL_SECONDS=int(os.environ.get("OUTPUT_TTL_SECONDS", str(60 * 60))),
        CORS_ALLOW_ORIGINS=os.environ.get("CORS_ALLOW_ORIGINS", "*"),
        LLM_BASE_URL=os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"),
        LLM_MODEL=os.environ.get("LLM_MODEL", "openai/gpt-oss-20b"),
        LLM_TIMEOUT_SECONDS=int(os.environ.get("LLM_TIMEOUT_SECONDS", "45")),
        LLM_API_KEY=os.environ.get("LLM_API_KEY", ""),
        LLM_API_STYLE=os.environ.get("LLM_API_STYLE", "auto"),  # auto|chat|completions
        LLM_CHAT_PATH=os.environ.get("LLM_CHAT_PATH", "/chat/completions"),
        LLM_COMPLETIONS_PATH=os.environ.get("LLM_COMPLETIONS_PATH", "/completions"),
    )

    Path(app.config["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    @app.after_request
    def add_cors_headers(resp):  # pragma: no cover
        resp.headers["Access-Control-Allow-Origin"] = app.config["CORS_ALLOW_ORIGINS"]
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp

    @app.route("/health2", methods=["GET"])  # health for app2
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/storyboard", methods=["GET", "OPTIONS"])
    def storyboard():
        if request.method == "OPTIONS":
            return ("", 204)

        theme = (request.args.get("theme") or "").strip() or None
        style = (request.args.get("style") or "").strip() or "colorful, soft, friendly, picture book illustration"
        seed_param = request.args.get("seed")
        try:
            seed = int(seed_param) if seed_param is not None else None
        except Exception:
            seed = None
        ttl_param = request.args.get("ttl_seconds")
        try:
            ttl_seconds = int(ttl_param) if ttl_param is not None else int(app.config["OUTPUT_TTL_SECONDS"])
        except Exception:
            ttl_seconds = int(app.config["OUTPUT_TTL_SECONDS"])
        ttl_seconds = max(10, ttl_seconds)

        require_local = (request.args.get("require_local") or "").lower() in {"1", "true", "yes"}

        api_style = (request.args.get("api_style") or app.config["LLM_API_STYLE"]).lower()

        story, story_source, llm_error = generate_kid_story(
            theme=theme,
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

        # Log the story scenes before image generation
        try:
            scene_texts = [s.get("text", "") for s in story]
            log.info("storyboard source=%s theme=%s scenes=%s", story_source, theme, json.dumps(scene_texts, ensure_ascii=False))
            if llm_error:
                log.warning("storyboard llm_error=%s", llm_error)
        except Exception:
            pass

        try:
            files = generate_images_for_story(
                story=story,
                style=style,
                model_name=app.config["MODEL"],
                out_dir=app.config["OUTPUT_DIR"],
            )
        except Exception as e:  # pragma: no cover
            return jsonify({"error": "Image generation failed", "details": str(e)}), 500

        base_url = request.url_root.rstrip("/")
        scenes = []
        for idx, scene in enumerate(story):
            filename = Path(files[idx]).name
            url = f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{filename}"
            scenes.append({
                "index": idx + 1,
                "text": scene["text"],
                "image_url": url,
                "filename": filename,
            })

        _schedule_cleanup(files, ttl_seconds)

        resp_payload = {
            "count": len(scenes),
            "model": app.config["MODEL"],
            "size": app.config["IMAGE_SIZE"],
            "style": style,
            "ttl_seconds": ttl_seconds,
            "scenes": scenes,
            "story_source": story_source,
        }
        if llm_error:
            resp_payload["llm_error"] = llm_error
        return jsonify(resp_payload)

    return app


def generate_kid_story(theme: str | None = None, seed: int | None = None, *, base_url: str, model: str, timeout: int, api_key: str, require_local: bool, api_style: str, chat_path: str, completions_path: str) -> tuple[list[dict[str, str]], str, str | None]:
    last_err: str | None = None
    try:
        story = _story_from_local_gpt(
            theme=theme,
            seed=seed,
            base_url=base_url,
            model=model,
            timeout=timeout,
            api_key=api_key,
            api_style=api_style,
            chat_path=chat_path,
            completions_path=completions_path,
        )
        return story, "local_gpt", None
    except Exception as e:
        last_err = str(e)
        if require_local:
            raise
    fallback = _fallback_story(theme=theme, seed=seed)
    return fallback, "fallback", last_err


def _story_from_local_gpt(*, theme: str | None, seed: int | None, base_url: str, model: str, timeout: int, api_key: str, api_style: str, chat_path: str, completions_path: str) -> List[Dict[str, str]]:
    chat_url = base_url.rstrip("/") + chat_path
    comp_url = base_url.rstrip("/") + completions_path
    sys_prompt = (
        "You are a children’s storyteller. Create a very short, gentle 5-scene story suitable for ages 4–7. "
        "Return ONLY a valid JSON object with this exact shape and nothing else: "
        "{\"scenes\":[{\"text\":\"...\"},{\"text\":\"...\"},{\"text\":\"...\"},{\"text\":\"...\"},{\"text\":\"...\"}]}. "
        "Each scene is one simple sentence. Avoid proper nouns; keep it kind and positive."
    )
    theme_prompt = f"Theme: {theme}." if theme else "Choose a sweet, simple theme."
    user_prompt = f"Create the storyboard. {theme_prompt}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 300,
        # Many OpenAI-compatible servers accept these; ignore if unsupported
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
    if json_text is not None:
        try:
            parsed = json.loads(json_text)
            scenes = parsed.get("scenes", [])
            return _normalize_scenes(scenes)
        except Exception:
            pass

    # Fallback: try to extract scenes array from messy text
    scenes = _recover_scenes_from_text(cleaned)
    if scenes:
        return _normalize_scenes(scenes)

    raise ValueError("Unable to parse storyboard JSON from LLM output")


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


def _recover_scenes_from_text(text: str):
    # Attempt to pull the scenes array even if braces are mismatched
    import re
    m = re.search(r'"scenes"\s*:\s*\[(.*)\]', text, re.DOTALL)
    if not m:
        return None
    inner = m.group(1)
    # Split on '},{' boundaries as a heuristic
    parts = [p.strip() for p in inner.split('},{') if p.strip()]
    scenes = []
    for p in parts:
        frag = p
        if not frag.startswith('{'):
            frag = '{' + frag
        if not frag.endswith('}'):
            frag = frag + '}'
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict) and 'text' in obj:
                scenes.append(obj)
                continue
        except Exception:
            pass
        # Regex-extract text if JSON fails
        tm = re.search(r'"text"\s*:\s*"(.*?)"', frag, re.DOTALL)
        if tm:
            scenes.append({"text": tm.group(1)})
    return scenes if scenes else None


def _normalize_scenes(scenes) -> List[Dict[str, str]]:
    norm: List[Dict[str, str]] = []
    for s in scenes[:5]:
        if isinstance(s, dict) and "text" in s:
            norm.append({"text": str(s["text"])})
        elif isinstance(s, str):
            norm.append({"text": s})
    while len(norm) < 5:
        norm.append({"text": "A calm moment passes."})
    return norm[:5]


def _fallback_story(theme: str | None, seed: int | None) -> List[Dict[str, str]]:
    rnd = random.Random(seed)
    characters = ["a curious fox", "a brave kitten", "a clever bunny", "a gentle dragon", "a friendly robot", "a tiny turtle"]
    settings = ["a sunny forest", "a cozy village", "a sandy beach", "a snowy mountain", "a colorful garden", "a bright library"]
    goals = ["find a lost key", "help a friend smile", "discover a hidden path", "fix a squeaky bridge", "bring water to flowers", "learn a new song"]
    obstacles = ["a wobbly log", "a tricky riddle", "a dark tunnel", "a tall fence", "a fast river", "a gusty wind"]
    helpers = ["a wise owl", "a giggly squirrel", "a singing bird", "a kind turtle", "a helpful bee", "a cheerful frog"]
    morals = ["Kindness makes us strong.", "Teamwork lights the way.", "Be brave and gentle.", "Small steps make big journeys.", "Sharing brings joy.", "Listening opens doors."]

    who = rnd.choice(characters)
    where = rnd.choice(settings)
    goal = theme if theme else rnd.choice(goals)
    obstacle = rnd.choice(obstacles)
    helper = rnd.choice(helpers)
    moral = rnd.choice(morals)

    scenes = [
        {"text": f"In {where}, {who} dreamed to {goal}."},
        {"text": f"On the path, {who} met {helper} and learned about {obstacle}."},
        {"text": f"Together they tried a gentle idea to pass the {obstacle}."},
        {"text": f"With patience and giggles, they solved it and reached the goal to {goal}."},
        {"text": f"They celebrated with a warm hug. The lesson: {moral}"},
    ]
    return scenes


def generate_images_for_story(story: List[Dict[str, str]], style: str, model_name: str, out_dir: str) -> List[str]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google-genai is not installed")

    client = genai.Client(api_key=api_key)
    cfg = genai_types.GenerateContentConfig(response_modalities=["Text", "Image"]) if genai_types else None

    saved_files: List[str] = []
    for idx, scene in enumerate(story):
        prompt = build_scene_prompt(scene["text"], style)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=cfg,
        )
        parts = _extract_images_from_genai_response(response)
        if not parts:
            continue
        saved_files.extend(_save_images_parts(parts[:1], out_dir))

    if len(saved_files) < len(story):
        # If fewer images than scenes, raise to signal partial failure
        raise RuntimeError(f"Generated {len(saved_files)} images for {len(story)} scenes")

    return saved_files


def build_scene_prompt(text: str, style: str) -> str:
    return (
        f"Illustrate the following kid-friendly story moment as one picture: {text} "
        f"Use a {style}. Friendly faces, clear actions, warm colors, high coherence."
    )


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
    saved = []
    for p in parts:
        data = p.get("data")
        mime = p.get("mime") or "image/png"
        if isinstance(data, str):
            try:
                data = base64.b64decode(data)
            except Exception:
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


if __name__ == "__main__":  # pragma: no cover
    app = create_app()
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8001")))
