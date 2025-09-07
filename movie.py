import os
import logging
from pathlib import Path
import re
import uuid
from datetime import datetime
from flask import Flask, jsonify, request, redirect
from dotenv import load_dotenv

from image_utils import stitch_images_grid, load_reference_image_parts, file_to_genai_part
from services.motion_poster_service import (
    generate_motion_poster_script,
    generate_images_from_synopsis,
    generate_poster_from_synopsis,
    get_first_look_path,
)
from services.video_teaser_service import generate_video_teaser
from services.youtube_upload import upload_video as youtube_upload_video, YouTubeUploadError
from services.video_postprocess import concat_videos, concat_videos_many, ensure_ffmpeg
import json
import secrets
import requests
from services.video_teaser_service import append_poster_and_tag
import sys
import subprocess


load_dotenv()


def create_app(*, enable_veo: bool = False) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    app.config.update(
        OUTPUT_DIR=os.environ.get("OUTPUT_DIR", "static/generated"),
        CORS_ALLOW_ORIGINS=os.environ.get("CORS_ALLOW_ORIGINS", "*"),
        # LLM
        LLM_BASE_URL=os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1"),
        LLM_MODEL=os.environ.get("LLM_MODEL", "openai/gpt-oss-20b"),
        LLM_TIMEOUT_SECONDS=int(os.environ.get("LLM_TIMEOUT_SECONDS", "45")),
        LLM_API_KEY=os.environ.get("LLM_API_KEY", ""),
        LLM_API_STYLE=os.environ.get("LLM_API_STYLE", "auto"),
        LLM_CHAT_PATH=os.environ.get("LLM_CHAT_PATH", "/chat/completions"),
        LLM_COMPLETIONS_PATH=os.environ.get("LLM_COMPLETIONS_PATH", "/completions"),
        # Images
        IMAGE_STORY_MODEL=os.environ.get("IMAGE_STORY_MODEL", "models/gemini-2.5-flash-image-preview"),
        FALLBACK_IMAGE_MODEL=os.environ.get("FALLBACK_IMAGE_MODEL", "models/gemini-2.5-flash-image-preview"),
        PEOPLE_DIR=os.environ.get("PEOPLE_DIR", "static/people"),
        # Video
        ENABLE_VEO=bool(enable_veo),
        FAL_API_KEY=os.environ.get("FAL_API_KEY", ""),
        FAL_VEO3_MODEL=os.environ.get("FAL_VEO3_MODEL", "fal-ai/veo3/image-to-video"),
        VIDEO_TIMEOUT_SECONDS=int(os.environ.get("VIDEO_TIMEOUT_SECONDS", "300")),
        # YouTube OAuth (optional; enables server-managed upload)
        YT_CLIENT_ID=os.environ.get("YT_CLIENT_ID", ""),
        YT_CLIENT_SECRET=os.environ.get("YT_CLIENT_SECRET", ""),
        YT_REDIRECT_URI=os.environ.get("YT_REDIRECT_URI", "http://localhost:8002/api/youtube/callback"),
        YT_TOKEN_STORE=os.environ.get("YT_TOKEN_STORE", ".tokens/youtube.json"),
    )

    Path(app.config["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    @app.after_request
    def add_cors_headers(resp):  # pragma: no cover
        resp.headers["Access-Control-Allow-Origin"] = app.config["CORS_ALLOW_ORIGINS"]
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp

    @app.route("/health", methods=["GET"])  # simple health check
    def health():
        return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})

    # Simple UI routes
    @app.route("/", methods=["GET"])  # serve UI
    @app.route("/ui", methods=["GET"])  # alias
    def ui_root():  # pragma: no cover
        try:
            return app.send_static_file("ui/index.html")
        except Exception:
            return "UI not found. Ensure static/ui/index.html exists.", 404

    # Silence browser favicon requests to avoid noisy 404s in the console
    @app.route("/favicon.ico", methods=["GET"])  # pragma: no cover
    def favicon():
        return ("", 204)

    @app.route("/api/motion_poster", methods=["GET", "OPTIONS"])
    def motion_poster():
        if request.method == "OPTIONS":
            return ("", 204)

        title = (request.args.get("title") or request.args.get("movie") or "").strip() or None
        genre = (request.args.get("genre") or "").strip() or None
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

        final_title_for_log = (result.get("title") or title or "Untitled")
        log.info("motion_poster source=%s title=%s genre=%s synopsis=%s", source, final_title_for_log, result.get("genre"), result.get("synopsis"))
        if llm_error:
            log.warning("motion_poster llm_error=%s", llm_error)

        payload = {
            "title": (result.get("title") or title or "Untitled"),
            "synopsis": result.get("synopsis", ""),
            "num_characters": int(result.get("num_characters", 1) or 1),
            "genre": result.get("genre", ""),
            "source": source,
        }

        # Images default on; disable with generate_images=0/false/no
        flag_raw = (request.args.get("generate_images") or request.args.get("images"))
        gen_images_flag = True if flag_raw is None else (str(flag_raw).lower() not in {"0", "false", "no"})
        if gen_images_flag:
            try:
                n_param = request.args.get("n")
                n = int(n_param) if n_param is not None else 8
                n = max(1, min(12, n))
            except Exception:
                n = 8

            # Character references from generated JSON
            character_target = int(payload.get("num_characters", 1) or 1)
            ref_parts = load_reference_image_parts(people_dir=app.config["PEOPLE_DIR"], k=character_target + 4)
            log.info("character_target=%s; reference_count=%s", character_target, len(ref_parts))

            # Per-generation output directory named after title (slug)
            def _slugify(title_str: str) -> str:
                s = (title_str or "Untitled").lower().strip()
                s = re.sub(r"[^a-z0-9]+", "-", s)
                s = re.sub(r"-+", "-", s).strip("-")
                return s or "untitled"

            base_slug = _slugify(payload.get("title") or "Untitled")
            base_out = Path(app.config["OUTPUT_DIR"]) / base_slug
            run_path = base_out
            # If folder exists, override by clearing it
            if run_path.exists():
                try:
                    import shutil
                    shutil.rmtree(run_path)
                except Exception:
                    pass
            run_path.mkdir(parents=True, exist_ok=True)
            run_dir = run_path.name
            out_dir = str(run_path)
            payload["run_dir"] = run_dir

            try:
                files = generate_images_from_synopsis(
                    synopsis=payload["synopsis"],
                    model_name=app.config["IMAGE_STORY_MODEL"],
                    out_dir=out_dir,
                    n=n,
                    reference_parts=ref_parts,
                    character_target=character_target,
                )
                log.info("saved scenes to out_dir=%s files=%s", out_dir, [Path(f).name for f in files])
                base_url = request.url_root.rstrip("/")
                payload["images"] = []
                for f in files:
                    rel = str(Path(f).relative_to(app.config["OUTPUT_DIR"]))
                    payload["images"].append({
                        "url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{rel}",
                        "filename": rel,
                    })

                # stitched (fixed name)
                stitched = stitch_images_grid(files, out_dir=out_dir, tile_size=512, fixed_name="stitched.jpg")
                if stitched:
                    rels = str(Path(stitched).relative_to(app.config["OUTPUT_DIR"]))
                    payload["stitched_image"] = {"url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{rels}", "filename": rels}
                    log.info("stitched collage saved to %s", rels)

                # poster: use only PEOPLE_DIR references (avoid biasing toward any single scene backdrop)
                strong_refs = list(ref_parts) if ref_parts else []
                poster_file = generate_poster_from_synopsis(
                    title=(payload.get("title") or "Untitled"),
                    synopsis=payload["synopsis"],
                    director="Bruno",
                    model_name=app.config["IMAGE_STORY_MODEL"],
                    out_dir=out_dir,
                    reference_parts=strong_refs,
                    character_target=character_target,
                    fixed_name="first_look",
                )
                if poster_file:
                    relp = str(Path(poster_file).relative_to(app.config["OUTPUT_DIR"]))
                    payload["poster_image"] = {"url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{relp}", "filename": relp}
                    log.info("poster saved to %s", relp)
            except Exception as e:  # pragma: no cover
                log.warning("image generation failed: %s", e)

        # Cache
        try:
            setattr(app, "_last_motion_poster_cache", dict(payload))
        except Exception:
            pass
        return jsonify(payload)

    @app.route("/api/video_teaser", methods=["GET", "OPTIONS"])
    def video_teaser_route():
        if request.method == "OPTIONS":
            return ("", 204)
        if not app.config.get("ENABLE_VEO"):
            return jsonify({"error": "Video teaser generation is disabled", "details": "Set ENABLE_VEO=1 (default) to enable video features."}), 503

        title = (request.args.get("title") or "").strip() or None
        synopsis = (request.args.get("synopsis") or "").strip() or None
        stitched_image_url = (request.args.get("stitched_image_url") or "").strip() or None
        run_dir_param = (request.args.get("run_dir") or "").strip() or None
        duration = 8  # enforced by service/model

        # Pull from cache if missing
        last_payload = getattr(app, "_last_motion_poster_cache", None)
        if not synopsis and last_payload:
            synopsis = last_payload.get("synopsis")
            title = title or last_payload.get("title")
        if not stitched_image_url:
            # Resolve from run_dir/title: prefer stitched, else first_look
            base_out = Path(app.config.get("OUTPUT_DIR", "static/generated"))
            # Determine run_dir in order: param -> last -> slug(title)
            run_dir = run_dir_param or (last_payload.get("run_dir") if last_payload else None)
            if not run_dir and title:
                run_dir = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "untitled"
            if run_dir:
                folder = base_out / run_dir
                if folder.exists():
                    base_url = request.url_root.rstrip("/")
                    stitched_path = folder / "stitched.jpg"
                    first_look_jpg = folder / "first_look.jpg"
                    first_look_png = folder / "first_look.png"
                    if stitched_path.exists():
                        rel = str(stitched_path.relative_to(base_out))
                        stitched_image_url = f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{rel}"
                    elif first_look_jpg.exists() or first_look_png.exists():
                        p = first_look_jpg if first_look_jpg.exists() else first_look_png
                        rel = str(p.relative_to(base_out))
                        stitched_image_url = f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{rel}"

        # If no synopsis at this point, generate a lightweight one
        if not synopsis:
            try:
                res, _src, _err = generate_motion_poster_script(
                    title=title,
                    genre=None,
                    synopsis=None,
                    seed=None,
                    base_url=app.config["LLM_BASE_URL"],
                    model=app.config["LLM_MODEL"],
                    timeout=app.config["LLM_TIMEOUT_SECONDS"],
                    api_key=app.config["LLM_API_KEY"],
                    require_local=False,
                    api_style=app.config["LLM_API_STYLE"],
                    chat_path=app.config["LLM_CHAT_PATH"],
                    completions_path=app.config["LLM_COMPLETIONS_PATH"],
                )
                synopsis = res.get("synopsis") or synopsis
                title = title or res.get("title")
            except Exception:
                pass
        if not synopsis:
            return jsonify({"error": "synopsis parameter is required", "details": "Unable to infer synopsis"}), 400
        if not stitched_image_url:
            return jsonify({"error": "image missing", "details": "Provide stitched_image_url or ensure folder has stitched.jpg or first_look"}), 400

        try:
            # Determine output dir: prefer explicit run_dir param, else cached, else derive from stitched URL, else base
            base_out = Path(app.config.get("OUTPUT_DIR", "static/generated"))
            run_dir_for_out = (request.args.get("run_dir") or "").strip() or (last_payload.get("run_dir") if last_payload else None)
            if not run_dir_for_out and stitched_image_url:
                # Try to derive run_dir from stitched image URL: /static/generated/<run_dir>/stitched.jpg
                try:
                    marker = "/static/generated/"
                    if marker in stitched_image_url:
                        tail = stitched_image_url.split(marker, 1)[1]
                        run_dir_for_out = tail.split("/", 1)[0]
                except Exception:
                    run_dir_for_out = None
            out_dir = str(base_out / run_dir_for_out) if run_dir_for_out else str(base_out)
            video_result = generate_video_teaser(
                title=title,
                synopsis=synopsis,
                image_url=stitched_image_url,
                duration=duration,
                fal_api_key=app.config["FAL_API_KEY"],
                model=app.config["FAL_VEO3_MODEL"],
                timeout=app.config["VIDEO_TIMEOUT_SECONDS"],
                output_dir=out_dir,
                style=(request.args.get("style") or request.args.get("reference") or None),
                append_extras=((request.args.get("append_extras") or "1").lower() not in {"0","false","no"}),
            )
            # attach convenience URL
            base_url = request.url_root.rstrip("/")
            if video_result.get("local_file"):
                rel = str(Path(video_result["local_file"]).relative_to(app.config["OUTPUT_DIR"]))
                video_result["url"] = f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{rel}"
            # Surface whether postprocessing ran
            if "postprocessed" not in video_result:
                video_result["postprocessed"] = False
            video_result["stitched_image_url"] = stitched_image_url
            return jsonify(video_result)
        except Exception as e:
            logging.getLogger(__name__).error("Video teaser generation failed: %s", e)
            return jsonify({"error": "Video generation failed", "details": str(e)}), 500

    @app.route("/api/resync_with_poster", methods=["POST", "GET", "OPTIONS"])
    def resync_with_poster():
        if request.method == "OPTIONS":
            return ("", 204)
        last_payload = getattr(app, "_last_motion_poster_cache", None)
        if not last_payload:
            return jsonify({"error": "No cached result", "details": "Run /api/motion_poster first to create poster and stitched image."}), 400

        synopsis = last_payload.get("synopsis") or (request.args.get("synopsis") or "").strip()
        if not synopsis:
            return jsonify({"error": "synopsis required", "details": "No cached synopsis found."}), 400

        base_out = app.config.get("OUTPUT_DIR", "static/generated")
        run_dir = last_payload.get("run_dir")
        out_dir = str(Path(base_out) / run_dir) if run_dir else base_out
        poster_path = get_first_look_path(out_dir)
        if not poster_path:
            return jsonify({"error": "poster not found", "details": "Generate a poster first via /api/motion_poster."}), 400

        n_param = request.args.get("n")
        try:
            n = int(n_param) if n_param is not None else len(last_payload.get("images", [])) or 8
        except Exception:
            n = 8
        n = max(1, min(12, n))

        refs = []
        poster_part = file_to_genai_part(poster_path)
        if poster_part is not None:
            refs.append(poster_part)
        people_refs = load_reference_image_parts(people_dir=app.config.get("PEOPLE_DIR", "static/people"), k=4)
        if people_refs:
            refs.extend(people_refs)

        try:
            character_target = int(last_payload.get("num_characters", 1) or 1)
        except Exception:
            character_target = 1

        try:
            files = generate_images_from_synopsis(
                synopsis=synopsis,
                model_name=app.config.get("IMAGE_STORY_MODEL", "models/gemini-2.5-flash-image-preview"),
                out_dir=out_dir,
                n=n,
                reference_parts=refs,
                character_target=character_target,
            )
            base_url = request.url_root.rstrip("/")
            images = []
            for f in files:
                rel = str(Path(f).relative_to(app.config["OUTPUT_DIR"]))
                images.append({"url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{rel}", "filename": rel})
            stitched = stitch_images_grid(files, out_dir=out_dir, tile_size=512, fixed_name="stitched.jpg")
            stitched_info = None
            if stitched:
                rels = str(Path(stitched).relative_to(app.config["OUTPUT_DIR"]))
                stitched_info = {"url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{rels}", "filename": rels}
            new_payload = dict(last_payload)
            new_payload["run_dir"] = run_dir or new_payload.get("run_dir")
            new_payload["images"] = images
            if stitched_info:
                new_payload["stitched_image"] = stitched_info
            setattr(app, "_last_motion_poster_cache", new_payload)
            return jsonify(new_payload)
        except Exception as e:
            logging.getLogger(__name__).error("resync_with_poster failed: %s", e)
            return jsonify({"error": "resync_failed", "details": str(e)}), 500

    @app.route("/api/postprocess_video", methods=["POST", "OPTIONS"])
    def postprocess_video():
        if request.method == "OPTIONS":
            return ("", 204)
        if "file" not in request.files:
            return jsonify({"error": "no_file", "details": "Upload a video via form field 'file'"}), 400

        f = request.files["file"]
        title_override = (request.form.get("title") or "").strip() or None
        run_dir_param = (request.form.get("run_dir") or "").strip() or None

        # Determine output dir in title folder if available
        last_payload = getattr(app, "_last_motion_poster_cache", None)
        base_out = Path(app.config.get("OUTPUT_DIR", "static/generated"))
        run_dir = None
        if run_dir_param:
            run_dir = run_dir_param
        elif title_override:
            slug = re.sub(r"[^a-z0-9]+", "-", title_override.lower()).strip("-") or "untitled"
            run_dir = slug
        elif last_payload and last_payload.get("run_dir"):
            run_dir = last_payload.get("run_dir")
        else:
            run_dir = "uploads"

        out_dir = base_out / run_dir
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Save uploaded as upload.mp4 (overwrite)
        upload_path = out_dir / "upload.mp4"
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass
        f.save(str(upload_path))
        app.logger.info("postprocess: saved upload to %s", str(upload_path))

        try:
            final_path = append_poster_and_tag(str(upload_path), str(out_dir))
            postprocessed = Path(final_path).exists() and Path(final_path).name == "teaser_with_credits.mp4"
        except Exception as e:
            return jsonify({"error": "postprocess_failed", "details": str(e)}), 500

        base_url = request.url_root.rstrip("/")
        rel = str(Path(final_path).relative_to(app.config["OUTPUT_DIR"])) if Path(final_path).exists() else str(upload_path.relative_to(app.config["OUTPUT_DIR"]))
        app.logger.info("postprocess: result postprocessed=%s final=%s", postprocessed, str(final_path))
        return jsonify({
            "url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{rel}",
            "local_file": str(final_path),
            "run_dir": run_dir,
            "postprocessed": postprocessed,
        })

    @app.route("/api/diag", methods=["GET"])  # pragma: no cover
    def diag():
        """Return environment diagnostics to help debug video tooling issues."""
        info = {}
        info["python"] = sys.executable
        info["output_dir"] = app.config.get("OUTPUT_DIR")
        # Versions and module paths
        def _ver(modname):
            try:
                m = __import__(modname)
                v = getattr(m, "__version__", "")
                p = getattr(m, "__file__", "")
                return {"version": v, "path": p}
            except Exception as e:
                return {"error": str(e)}
        info["moviepy"] = _ver("moviepy")
        info["PIL"] = _ver("PIL")
        try:
            import imageio_ffmpeg  # type: ignore
            info["imageio_ffmpeg_exe"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            info["imageio_ffmpeg_exe"] = {"error": str(e)}
        # ffmpeg -version
        try:
            exe = os.environ.get("IMAGEIO_FFMPEG_EXE") or "ffmpeg"
            out = subprocess.check_output([exe, "-version"], stderr=subprocess.STDOUT, timeout=5)
            info["ffmpeg_version"] = out.decode("utf-8", errors="ignore").splitlines()[:2]
        except Exception as e:
            info["ffmpeg_version"] = {"error": str(e)}
        # Optional: folder probe
        run_dir = (request.args.get("run_dir") or "").strip()
        if run_dir:
            folder = Path(app.config.get("OUTPUT_DIR", "static/generated")) / run_dir
            info["probe_folder"] = str(folder)
            info["files_present"] = [p.name for p in folder.iterdir()] if folder.exists() else []
        return jsonify(info)

    # --- YouTube OAuth helpers ---
    def _yt_tokens_load():
        try:
            p = Path(app.config["YT_TOKEN_STORE"]).expanduser()
            if p.exists():
                return json.loads(p.read_text())
        except Exception:
            pass
        return {}

    def _yt_tokens_save(data):
        try:
            p = Path(app.config["YT_TOKEN_STORE"]).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data))
            return True
        except Exception:
            return False

    @app.route("/api/youtube/status", methods=["GET"])  # Do we have refresh token?
    def youtube_status():
        toks = _yt_tokens_load()
        ok = bool(toks.get("refresh_token"))
        return jsonify({"authorized": ok})

    @app.route("/api/youtube/auth", methods=["GET"])  # Start OAuth
    def youtube_auth_start():
        cid = app.config.get("YT_CLIENT_ID")
        redirect_uri = app.config.get("YT_REDIRECT_URI")
        if not cid:
            return "YouTube OAuth not configured: set YT_CLIENT_ID and YT_CLIENT_SECRET", 500
        state = secrets.token_urlsafe(16)
        try:
            setattr(app, "_yt_oauth_state", state)
        except Exception:
            pass
        scope = "https://www.googleapis.com/auth/youtube.upload"
        url = (
            "https://accounts.google.com/o/oauth2/v2/auth"
            f"?response_type=code&client_id={requests.utils.quote(cid)}"
            f"&redirect_uri={requests.utils.quote(redirect_uri)}"
            f"&scope={requests.utils.quote(scope)}&access_type=offline&include_granted_scopes=true&prompt=consent"
            f"&state={state}"
        )
        return redirect(url, code=302)

    @app.route("/api/youtube/callback", methods=["GET"])  # OAuth redirect URI
    def youtube_auth_callback():
        code = request.args.get("code")
        state = request.args.get("state")
        expect = getattr(app, "_yt_oauth_state", None)
        # Basic state check (best-effort in local dev)
        if expect and state and expect != state:
            return "State mismatch", 400
        cid = app.config.get("YT_CLIENT_ID")
        csecret = app.config.get("YT_CLIENT_SECRET")
        redirect_uri = app.config.get("YT_REDIRECT_URI")
        if not (cid and csecret and code):
            return "Missing OAuth parameters; check YT_CLIENT_ID/SECRET and try again.", 400
        try:
            token_resp = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": cid,
                    "client_secret": csecret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
                timeout=30,
            )
            token_resp.raise_for_status()
            data = token_resp.json()
            # Persist refresh token for reuse
            store = _yt_tokens_load()
            for k in ("refresh_token", "access_token", "scope", "token_type", "expiry", "expires_in"):
                if k in data:
                    store[k] = data[k]
            _yt_tokens_save(store)
            # Redirect back to root UI
            return redirect("/", code=302)
        except Exception as e:
            return f"OAuth token exchange failed: {e}", 500

    def _yt_mint_access_token_from_request(req) -> str | None:
        # Prefer explicit Authorization header
        auth = req.headers.get("Authorization", "").strip()
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1].strip()
        # Else attempt refresh token flow
        toks = _yt_tokens_load()
        rtok = toks.get("refresh_token")
        cid = app.config.get("YT_CLIENT_ID")
        csecret = app.config.get("YT_CLIENT_SECRET")
        if rtok and cid and csecret:
            try:
                r = requests.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": rtok,
                        "client_id": cid,
                        "client_secret": csecret,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                jd = r.json()
                atok = jd.get("access_token")
                if atok:
                    toks.update({k: jd[k] for k in jd if k in ("access_token", "expires_in", "scope", "token_type")})
                    _yt_tokens_save(toks)
                    return atok
            except Exception:
                return None
        return None

    @app.route("/api/upload_youtube", methods=["POST", "OPTIONS"])  # upload local teaser to YouTube
    def upload_youtube():
        if request.method == "OPTIONS":
            return ("", 204)
        data = request.get_json(silent=True) or {}
        access_token = _yt_mint_access_token_from_request(request)
        if not access_token:
            return jsonify({"error": "missing_token", "details": "Not authorized for YouTube. Visit /api/youtube/auth to grant access."}), 401

        # Resolve local file path; prefer local_file under OUTPUT_DIR; else allow URL download
        local_file = (data.get("local_file") or "").strip()
        url = (data.get("url") or "").strip()
        title = (data.get("title") or "").strip() or "Story Spark Teaser"
        description = (data.get("description") or "").strip() or "Created with Story Spark"
        privacy = (data.get("privacyStatus") or data.get("privacy_status") or "").strip() or "unlisted"

        # If only URL provided but we have a local path on server, prefer it for stability
        if (not local_file) and url:
            # Attempt to map known /static/generated/... URL to a local path
            try:
                marker = f"{app.static_url_path.rstrip('/')}/generated/"
                base_out = Path(app.config.get("OUTPUT_DIR", "static/generated"))
                if marker in url:
                    rel = url.split(marker, 1)[1]
                    candidate = base_out / rel
                    if candidate.exists():
                        local_file = str(candidate)
            except Exception:
                pass

        if not local_file:
            return jsonify({"error": "missing_file", "details": "Provide local_file (preferred) or a mapped /static/generated URL"}), 400
        try:
            result = youtube_upload_video(
                access_token=access_token,
                file_path=local_file,
                title=title,
                description=description,
                privacy_status=privacy,
            )
            return jsonify({
                "status": "uploaded",
                "id": result.get("id"),
                "link": result.get("link"),
                "video": result.get("video"),
            })
        except YouTubeUploadError as e:
            return jsonify({"error": "upload_failed", "details": str(e)}), 500

    @app.route("/api/upload_youtube_file", methods=["POST", "OPTIONS"])  # upload a user-selected file to YouTube
    def upload_youtube_file():
        if request.method == "OPTIONS":
            return ("", 204)
        access_token = _yt_mint_access_token_from_request(request)
        if not access_token:
            return jsonify({"error": "missing_token", "details": "Not authorized for YouTube. Visit /api/youtube/auth to grant access."}), 401
        if "file" not in request.files:
            return jsonify({"error": "no_file", "details": "Upload a video via form field 'file'"}), 400
        f = request.files["file"]
        title = (request.form.get("title") or "").strip() or f.filename or "Story Spark Upload"
        description = (request.form.get("description") or "").strip() or "Uploaded with Story Spark"
        privacy = (request.form.get("privacyStatus") or request.form.get("privacy_status") or "").strip() or "unlisted"

        # Save to temp under OUTPUT_DIR/uploads
        base_out = Path(app.config.get("OUTPUT_DIR", "static/generated"))
        out_dir = base_out / "uploads"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_dir / ("upload_" + uuid.uuid4().hex + ".mp4")
        f.save(str(tmp_path))
        try:
            result = youtube_upload_video(
                access_token=access_token,
                file_path=str(tmp_path),
                title=title,
                description=description,
                privacy_status=privacy,
            )
            return jsonify({
                "status": "uploaded",
                "id": result.get("id"),
                "link": result.get("link"),
                "video": result.get("video"),
            })
        except YouTubeUploadError as e:
            return jsonify({"error": "upload_failed", "details": str(e)}), 500
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @app.route("/api/concat_videos", methods=["POST", "OPTIONS"])  # stitch uploaded videos (2+)
    def concat_videos_route():
        if request.method == "OPTIONS":
            return ("", 204)
        files_list = request.files.getlist("files")
        if not files_list:
            if "file1" in request.files and "file2" in request.files:
                files_list = [request.files["file1"], request.files["file2"]]
            else:
                return jsonify({"error": "no_files", "details": "Upload videos via form field 'files' (multiple) or provide 'file1' and 'file2'"}), 400
        run_dir_param = (request.form.get("run_dir") or request.form.get("folder") or "").strip() or "uploads"

        base_out = Path(app.config.get("OUTPUT_DIR", "static/generated"))
        out_dir = base_out / run_dir_param
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Save temporary parts preserving order
        temp_parts = []
        for idx, f in enumerate(files_list):
            p = out_dir / (f"part{idx+1}_" + uuid.uuid4().hex + ".mp4")
            f.save(str(p))
            temp_parts.append(p)

        final_path = out_dir / "stitched.mp4"
        try:
            ensure_ffmpeg()
            if len(temp_parts) == 1:
                try:
                    if final_path.exists():
                        final_path.unlink()
                    Path(temp_parts[0]).rename(final_path)
                except Exception:
                    concat_videos_many([str(temp_parts[0])], final_path)
            elif len(temp_parts) == 2:
                concat_videos(str(temp_parts[0]), str(temp_parts[1]), final_path)
            else:
                concat_videos_many([str(p) for p in temp_parts], final_path)
        except Exception as e:
            return jsonify({"error": "concat_failed", "details": str(e)}), 500
        finally:
            for p in temp_parts:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        base_url = request.url_root.rstrip("/")
        rel = str(final_path.relative_to(app.config["OUTPUT_DIR"])) if final_path.exists() else ""
        url = f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{rel}" if rel else ""
        return jsonify({
            "url": url,
            "local_file": str(final_path),
            "run_dir": run_dir_param,
        })

    return app


if __name__ == "__main__":  # pragma: no cover
    # Video features enabled by default; set ENABLE_VEO=0/false to disable.
    env_flag = (os.environ.get("ENABLE_VEO") or "").lower()
    enable_veo = not (env_flag in {"0", "false", "no"})
    app = create_app(enable_veo=enable_veo)
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8002")))
