import os
import argparse
import logging
from pathlib import Path
import re
import uuid
from datetime import datetime
from flask import Flask, jsonify, request
from dotenv import load_dotenv

from image_utils import stitch_images_grid, load_reference_image_parts, file_to_genai_part
from services.motion_poster_service import (
    generate_motion_poster_script,
    generate_images_from_synopsis,
    generate_poster_from_synopsis,
    get_first_look_path,
)
from services.video_teaser_service import generate_video_teaser
from services.video_teaser_service import append_poster_and_tag


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
            return jsonify({"error": "Video teaser generation is disabled", "details": "Start with --model veo or ENABLE_VEO=1"}), 503

        title = (request.args.get("title") or "").strip() or None
        synopsis = (request.args.get("synopsis") or "").strip() or None
        stitched_image_url = (request.args.get("stitched_image_url") or "").strip() or None
        duration = 8  # enforced by service/model

        # Pull from cache if missing
        last_payload = getattr(app, "_last_motion_poster_cache", None)
        if not synopsis and last_payload:
            synopsis = last_payload.get("synopsis")
            title = title or last_payload.get("title")
        if not stitched_image_url and last_payload and last_payload.get("stitched_image"):
            stitched_image_url = last_payload["stitched_image"].get("url")

        if not synopsis:
            return jsonify({"error": "synopsis parameter is required or call /api/motion_poster first"}), 400
        if not stitched_image_url:
            return jsonify({"error": "stitched_image_url parameter is required or call /api/motion_poster first"}), 400

        try:
            # Determine output dir based on cached run_dir (title folder); fallback to base
            base_out = Path(app.config.get("OUTPUT_DIR", "static/generated"))
            out_dir = str(base_out / (last_payload.get("run_dir") or "")) if last_payload else str(base_out)
            if not out_dir.strip():
                out_dir = str(base_out)
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

        # Determine output dir in title folder if available
        last_payload = getattr(app, "_last_motion_poster_cache", None)
        base_out = Path(app.config.get("OUTPUT_DIR", "static/generated"))
        run_dir = None
        if title_override:
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

        try:
            final_path = append_poster_and_tag(str(upload_path), str(out_dir))
        except Exception as e:
            return jsonify({"error": "postprocess_failed", "details": str(e)}), 500

        base_url = request.url_root.rstrip("/")
        rel = str(Path(final_path).relative_to(app.config["OUTPUT_DIR"]))
        return jsonify({
            "url": f"{base_url}/{app.static_url_path.lstrip('/')}/generated/{rel}",
            "local_file": final_path,
            "run_dir": run_dir,
        })

    return app


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run the Story Spark server")
    parser.add_argument("--model", choices=["veo"], help="Enable video teaser generation via FAL.ai Veo3")
    args = parser.parse_args()

    enable_veo = (os.environ.get("ENABLE_VEO") or "").lower() in {"1", "true", "yes"} or (args.model == "veo")
    app = create_app(enable_veo=enable_veo)
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8002")))
