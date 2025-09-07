# Story Spark — Movies First, Storybook Too

Story Spark is a pair of Flask services and a lightweight UI to help you ideate visual stories fast.

- Movie server (primary): generate a short synopsis, scene images, a stitched collage, and a poster — then optionally cut a teaser video or post‑process a local clip.
- Storybook server (companion): generate a 5‑scene illustrated children’s story and view it as a slideshow.

Main entry points:
- `movie.py` — primary server (port 8002)
- `storybook.py` — storybook server (port 8001)

## Quick Start

Prereqs:
- Python 3.9+ and ffmpeg available in PATH
- Google image generation SDK and API key

Install and configure:

```bash
pip install -r requirements.txt
pip install -U google-genai
export GOOGLE_API_KEY="<your-key>"
```

Run the Movie server (primary):

```bash
python movie.py
```

- Open UI: http://localhost:8002/
- API writes artifacts to `static/generated/<title-slug>/`.

Run the Storybook server:

```bash
python storybook.py
```

- API: http://localhost:8001/api/storyboard
- Dev UI: `cd static/ui/story_book && npm install && npm run dev` → http://localhost:5173/
- Prod UI: `npm run build`, then http://localhost:8001/storybook

## Movie API (movie.py)

- GET `/api/motion_poster`
  - Purpose: Generate synopsis, scene images, stitched collage, and a “first look” poster.
  - Query:
    - `title` (string, optional) — display title; also controls output folder slug
    - `genre` (string, optional) — guiding genre
    - `synopsis` (string, optional) — if omitted, it is generated
    - `n` (int, optional) — number of images, default 8 (1–12)
    - `generate_images` (bool, optional) — default true; set `0`/`false` to skip
    - `require_local` (bool) — require local LLM; otherwise falls back
    - `api_style` (string) — `auto` (default), `chat`, `completions`
  - Response:
    - `title`, `synopsis`, `genre`, `run_dir`
    - `images`: array of `{ url, filename }`
    - `stitched_image`: `{ url, filename }`
    - `poster_image`: `{ url, filename }`

- GET `/api/resync_with_poster`
  - Purpose: Regenerate scene images using the poster as a style/face reference.
  - Query: `n` (int, optional) — defaults to current count
  - Response: Same shape as `/api/motion_poster` (updates `images`, `stitched_image`).

- GET `/api/video_teaser`
  - Purpose: Generate an 8s teaser via FAL.ai Veo3.
  - Enable: run `python movie.py --model veo` or set `ENABLE_VEO=1`.
  - Query: `title`, `synopsis`, `stitched_image_url`, `run_dir` (all optional; inferred when possible)
  - Response: `{ url, local_file, postprocessed, stitched_image_url }`

- POST `/api/postprocess_video`
  - Purpose: Append poster + end card to an uploaded local video.
  - Form fields: `file` (required), `title` (optional), `run_dir` (optional)
  - Response: `{ url, local_file, run_dir, postprocessed }`

- GET `/api/diag` — Diagnostics (ffmpeg, versions, optional folder probe)
- GET `/health` — Health check

Movie configuration:
- Base: `OUTPUT_DIR` (default `static/generated`), `CORS_ALLOW_ORIGINS` (default `*`), `HOST`/`PORT` (default `0.0.0.0:8002`)
- LLM: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`, `LLM_API_KEY`, `LLM_API_STYLE`, `LLM_CHAT_PATH`, `LLM_COMPLETIONS_PATH`
- Images: `IMAGE_STORY_MODEL` (default `models/gemini-2.5-flash-image-preview`), `FALLBACK_IMAGE_MODEL`, `PEOPLE_DIR`
- Video: `ENABLE_VEO`, `FAL_API_KEY`, `FAL_VEO3_MODEL`, `VIDEO_TIMEOUT_SECONDS`

## Storybook API (storybook.py)

- GET `/api/storyboard`
  - Purpose: Create a 5‑scene kid‑safe storyboard and one illustration per scene.
  - Query: `theme`, `style`, `seed`, `ttl_seconds`, `require_local`, `api_style`
  - Response: `{ count, model, size, style, ttl_seconds, story_source, scenes[] }` where `scenes[]` → `{ index, text, image_url, filename }`

- GET `/storybook` — UI (prod build) or redirect to dev server
- GET `/health2` — Health check

Storybook configuration:
- Base: `OUTPUT_DIR` (default `static/generated/storybook`), `OUTPUT_TTL_SECONDS` (default `3600`), `CORS_ALLOW_ORIGINS`, `HOST`/`PORT` (default `0.0.0.0:8001`)
- LLM: same as Movie (`LLM_*`)
- Images: `GENERATION_MODEL` (default `models/gemini-2.5-flash-image-preview`), `IMAGE_SIZE`
- UI: `STORYBOOK_DIST_DIR` (default `static/ui/story_book/dist`), `STORYBOOK_DEV_URL` (default `http://localhost:5173`)

## Development

- Movie web UI: open http://localhost:8002/ after starting `movie.py`
- Storybook React UI (dev): `cd static/ui/story_book && npm install && npm run dev`
- Storybook React UI (prod): `npm run build`, then http://localhost:8001/storybook

## Troubleshooting

- Image generation errors or 401/403: ensure `GOOGLE_API_KEY` is set and has access
- LLM call failed: verify your OpenAI‑compatible server and `LLM_*` settings
- Video disabled: start Movie with `--model veo` or `ENABLE_VEO=1` and set `FAL_API_KEY`
- Storybook images not found: confirm they are under `/static/generated/storybook/`

## Files

- `movie.py` — Movie server (primary, port 8002). UI at `/`
- `storybook.py` — Storybook server (port 8001). UI at `/storybook` in prod
- `services/` — Motion poster, video teaser, post‑processing logic
- `static/ui/` — Movie server vanilla UI
- `static/ui/story_book/` — React/Vite storybook UI
- `static/generated/` — Movie artifacts under `static/generated/<title‑slug>/`
- `static/generated/storybook/` — Storybook images
- `requirements.txt` — Python dependencies

No license specified by default. Add one if needed.

