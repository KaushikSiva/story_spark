# Time Travel Image API (Flask)

Backend API that generates time‑travel themed images given a person's name and a target year. Uses Google’s image generation via the `google-genai` client (preferred) or falls back to `google.generativeai`. Default model: `models/gemini-2.5-flash-image-preview`. Generates 5 images and serves them from `/static/generated` so your frontend can display them by URL.

## Endpoints

- GET `/api/generate`
  - Body (JSON):
    - Query params:
      - `name` (string, required)
      - `year` (number/string, required)
      - `n` (number, optional, default 5)
      - `style` (string, optional)
  - Response (JSON):
    - `images`: array of `{ url, filename }` for display
    - `count`, `model`, `size`, `name`, `year`

- GET `/health` — basic health check

## Quick Start

1. Python 3.9+ recommended.
2. Install deps:

   ```bash
   pip install -U google-genai && pip install -r requirements.txt
   ```

3. Set your Google API key (needed for image generation):

   ```bash
   export GOOGLE_API_KEY="<your-key>"
   ```

4. Run the server:

   ```bash
   python app.py
   ```

5. Test the API:

   ```bash
   curl "http://localhost:8000/api/generate?name=Ada%20Lovelace&year=1850&n=5"
   ```

The response includes image URLs under `/static/generated/...` which you can use directly in your frontend.

## Configuration

Environment variables:

- `GOOGLE_API_KEY` (required): API key for Google Generative AI.
- `GENERATION_MODEL` (optional): Defaults to `models/gemini-2.5-flash-image-preview`. Adjust to your preferred model.
- `IMAGE_COUNT` (optional): Default images per request (default `5`).
- `IMAGE_SIZE` (optional): Default resolution like `1024x1024`.
- `OUTPUT_DIR` (optional): Where images are saved (default `static/generated`).
- `CORS_ALLOW_ORIGINS` (optional): Default `*`. Set to your frontend origin in production.
- `HOST` / `PORT` (optional): Server bind (defaults `0.0.0.0:8000`).
- `OUTPUT_TTL_SECONDS` (optional): How long images are kept before deletion (default 3600 seconds).
- `OUTPUT_SWEEP_INTERVAL_SECONDS` (optional): How often the background sweeper runs to clean old images (default 600 seconds).

## Notes on Models

- This backend prefers the new `google-genai` client (`from google import genai`) that aligns with the Gemini cookbook. It calls `client.images.generate(model=..., prompt=..., number_of_images=..., size=...)`.
- If `google-genai` isn’t installed, it falls back to `google.generativeai` and uses `GenerativeModel(...).generate_images(...)`.
- If you want a specific model variant (e.g., nano/flash/pro), set `GENERATION_MODEL` accordingly.

## Frontend Integration Tips

- The API returns URLs you can drop into `<img src=...>` tags.
- If your frontend runs on a different origin, set `CORS_ALLOW_ORIGINS` to that origin.
- You can pass an optional `style` field to nudge the visual look.

## Troubleshooting

- 401/403: Ensure `GOOGLE_API_KEY` is set and valid for image generation.
- 500 with `No images returned`: Model or quota issues; try a different model or reduce `n`.
- Empty/blocked images: Some prompts may be filtered; adjust prompt content.
- Images missing when reloading: Files are auto-deleted after TTL; increase `OUTPUT_TTL_SECONDS` if you need longer availability.

## License

No license specified by default. Add one if needed.
