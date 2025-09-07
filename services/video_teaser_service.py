import os
import json
import logging
from typing import Dict, Any

import requests

from video_teaser import build_video_prompt_base, build_video_prompt_safe_suffix


def build_video_prompt(*, title: str | None, synopsis: str, duration: int, safe: bool = False, style: str | None = None) -> str:
    base = build_video_prompt_base(duration)
    parts = [base]
    if title:
        parts.append(f"Movie title context: {title}")
    # sanitize synopsis
    parts.append(f"Story context: {sanitize_synopsis(synopsis)}")
    # Optional reference-style structure inspired by typical cinematic teasers
    if (style or "").lower() in {"ref", "reference", "cinematic", "hollywood"}:
        parts.append(
            "Structure beats (8s total): "
            "0.0–0.8s cold open with a striking visual and a percussive hit; "
            "0.8–2.5s three quick shots that establish WHO/WHERE (match cuts, sound pulses); "
            "2.5–5.5s rising stakes with two medium shots and one wide, tension ramps; "
            "5.5–7.2s micro‑montage of 5–6 ultra‑fast cuts synced to stingers; "
            "7.2–8.0s title/emblem reveal with a hard audio button. "
            "Shot grammar: alternate wide/medium/close for rhythm; use whip pans and push‑ins tastefully. "
            "Audio: motif + pulses + risers + stingers; purposeful near‑silence before impacts."
        )
    if safe:
        parts.append(build_video_prompt_safe_suffix())
    return " ".join(p for p in parts if p)


def sanitize_synopsis(text: str) -> str:
    import re
    problematic_terms = [
        r'demonic?\s+possession', r'evil', r'malicious\s+spirit', r'paranormal', r'horror', r'demon', r'supernatural', r'occult', r'exorcism',
        r'stag\s+party', r'bachelor\s+party', r'wild.*party', r'drinking', r'drunk', r'alcohol', r'drug[s]?', r'substance', r'hangover',
        r'casino', r'gambling', r'strip\s+club',
        r'gangster', r'gang\b', r'crime\b', r'criminal', r'syndicate', r'cartel', r'mafia', r'kidnap', r'hostage', r'assassin', r'assassination',
        r'murder', r'kill', r'killed', r'killer', r'weapon', r'gun', r'knife', r'blood', r'violent', r'violence', r'revenge', r'fight', r'warfare',
    ]
    sanitized = text
    for term in problematic_terms:
        sanitized = re.sub(term, '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'\b([A-Z][a-z]{2,})\b', 'a character', sanitized)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    sanitized = re.sub(r'[,;]\s*[,;]', ',', sanitized)
    if len(sanitized.strip()) < 50:
        return "A determined character faces escalating challenges and must make a courageous choice with allies by their side."
    return sanitized


def generate_video_teaser(*, title: str | None, synopsis: str, image_url: str, duration: int, fal_api_key: str, model: str, timeout: int, output_dir: str = "static/generated", style: str | None = None) -> Dict[str, Any]:
    log = logging.getLogger(__name__)
    if not fal_api_key:
        raise RuntimeError("FAL_API_KEY not set")

    # Convert localhost URLs to data URIs
    final_image_url = image_url
    if image_url.startswith(('http://localhost', 'http://127.0.0.1')):
        try:
            img_response = requests.get(image_url, timeout=10)
            img_response.raise_for_status()
            content_type = img_response.headers.get('content-type', 'image/jpeg')
            if not content_type.startswith('image/'):
                if image_url.lower().endswith('.png'):
                    content_type = 'image/png'
                else:
                    content_type = 'image/jpeg'
            import base64
            image_data = base64.b64encode(img_response.content).decode('utf-8')
            final_image_url = f"data:{content_type};base64,{image_data}"
            log.info("Converted localhost URL to base64 data URI")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot access local image URL {image_url}: {str(e)}")
    else:
        if not image_url.startswith(('http://', 'https://')):
            raise RuntimeError("image_url must be a valid HTTP/HTTPS URL")
        try:
            img_response = requests.head(image_url, timeout=10)
            if img_response.status_code != 200:
                raise RuntimeError(f"Image URL not accessible: HTTP {img_response.status_code}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot access image URL {image_url}: {str(e)}")

    # Veo3 currently supports only 8s
    duration_effective = 8
    duration_str = "8s"

    def make_payload(prompt_text: str) -> dict:
        return {
            "prompt": prompt_text,
            "image_url": final_image_url,
            "duration": duration_str,
            "resolution": "720p",
            "generate_audio": True,
        }

    safe = False
    prompt = build_video_prompt(title=title, synopsis=synopsis, duration=duration_effective, safe=safe, style=style)
    payload = make_payload(prompt)

    headers = {"Authorization": f"Key {fal_api_key}", "Content-Type": "application/json"}
    submit_url = f"https://fal.run/{model}"

    log.info(f"FAL.ai request to {submit_url}")
    response = requests.post(submit_url, json=payload, headers=headers, timeout=timeout)
    if response.status_code != 200:
        try:
            error_details = response.json()
            log.error(f"FAL.ai API error details: {error_details}")
            err_type = None
            det = error_details.get("detail") if isinstance(error_details, dict) else None
            if isinstance(det, list) and det and isinstance(det[0], dict):
                err_type = det[0].get("type")
            if response.status_code == 422 and err_type == "content_policy_violation":
                log.info("Retrying video generation with policy‑safe prompt…")
                safe = True
                payload2 = make_payload(build_video_prompt(title=title, synopsis=synopsis, duration=duration_effective, safe=True, style=style))
                response2 = requests.post(submit_url, json=payload2, headers=headers, timeout=timeout)
                if response2.status_code != 200:
                    try:
                        error_details2 = response2.json()
                        log.error(f"FAL.ai SAFE retry error: {error_details2}")
                    except Exception:
                        log.error(f"FAL.ai SAFE retry error text: {response2.text}")
                    raise RuntimeError(f"FAL.ai API error ({response2.status_code}): SAFE retry failed")
                result = response2.json()
            else:
                raise RuntimeError(f"FAL.ai API error ({response.status_code}): {error_details}")
        except Exception:
            raise RuntimeError(f"FAL.ai API error ({response.status_code}): {response.text}")
    else:
        result = response.json()

    # Direct video url
    if "video" in result and "url" in result["video"]:
        video_url = result["video"]["url"]
        local_file = download_video_file(video_url, output_dir)
        return {"video_url": video_url, "local_file": local_file, "duration": duration_effective, "prompt": prompt, "model": model, "status": "completed"}
    elif "request_id" in result:
        return poll_video_completion(result["request_id"], fal_api_key, model, timeout, output_dir)
    else:
        raise RuntimeError(f"Unexpected response format: {result}")


def poll_video_completion(request_id: str, fal_api_key: str, model: str, timeout: int, output_dir: str) -> Dict[str, Any]:
    import time
    headers = {"Authorization": f"Key {fal_api_key}"}
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
                    local_file = download_video_file(video_url, output_dir)
                    return {"video_url": video_url, "local_file": local_file, "duration": result.get("duration", 8), "model": model, "status": "completed", "request_id": request_id}
                else:
                    raise RuntimeError("Video completed but no URL found in response")
            elif status in ["failed", "error"]:
                error_msg = result.get("error", "Unknown error occurred")
                raise RuntimeError(f"Video generation failed: {error_msg}")
            elif status in ["queued", "in_progress"]:
                time.sleep(5)
                continue
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error polling video status: {str(e)}")
    raise RuntimeError(f"Video generation timed out after {timeout} seconds")


def download_video_file(video_url: str, output_dir: str) -> str:
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / "teaser.mp4"
    try:
        filepath.unlink(missing_ok=True)
    except Exception:
        pass
    response = requests.get(video_url, stream=True, timeout=60)
    response.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return str(filepath)
