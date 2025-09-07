from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import requests


class YouTubeUploadError(Exception):
    pass


def upload_video(
    *,
    access_token: str,
    file_path: str,
    title: str,
    description: str,
    privacy_status: str = "unlisted",
) -> Dict[str, Any]:
    """Upload a video to YouTube using a simple resumable upload flow.

    Requires an OAuth 2.0 access token with scope:
      https://www.googleapis.com/auth/youtube.upload

    Returns the created video resource (at least id/link fields when available).
    """
    log = logging.getLogger(__name__)
    fp = Path(file_path)
    if not fp.exists():
        raise YouTubeUploadError(f"File not found: {file_path}")

    size = fp.stat().st_size
    if size <= 0:
        raise YouTubeUploadError("File is empty")

    init_url = "https://www.googleapis.com/upload/youtube/v3/videos?part=snippet,status&uploadType=resumable"
    meta = {
        "snippet": {
            "title": title or fp.stem,
            "description": description or "",
        },
        "status": {"privacyStatus": privacy_status or "unlisted"},
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(size),
    }
    try:
        init_resp = requests.post(init_url, json=meta, headers=headers, timeout=30)
        if init_resp.status_code not in (200, 201):
            raise YouTubeUploadError(f"Init failed: HTTP {init_resp.status_code} {init_resp.text}")
        upload_url = init_resp.headers.get("Location")
        if not upload_url:
            raise YouTubeUploadError("No upload URL returned by YouTube API")
    except requests.RequestException as e:
        raise YouTubeUploadError(f"Init error: {e}") from e

    # Upload the file in one shot (works for small/medium files). For very large files
    # a chunked upload would be preferable.
    try:
        with open(fp, "rb") as f:
            put_headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "video/mp4",
                "Content-Length": str(size),
            }
            put_resp = requests.put(upload_url, data=f, headers=put_headers, timeout=300)
        if put_resp.status_code not in (200, 201):
            raise YouTubeUploadError(f"Upload failed: HTTP {put_resp.status_code} {put_resp.text}")
        data = put_resp.json()
        vid = data.get("id")
        link = f"https://youtu.be/{vid}" if vid else None
        result: Dict[str, Any] = {"video": data}
        if vid:
            result["id"] = vid
            result["link"] = link
        return result
    except requests.RequestException as e:
        raise YouTubeUploadError(f"Upload error: {e}") from e

