const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

function addCacheBust(url) {
  try {
    const u = new URL(url, window.location.origin);
    u.searchParams.set('cb', String(Date.now()));
    return u.toString();
  } catch {
    const sep = url.includes('?') ? '&' : '?';
    return `${url}${sep}cb=${Date.now()}`;
  }
}

function slugify(s) {
  return (s || '').toLowerCase().trim().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'untitled';
}

function qs(params) {
  const sp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v === undefined || v === null) return;
    const sv = String(v).trim();
    if (sv.length === 0) return;
    sp.set(k, sv);
  });
  return sp.toString();
}

let LAST_RESULT = null;

function renderResult(container, data) {
  const title = data.title || "Untitled";
  const genre = data.genre || "";
  const nChars = data.num_characters ?? "";
  const synopsis = (data.synopsis || "").split(/\n+/).map(l => l.trim()).filter(Boolean);

  const images = data.images || [];
  const stitched = data.stitched_image || null;
  const poster = data.poster_image || null;

  container.innerHTML = `
    <div class="meta">
      <div>
        <h2>${escapeHtml(title)}</h2>
        <div class="sub">
          ${genre ? `<span class="pill">${escapeHtml(genre)}</span>` : ""}
          ${nChars !== "" ? `<span class="pill">Characters: ${nChars}</span>` : ""}
          ${data.source ? `<span class="pill alt">${escapeHtml(data.source)}</span>` : ""}
        </div>
      </div>
    </div>
    <div class="synopsis">
      ${synopsis.map(l => `<p>${escapeHtml(l)}</p>`).join("")}
    </div>
    <div class="media">
      ${poster ? renderImgCard("Poster", poster) : ""}
      ${stitched ? renderImgCard("Stitched Collage", stitched) : ""}
    </div>
    ${images.length ? `<h3>All Images</h3><div class="grid imgs">${images.map(renderImage).join("")}</div>` : ""}
  `;

  // Cache for video step and toggle UI
  LAST_RESULT = data;
  try { localStorage.setItem('last_result', JSON.stringify(data)); } catch {}
  const hasStitched = !!(data.stitched_image && data.stitched_image.url);
  const hasPoster = !!(data.poster_image && data.poster_image.url);
  const videoTools = document.getElementById("video-tools");
  const btnVideo = document.getElementById("btn-video");
  const consistencyTools = document.getElementById("consistency-tools");
  const btnResync = document.getElementById("btn-resync");
  if (videoTools) {
    videoTools.classList.remove("hidden");
  }
  if (btnVideo) {
    btnVideo.disabled = !hasStitched;
  }
  if (consistencyTools) {
    consistencyTools.classList.remove("hidden");
  }
  if (btnResync) {
    btnResync.disabled = !hasPoster;
  }
}

function renderImgCard(title, obj) {
  const url = addCacheBust(obj.url || "");
  const filename = obj.filename || "";
  return `
    <figure class="card media-card">
      <img src="${encodeURI(url)}" alt="${escapeHtml(filename)}" loading="lazy" />
      <figcaption>${escapeHtml(title)}</figcaption>
    </figure>
  `;
}

function renderImage(obj) {
  const url = addCacheBust(obj.url || "");
  const filename = obj.filename || "";
  return `
    <a class="img" href="${encodeURI(url)}" target="_blank" rel="noopener">
      <img src="${encodeURI(url)}" alt="${escapeHtml(filename)}" loading="lazy" />
    </a>
  `;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function handleSubmit(e) {
  e.preventDefault();
  const form = e.currentTarget;
  const status = $("#status");
  const result = $("#result");

  const title = form.title.value.trim();
  const genre = form.genre.value.trim();
  const synopsis = form.synopsis.value.trim();
  const n = form.n.value;
  const generate_images = form.generate_images.checked;

  const params = { n };
  if (generate_images === false) params.generate_images = "0"; // default is true
  if (title) params.title = title;
  if (genre) params.genre = genre;
  if (synopsis) params.synopsis = synopsis;

  const url = `/api/motion_poster?${qs(params)}`;
  status.textContent = "Generating…";
  result.classList.add("hidden");
  result.innerHTML = "";
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    const data = await res.json();
    renderResult(result, data);
    try { localStorage.setItem('last_result', JSON.stringify(data)); } catch {}
    result.classList.remove("hidden");
    status.textContent = "Done";
  } catch (err) {
    console.error(err);
    status.textContent = "Error generating. See console.";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  $("#gen-form").addEventListener("submit", handleSubmit);
  const btnVideo = document.getElementById("btn-video");
  if (btnVideo) btnVideo.addEventListener("click", handleGenerateVideo);
  const btnResync = document.getElementById("btn-resync");
  if (btnResync) btnResync.addEventListener("click", handleResyncWithPoster);
  const btnClear = document.getElementById("btn-clear");
  if (btnClear) btnClear.addEventListener("click", handleClear);
  const btnUpload = document.getElementById("btn-upload-process");
  if (btnUpload) btnUpload.addEventListener("click", handleUploadProcess);
  const btnYtUpload = document.getElementById('btn-yt-upload-file');
  if (btnYtUpload) btnYtUpload.addEventListener('click', handleYouTubeFileUpload);
  const folderEl = document.getElementById('upload-folder');
  const titleEl = document.querySelector('#gen-form input[name="title"]');
  if (folderEl) folderEl.addEventListener('input', updateUploadPreview);
  if (titleEl) titleEl.addEventListener('input', updateUploadPreview);
  // Restore last result on refresh
  try {
    const raw = localStorage.getItem('last_result');
    if (raw) {
      const data = JSON.parse(raw);
      const result = document.getElementById('result');
      renderResult(result, data);
      result.classList.remove('hidden');
    }
  } catch {}
  // Initialize preview
  updateUploadPreview();
});

async function handleGenerateVideo() {
  const btn = document.getElementById("btn-video");
  const status = document.getElementById("video-status");
  const out = document.getElementById("video-out");
  const durSel = document.getElementById("teaser-duration");
  const refStyle = document.getElementById("teaser-ref-style");
  const appendExtras = document.getElementById("teaser-append-extras");
  const folderInput = document.getElementById("teaser-folder");

  const stitched = (LAST_RESULT && LAST_RESULT.stitched_image && LAST_RESULT.stitched_image.url) || "";
  const synopsis = (LAST_RESULT.synopsis || "").trim();
  const duration = durSel ? durSel.value : "10";

  // Build query safely
  const sp = new URLSearchParams();
  if (synopsis) sp.set("synopsis", synopsis);
  if (stitched) sp.set("stitched_image_url", stitched);
  // Folder inference: prefer explicit input; else last run_dir; else slug from Title
  const form = document.getElementById('gen-form');
  let run_dir = folderInput && folderInput.value.trim();
  if (!run_dir && LAST_RESULT && LAST_RESULT.run_dir) run_dir = LAST_RESULT.run_dir;
  if (!run_dir && form && form.title && form.title.value.trim()) {
    run_dir = form.title.value.trim().toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-+|-+$/g,'') || 'untitled';
  }
  if (run_dir) sp.set('run_dir', run_dir);
  sp.set("duration", duration);
  if (refStyle && refStyle.checked) sp.set("style", "reference");
  if (appendExtras && !appendExtras.checked) sp.set("append_extras", "0");

  const url = `/api/video_teaser?${sp.toString()}`;
  btn.disabled = true;
  status.textContent = "Creating teaser…";
  out.classList.add("hidden");
  out.innerHTML = "";
  try {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data && data.error ? `${data.error}: ${data.details || ''}` : `HTTP ${res.status}`);
    }
    renderVideo(out, data);
    try {
      const last = localStorage.getItem('last_result');
      if (last) {
        const parsed = JSON.parse(last);
        parsed.teaser = data.url || data.local_file || '';
        localStorage.setItem('last_result', JSON.stringify(parsed));
      }
    } catch {}
    out.classList.remove("hidden");
    status.textContent = "Done";
  } catch (e) {
    console.error(e);
    status.textContent = `Error: ${e.message || e}`;
  } finally {
    btn.disabled = false;
  }
}

function renderVideo(container, data) {
  const url = addCacheBust(data.url || (data.local_file ? `/static/generated/${data.local_file.split('/').pop()}` : ""));
  const remote = data.video_url || "";
  const prompt = data.prompt || "";
  const model = data.model || "";
  const dur = data.duration || data.duration_seconds || "";
  const localFile = data.local_file || '';
  container.innerHTML = `
    <div class="video-wrap">
      ${url ? `<video controls src="${encodeURI(url)}"></video>` : "<div class=muted>No local video file created</div>"}
      ${remote && !url ? `<p><a href="${encodeURI(remote)}" target="_blank" rel="noopener">Open remote video</a></p>` : ""}
    </div>
    <div class="meta small">
      ${dur ? `<span class="pill">${dur}s</span>` : ""}
      ${model ? `<span class="pill alt">${escapeHtml(model)}</span>` : ""}
    </div>
    <div class="actions" style="margin-top: 12px;">
      <button id="btn-upload-youtube" class="secondary">Upload to YouTube</button>
      <span id="yt-status" class="small"></span>
    </div>
  `;
  const btn = document.getElementById('btn-upload-youtube');
  if (btn) btn.addEventListener('click', () => uploadToYouTube({ url, local_file: localFile }));
}

async function uploadToYouTube(info) {
  const status = document.getElementById('yt-status');
  if (status) status.textContent = '';
  // Compose metadata from current form / last result
  const form = document.getElementById('gen-form');
  let title = (form && form.title && form.title.value.trim()) || '';
  let description = '';
  try { description = (LAST_RESULT && LAST_RESULT.synopsis) ? LAST_RESULT.synopsis : ''; } catch {}
  const payload = {
    url: info && info.url || '',
    local_file: info && info.local_file || '',
    title: title || 'Story Spark Teaser',
    description: description || 'Created with Story Spark',
    privacyStatus: 'unlisted',
  };
  // Check server-side YouTube auth first (after we have payload so we can resume)
  try {
    const check = await fetch('/api/youtube/status');
    const js = await check.json();
    if (!js.authorized) {
      try { localStorage.setItem('yt_pending', JSON.stringify({ type: 'teaser', payload })); } catch {}
      if (status) status.textContent = 'Redirecting to YouTube to authorize…';
      window.location.href = '/api/youtube/auth';
      return;
    }
  } catch {}
  if (status) status.textContent = 'Uploading to YouTube…';
  try {
    const res = await fetch('/api/upload_youtube', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data && data.error ? `${data.error}: ${data.details || ''}` : `HTTP ${res.status}`);
    const link = data.link || (data.video && data.video.id ? `https://youtu.be/${data.video.id}` : '');
    if (status) status.innerHTML = link ? `Uploaded: <a href="${encodeURI(link)}" target="_blank" rel="noopener">${escapeHtml(link)}</a>` : 'Uploaded';
  } catch (e) {
    console.error(e);
    if (status) status.textContent = `Upload failed: ${e.message || e}`;
  }
}

async function handleUploadProcess() {
  const fileInput = document.getElementById('upload-video');
  const status = document.getElementById('upload-status');
  const out = document.getElementById('upload-video-out');
  const formEl = document.querySelector('#gen-form');
  const folderEl = document.getElementById('upload-folder');
  if (!fileInput || !fileInput.files || !fileInput.files[0]) {
    status.textContent = 'Choose a video file first';
    return;
  }
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  // Pass title if present so backend uses that folder
  if (formEl && formEl.title && formEl.title.value.trim()) {
    fd.append('title', formEl.title.value.trim());
  }
  if (folderEl && folderEl.value.trim()) {
    fd.append('run_dir', folderEl.value.trim());
  }
  status.textContent = 'Processing…';
  out.classList.add('hidden');
  out.innerHTML = '';
  try {
    const res = await fetch('/api/postprocess_video', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data && data.error ? `${data.error}: ${data.details || ''}` : `HTTP ${res.status}`);
    renderVideo(out, data);
    out.classList.remove('hidden');
    status.textContent = data.postprocessed ? 'Done (with end card)' : 'Done';
  } catch (e) {
    console.error(e);
    status.textContent = `Error: ${e.message || e}`;
  }
}

async function updateUploadPreview() {
  const info = document.getElementById('upload-preview');
  if (!info) return;
  const folderEl = document.getElementById('upload-folder');
  const titleEl = document.querySelector('#gen-form input[name="title"]');
  let run_dir = folderEl && folderEl.value.trim();
  if (!run_dir && titleEl && titleEl.value.trim()) run_dir = slugify(titleEl.value);
  if (!run_dir && LAST_RESULT && LAST_RESULT.run_dir) run_dir = LAST_RESULT.run_dir;
  if (!run_dir) {
    info.textContent = 'Folder not set — will default to "uploads".';
    return;
  }
  const posterJpg = `/static/generated/${run_dir}/first_look.jpg`;
  const posterPng = `/static/generated/${run_dir}/first_look.png`;
  const stitched = `/static/generated/${run_dir}/stitched.jpg`;
  const found = await probeAny([posterJpg, posterPng, stitched]);
  if (found) {
    const label = found.includes('first_look') ? 'first_look' : 'stitched.jpg';
    info.textContent = `Folder: ${run_dir} — using ${label} if present; end card will be added.`;
  } else {
    info.textContent = `Folder: ${run_dir} — no poster/stitched found; only end card will be added.`;
  }
}

async function probeAny(urls) {
  for (const u of urls) {
    try {
      const res = await fetch(addCacheBust(u), { method: 'HEAD' });
      if (res.ok) return u;
      // fallback GET if HEAD blocked
      const res2 = await fetch(addCacheBust(u), { method: 'GET', cache: 'no-store' });
      if (res2.ok) return u;
    } catch {}
  }
  return '';
}

function handleClear() {
  try { localStorage.removeItem('last_result'); } catch {}
  LAST_RESULT = null;
  const result = document.getElementById('result');
  if (result) {
    result.innerHTML = '';
    result.classList.add('hidden');
  }
  const videoOut = document.getElementById('video-out');
  if (videoOut) {
    videoOut.innerHTML = '';
    videoOut.classList.add('hidden');
  }
  const status = document.getElementById('status');
  if (status) status.textContent = '';
  const vStatus = document.getElementById('video-status');
  if (vStatus) vStatus.textContent = '';
  const btnVideo = document.getElementById('btn-video');
  if (btnVideo) btnVideo.disabled = true;
  const btnResync = document.getElementById('btn-resync');
  if (btnResync) btnResync.disabled = true;
}

async function handleYouTubeFileUpload() {
  const status = document.getElementById('yt-file-status');
  const fileInput = document.getElementById('yt-upload-file');
  const titleEl = document.getElementById('yt-upload-title');
  const descEl = document.getElementById('yt-upload-desc');
  const privEl = document.getElementById('yt-upload-privacy');
  if (!fileInput || !fileInput.files || !fileInput.files[0]) {
    if (status) status.textContent = 'Choose a video file first';
    return;
  }
  // Check auth
  try {
    const check = await fetch('/api/youtube/status');
    const js = await check.json();
    if (!js.authorized) {
      if (status) status.textContent = 'Redirecting to YouTube to authorize… (reselect file after)';
      try { localStorage.setItem('yt_pending', JSON.stringify({ type: 'file' })); } catch {}
      window.location.href = '/api/youtube/auth';
      return;
    }
  } catch {}
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  if (titleEl && titleEl.value.trim()) fd.append('title', titleEl.value.trim());
  if (descEl && descEl.value.trim()) fd.append('description', descEl.value.trim());
  if (privEl && privEl.value) fd.append('privacyStatus', privEl.value);
  if (status) status.textContent = 'Uploading…';
  try {
    const res = await fetch('/api/upload_youtube_file', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data && data.error ? `${data.error}: ${data.details || ''}` : `HTTP ${res.status}`);
    const link = data.link || (data.video && data.video.id ? `https://youtu.be/${data.video.id}` : '');
    if (status) status.innerHTML = link ? `Uploaded: <a href="${encodeURI(link)}" target="_blank" rel="noopener">${escapeHtml(link)}</a>` : 'Uploaded';
  } catch (e) {
    console.error(e);
    if (status) status.textContent = `Upload failed: ${e.message || e}`;
  }
}

// Auto-resume pending actions after returning from YouTube OAuth
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const raw = localStorage.getItem('yt_pending');
    if (!raw) return;
    const pending = JSON.parse(raw);
    // Check if we are authorized now
    const resp = await fetch('/api/youtube/status');
    const js = await resp.json();
    if (!js.authorized) return; // still not authorized
    // Clear flag early to avoid loops
    localStorage.removeItem('yt_pending');
    if (pending && pending.type === 'teaser' && pending.payload) {
      const status = document.getElementById('yt-status');
      if (status) status.textContent = 'Authorized. Uploading teaser to YouTube…';
      try {
        const res = await fetch('/api/upload_youtube', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(pending.payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data && data.error ? `${data.error}: ${data.details || ''}` : `HTTP ${res.status}`);
        const link = data.link || (data.video && data.video.id ? `https://youtu.be/${data.video.id}` : '');
        if (status) status.innerHTML = link ? `Uploaded: <a href="${encodeURI(link)}" target="_blank" rel="noopener">${escapeHtml(link)}</a>` : 'Uploaded';
      } catch (e) {
        console.error(e);
        const status = document.getElementById('yt-status');
        if (status) status.textContent = `Upload failed: ${e.message || e}`;
      }
    } else if (pending && pending.type === 'file') {
      const s = document.getElementById('yt-file-status');
      if (s) s.textContent = 'Authorized. Please reselect your file and click Upload again.';
    }
  } catch {}
});

async function handleResyncWithPoster() {
  const btn = document.getElementById("btn-resync");
  const status = document.getElementById("resync-status");
  const result = document.getElementById("result");
  if (!LAST_RESULT || !LAST_RESULT.poster_image || !LAST_RESULT.poster_image.url) {
    status.textContent = "Need poster first. Generate images above.";
    return;
  }
  btn.disabled = true;
  status.textContent = "Re‑syncing faces…";
  try {
    const sp = new URLSearchParams();
    // optional: preserve image count
    const n = (LAST_RESULT.images && LAST_RESULT.images.length) ? String(LAST_RESULT.images.length) : "8";
    sp.set("n", n);
    const res = await fetch(`/api/resync_with_poster?${sp.toString()}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data && data.error ? `${data.error}: ${data.details || ''}` : `HTTP ${res.status}`);
    renderResult(result, data);
    result.classList.remove("hidden");
    status.textContent = "Faces re‑synced";
  } catch (e) {
    console.error(e);
    status.textContent = `Error: ${e.message || e}`;
  } finally {
    btn.disabled = false;
  }
}
