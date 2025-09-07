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
  if (typeof initTabs === 'function') initTabs();
  initTheme();
  initBrand();
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
  const btnStitch = document.getElementById('btn-stitch');
  if (btnStitch) btnStitch.addEventListener('click', handleStitchVideos);
  // Stitch drop/browse handlers
  const addBtn = document.getElementById('stitch-add');
  const inputEl = document.getElementById('stitch-files');
  const drop = document.getElementById('stitch-drop');
  if (addBtn && inputEl) addBtn.addEventListener('click', () => inputEl.click());
  if (inputEl) inputEl.addEventListener('change', () => { if (inputEl.files) { addStitchFiles(inputEl.files); inputEl.value = ''; }});
  if (drop) {
    ['dragenter','dragover'].forEach(evt => drop.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.add('dragover'); }));
    ['dragleave','drop'].forEach(evt => drop.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.remove('dragover'); }));
    drop.addEventListener('drop', (e) => { const dt = e.dataTransfer; if (dt && dt.files) addStitchFiles(dt.files); });
  }
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
      // Restore teaser player if previously generated
      if (data.teaser) {
        if (window.__setActiveTab) window.__setActiveTab('create');
        const out = document.getElementById('video-out');
        if (out) {
          renderVideo(out, { url: data.teaser, duration: 8 });
          out.classList.remove('hidden');
        }
      }
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

function initTabs() {
  const links = $$('.tab-link');
  const panels = $$('.tab-panel');
  const setActive = (name) => {
    links.forEach(l => l.classList.toggle('active', l.dataset.tab === name));
    panels.forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
    try { localStorage.setItem('active_tab', name); } catch {}
    if (name) { try { history.replaceState(null, '', `#${name}`); } catch {} }
  };
  links.forEach(l => l.addEventListener('click', () => setActive(l.dataset.tab)));
  const hash = (location.hash || '').replace('#','');
  const saved = (() => { try { return localStorage.getItem('active_tab') || ''; } catch { return ''; } })();
  const map = (v) => (v === 'teaser' ? 'create' : v);
  const allowed = ['create','stitch','post','youtube'];
  const initialHash = map(hash);
  const initialSaved = map(saved);
  const initial = allowed.includes(initialHash) ? initialHash : (allowed.includes(initialSaved) ? initialSaved : 'create');
  setActive(initial);
  window.__setActiveTab = setActive;
}

async function handleStitchVideos() {
  const status = document.getElementById('stitch-status');
  const out = document.getElementById('stitch-video-out');
  const filesEl = document.getElementById('stitch-files');
  const folder = document.getElementById('stitch-folder');
  const files = (window.__STITCH_QUEUE__ && window.__STITCH_QUEUE__.slice()) || (filesEl && filesEl.files ? Array.from(filesEl.files) : []);
  if (!files || files.length < 2) {
    if (status) status.textContent = 'Choose at least two video files';
    return;
  }
  const fd = new FormData();
  for (const f of files) {
    fd.append('files', f);
  }
  if (folder && folder.value.trim()) fd.append('run_dir', folder.value.trim());
  if (status) status.textContent = 'Stitching…';
  if (out) { out.classList.add('hidden'); out.innerHTML=''; }
  try {
    const res = await fetch('/api/concat_videos', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data && data.error ? `${data.error}: ${data.details || ''}` : `HTTP ${res.status}`);
    if (out) {
      renderVideo(out, data);
      out.classList.remove('hidden');
    }
    if (status) status.textContent = 'Done';
  } catch (e) {
    console.error(e);
    if (status) status.textContent = `Error: ${e.message || e}`;
  }
}

// Accumulate selected/dropped files for stitching
function addStitchFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const arr = Array.from(fileList).filter(f => f && f.type && f.type.startsWith('video/'));
  if (!arr.length) return;
  if (!window.__STITCH_QUEUE__) window.__STITCH_QUEUE__ = [];
  window.__STITCH_QUEUE__.push(...arr);
  renderStitchQueue();
}

function renderStitchQueue() {
  const ul = document.getElementById('stitch-list');
  if (!ul) return;
  const q = (window.__STITCH_QUEUE__ || []).slice(0, 50); // safety cap
  ul.innerHTML = q.map((f, i) => `
    <li draggable="true" data-index="${i}">
      <div class="file-row">
        <span class="badge">${i+1}</span>
        <span class="file-name">${escapeHtml(f.name || ('video'+(i+1)))}</span>
      </div>
      <div class="row-actions">
        <button class="btn-up" title="Move up" aria-label="Move up">▲</button>
        <button class="btn-down" title="Move down" aria-label="Move down">▼</button>
        <button class="btn-remove" title="Remove" aria-label="Remove">✕</button>
      </div>
    </li>`).join('');

  // Attach DnD handlers
  let dragSrcIdx = null;
  ul.querySelectorAll('li').forEach(li => {
    li.addEventListener('dragstart', (e) => {
      dragSrcIdx = Number(li.getAttribute('data-index'));
      li.classList.add('dragging');
      try { e.dataTransfer.effectAllowed = 'move'; } catch {}
    });
    li.addEventListener('dragend', () => {
      li.classList.remove('dragging');
      dragSrcIdx = null;
    });
    li.addEventListener('dragover', (e) => { e.preventDefault(); });
    li.addEventListener('drop', (e) => {
      e.preventDefault();
      const targetIdx = Number(li.getAttribute('data-index'));
      if (dragSrcIdx === null || isNaN(targetIdx)) return;
      reorderStitchQueue(dragSrcIdx, targetIdx);
    });
  });

  // Button actions via delegation
  ul.onclick = (ev) => {
    const btn = ev.target.closest('button');
    if (!btn) return;
    const li = btn.closest('li');
    if (!li) return;
    const idx = Number(li.getAttribute('data-index'));
    if (btn.classList.contains('btn-up')) {
      reorderStitchQueue(idx, Math.max(0, idx - 1));
    } else if (btn.classList.contains('btn-down')) {
      reorderStitchQueue(idx, Math.min((window.__STITCH_QUEUE__ || []).length - 1, idx + 1));
    } else if (btn.classList.contains('btn-remove')) {
      removeStitchItem(idx);
    }
  };
}

function reorderStitchQueue(from, to) {
  const q = window.__STITCH_QUEUE__ || [];
  if (from === to || from < 0 || to < 0 || from >= q.length || to >= q.length) return;
  const item = q.splice(from, 1)[0];
  q.splice(to, 0, item);
  window.__STITCH_QUEUE__ = q;
  renderStitchQueue();
}

function removeStitchItem(idx) {
  const q = window.__STITCH_QUEUE__ || [];
  if (idx < 0 || idx >= q.length) return;
  q.splice(idx, 1);
  window.__STITCH_QUEUE__ = q;
  renderStitchQueue();
}

function initTheme() {
  const toggle = document.getElementById('theme-toggle');
  const apply = (mode) => {
    document.body.classList.toggle('theme-light', mode === 'light');
    try { localStorage.setItem('theme', mode); } catch {}
    if (toggle) {
      toggle.setAttribute('title', mode === 'light' ? 'Switch to Dark' : 'Switch to Light');
      setThemeIcon(mode);
    }
  };
  let mode = (() => { try { return localStorage.getItem('theme'); } catch { return null; } })();
  if (!mode) {
    try { mode = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'; } catch { mode = 'dark'; }
  }
  apply(mode === 'light' ? 'light' : 'dark');
  if (toggle) toggle.addEventListener('click', () => {
    const next = document.body.classList.contains('theme-light') ? 'dark' : 'light';
    apply(next);
  });
}

function setThemeIcon(mode) {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;
  if (mode === 'light') {
    // show sun icon (light active)
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="5"/>
        <path d="M12 2v2M12 20v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M2 12h2M20 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
      </svg>`;
  } else {
    // show moon icon (dark active)
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
      </svg>`;
  }
}

function initBrand() {
  const applyAccent = (hex) => {
    document.documentElement.style.setProperty('--accent', hex);
    if (document.body) document.body.style.setProperty('--accent', hex);
    try { localStorage.setItem('accent', hex); } catch {}
  };
  const saved = (() => { try { return localStorage.getItem('accent'); } catch { return null; } })();
  if (saved) applyAccent(saved);
  $$('.swatch').forEach(btn => {
    btn.addEventListener('click', () => {
      const v = btn.getAttribute('data-accent');
      if (v) applyAccent(v);
    });
  });
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
      if (window.__setActiveTab) window.__setActiveTab('create');
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
