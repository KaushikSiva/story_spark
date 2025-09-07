const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

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
  const url = obj.url || "";
  const filename = obj.filename || "";
  return `
    <figure class="card media-card">
      <img src="${encodeURI(url)}" alt="${escapeHtml(filename)}" loading="lazy" />
      <figcaption>${escapeHtml(title)}</figcaption>
    </figure>
  `;
}

function renderImage(obj) {
  const url = obj.url || "";
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
});

async function handleGenerateVideo() {
  const btn = document.getElementById("btn-video");
  const status = document.getElementById("video-status");
  const out = document.getElementById("video-out");
  const durSel = document.getElementById("teaser-duration");
  const refStyle = document.getElementById("teaser-ref-style");

  if (!LAST_RESULT || !LAST_RESULT.stitched_image || !LAST_RESULT.stitched_image.url) {
    status.textContent = "Need stitched image first. Generate images above.";
    return;
  }

  const stitched = LAST_RESULT.stitched_image.url;
  const synopsis = (LAST_RESULT.synopsis || "").trim();
  const duration = durSel ? durSel.value : "10";

  // Build query safely
  const sp = new URLSearchParams();
  if (synopsis) sp.set("synopsis", synopsis);
  sp.set("stitched_image_url", stitched);
  sp.set("duration", duration);
  if (refStyle && refStyle.checked) sp.set("style", "reference");

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
  const url = data.url || (data.local_file ? `/static/generated/${data.local_file.split('/').pop()}` : "");
  const remote = data.video_url || "";
  const prompt = data.prompt || "";
  const model = data.model || "";
  const dur = data.duration || data.duration_seconds || "";
  container.innerHTML = `
    <div class="video-wrap">
      ${url ? `<video controls src="${encodeURI(url)}"></video>` : "<div class=muted>No local video file created</div>"}
      ${remote && !url ? `<p><a href="${encodeURI(remote)}" target="_blank" rel="noopener">Open remote video</a></p>` : ""}
    </div>
    <div class="meta small">
      ${dur ? `<span class="pill">${dur}s</span>` : ""}
      ${model ? `<span class="pill alt">${escapeHtml(model)}</span>` : ""}
    </div>
  `;
}

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
