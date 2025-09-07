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
});

