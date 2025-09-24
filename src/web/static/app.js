// static/app.js
// Small, readable front-end controller for BookAI Web.

// ---------- Element refs ----------
const form          = document.getElementById("run-form");
const logsEl        = document.getElementById("logs");
const statusEl      = document.getElementById("status");
const startBtn      = document.getElementById("start-btn");
const galleryEl     = document.getElementById("gallery");      // current-run gallery
const allGalleryEl  = document.getElementById("all-gallery");  // all-time gallery
const sidenav       = document.getElementById("sidenav");
const hamburger     = document.getElementById("hamburger");
const backdrop      = document.getElementById("nav-backdrop");
const navLinks      = [...document.querySelectorAll(".nav-link")];

let pollLogsTimer = null;
let pollOutputsTimer = null;

// ---------- View switching (left nav) ----------
function showView(name) {
  navLinks.forEach(btn => btn.classList.toggle("active", btn.dataset.view === name));
  document.getElementById("view-generate").classList.toggle("hidden", name !== "generate");
  document.getElementById("view-gallery").classList.toggle("hidden", name !== "gallery");

  // When opening the gallery, refresh the full list once.
  if (name === "gallery") loadAllOutputs();
}

navLinks.forEach(btn => btn.addEventListener("click", () => showView(btn.dataset.view)));

// ---------- Sidebar slide-in/out ----------
const isMobile = () => window.matchMedia("(max-width: 900px)").matches;

function toggleNav(force) {
  if (isMobile()) {
    const willOpen = force !== undefined ? force : !document.body.classList.contains("nav-open");
    document.body.classList.toggle("nav-open", willOpen);         // overlay mode (shows backdrop)
    document.body.classList.toggle("nav-collapsed", !willOpen);   // keep logic consistent
    hamburger.setAttribute("aria-expanded", String(willOpen));
  } else {
    const collapsed = force !== undefined ? !force : !document.body.classList.contains("nav-collapsed");
    document.body.classList.toggle("nav-collapsed", collapsed);   // slides sidebar + shifts content
    document.body.classList.remove("nav-open");                   // no overlay on desktop
    hamburger.setAttribute("aria-expanded", String(!collapsed));
  }
}

window.addEventListener("load", () => toggleNav(!isMobile()));
window.addEventListener("resize", () => {
  if (isMobile()) {
    document.body.classList.add("nav-collapsed");
    document.body.classList.remove("nav-open");
    hamburger.setAttribute("aria-expanded", "false");
  } else {
    document.body.classList.remove("nav-open");
    document.body.classList.remove("nav-collapsed"); // expanded by default
    hamburger.setAttribute("aria-expanded", "true");
  }
});

hamburger.addEventListener("click", () => toggleNav());
backdrop.addEventListener("click", () => toggleNav(false));

// ---------- Run state helpers ----------
function setRunning(running) {
  startBtn.disabled = running;
  statusEl.textContent = running ? "Running…" : "Idle";
}

// ---------- Logs + current-run gallery polling ----------
async function pollLogs() {
  try {
    const res = await fetch("/logs");
    const data = await res.json();
    logsEl.textContent = (data.lines || []).join("\n") || "(no logs yet)";

    if (!data.running && pollLogsTimer) {
      clearInterval(pollLogsTimer);
      pollLogsTimer = null;
      setRunning(false);
    }
  } catch (e) {
    console.error("Log polling failed:", e);
  }
}

async function pollOutputs() {
  try {
    const res = await fetch("/outputs");
    const data = await res.json();
    const files = data.files || [];

    galleryEl.innerHTML = files.map(name => {
      const url = `/download/${encodeURIComponent(name)}`;
      return `
        <div class="thumb">
          <a href="${url}" target="_blank" rel="noopener">
            <img src="${url}" alt="${name}">
          </a>
          <div class="name">${name}</div>
        </div>
      `;
    }).join("");

    if (!data.running && pollOutputsTimer) {
      clearInterval(pollOutputsTimer);
      pollOutputsTimer = null;
    }
  } catch (e) {
    console.error("Outputs polling failed:", e);
  }
}

// ---------- All-time gallery (Gallery tab) ----------
async function loadAllOutputs() {
  try {
    const res = await fetch("/all-outputs");
    const data = await res.json();
    const files = data.files || [];

    allGalleryEl.innerHTML = files.length
      ? files.map(name => {
          const url = `/download/${encodeURIComponent(name)}`;
          return `
            <div class="thumb">
              <a href="${url}" target="_blank" rel="noopener">
                <img src="${url}" alt="${name}">
              </a>
              <div class="name">${name}</div>
            </div>
          `;
        }).join("")
      : `<p class="muted">No images found in the output folder yet.</p>`;
  } catch (e) {
    console.error("All outputs fetch failed:", e);
  }
}

// ---------- Form submission ----------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  setRunning(true);

  const formData = new FormData(form);

  try {
    const res = await fetch("/start", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || !data.ok) {
      alert(data.error || "Failed to start.");
      setRunning(false);
      return;
    }

    // Start polling during the run.
    pollLogs();
    pollOutputs();
    pollLogsTimer = setInterval(pollLogs, 1000);
    pollOutputsTimer = setInterval(pollOutputs, 1500);

    // Keep the Generate view active to show logs.
    showView("generate");
  } catch (err) {
    console.error(err);
    alert("Could not start run.");
    setRunning(false);
  }
});

// ---------- Initial load ----------
setRunning(false);
showView("generate");
pollLogs();
pollOutputs();
loadAllOutputs();
