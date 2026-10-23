const $ = (id) => document.getElementById(id);

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

$("startBtn").addEventListener("click", async () => {
  const source = $("source").value.trim();
  $("status").textContent = "starting...";
  // reconnect the video stream to the new session
  $("video").src = "/video_feed?t=" + Date.now();
  await postJSON("/api/start", { source });
});

$("stopBtn").addEventListener("click", async () => {
  await postJSON("/api/stop");
  $("status").textContent = "stopping...";
});

function cardHTML(p) {
  // cache-bust the crop so updated images actually refresh
  const src = p.image ? `${p.image}?v=${p.ver || 0}-${p.confidence}` : "";
  const thumb = src
    ? `<div class="thumb"><img src="${src}" alt="plate crop" /></div>`
    : `<div class="thumb empty"></div>`;
  const tag = p.matched
    ? `<div><span class="tag">WATCHLIST: ${p.matched_plate} ${
        p.exact ? "(exact)" : "~" + p.match_score + "%"
      }</span></div>`
    : "";
  const conf = Math.max(0, Math.min(100, p.confidence || 0));
  return `
    <div class="card ${p.matched ? "matched" : ""}">
      ${thumb}
      <div class="body">
        <div class="plate-text">${p.text || "—"}</div>
        <div class="confbar"><span style="width:${conf}%"></span></div>
        <div class="meta">
          ${p.vehicle || "vehicle"} · ${p.confidence}% confidence<br />
          seen ${p.count || 1}× · car id ${p.track_id}
        </div>
        ${tag}
      </div>
    </div>`;
}

function render(plates) {
  const wrap = $("plates");
  const empty = $("empty");
  if (!plates.length) {
    empty.style.display = "block";
    wrap.innerHTML = "";
    return;
  }
  empty.style.display = "none";
  wrap.innerHTML = plates.map(cardHTML).join("");
}

// ---- live tuning panel ----
let settingsLoaded = false;

function fillSettings(s) {
  if (!s) return;
  $("t_conf").value = s.display_min_confidence;
  $("t_conf_slider").value = s.display_min_confidence;
  $("t_conf_val").textContent = s.display_min_confidence + "%";
  $("t_dedupe").value = s.dedupe_similarity;
  $("t_sharp").value = s.min_sharpness;
  $("t_width").value = s.min_plate_width;
  $("t_roi").checked = !!s.roi_enabled;
  const r = s.roi || [0, 0, 1, 1];
  $("t_x1").value = r[0];
  $("t_y1").value = r[1];
  $("t_x2").value = r[2];
  $("t_y2").value = r[3];
}

async function applySettings() {
  const body = {
    display_min_confidence: parseFloat($("t_conf").value),
    dedupe_similarity: parseFloat($("t_dedupe").value),
    min_sharpness: parseFloat($("t_sharp").value),
    min_plate_width: parseInt($("t_width").value, 10),
    roi_enabled: $("t_roi").checked,
    roi: [
      parseFloat($("t_x1").value),
      parseFloat($("t_y1").value),
      parseFloat($("t_x2").value),
      parseFloat($("t_y2").value),
    ],
  };
  await postJSON("/api/settings", body);
  const msg = $("applyMsg");
  msg.textContent = "✓ applied";
  setTimeout(() => (msg.textContent = ""), 1500);
}

$("applyBtn").addEventListener("click", applySettings);

// Threshold slider: live readout while dragging, apply on release.
$("t_conf_slider").addEventListener("input", () => {
  const v = $("t_conf_slider").value;
  $("t_conf_val").textContent = v + "%";
  $("t_conf").value = v;
});
$("t_conf_slider").addEventListener("change", applySettings);
// Keep slider in sync if the number box is edited.
$("t_conf").addEventListener("input", () => {
  const v = $("t_conf").value;
  $("t_conf_slider").value = v;
  $("t_conf_val").textContent = v + "%";
});

function setDot(state) {
  const dot = $("dot");
  dot.className = "dot" + (state === "live" ? " live" : state === "err" ? " err" : "");
}

async function poll() {
  try {
    const s = await fetch("/api/state").then((r) => r.json());
    $("status").textContent = s.status;
    $("frameInfo").textContent = s.frame;
    $("fpsInfo").textContent = s.fps + " fps";
    $("countInfo").textContent = s.count;
    $("count").textContent = s.count;
    $("thrInfo").textContent = (s.threshold ?? "—") + "%";
    if (!settingsLoaded && s.settings) {
      fillSettings(s.settings);
      settingsLoaded = true;
    }
    setDot(
      s.running ? "live" : String(s.status).startsWith("error") || String(s.status).includes("could not") ? "err" : "idle"
    );
    render(s.plates);
  } catch (e) {
    $("status").textContent = "disconnected";
    setDot("err");
  }
}

setInterval(poll, 1000);
poll();
