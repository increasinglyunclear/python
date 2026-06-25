/* Online viewer (read-only) — live network diagram on paper.
 *
 * Receives the shared state over the websocket and draws, in the same
 * black-on-white style as /presentation, a network of the themes around
 * 'creativity' that updates live: captured phrases appear as dots clustered
 * toward the themes they relate to. Click a theme to zoom in and list the
 * phrases captured there; click a phrase to find and flash the matching line in
 * the transcript scrolling along the bottom. */

(() => {
  "use strict";

  const canvas = document.getElementById("map");
  const ctx = canvas.getContext("2d");

  const panel = document.getElementById("panel");
  const panelTitle = document.getElementById("panel-title");
  const panelCount = document.getElementById("panel-count");
  const panelList = document.getElementById("panel-list");
  const panelEmpty = document.getElementById("panel-empty");
  const hintEl = document.getElementById("hint");
  const statusEl = document.getElementById("status");
  const transcriptEl = document.getElementById("transcript");
  const transcriptScroll = document.getElementById("transcript-scroll");

  // ---- Tunables ---------------------------------------------------------
  const RING_RADIUS = 0.62;
  const FOCUS_ZOOM = 1.7;
  const CREATIVITY = 0;
  const MAX_DOTS = 48; // density cap on the canvas
  const PANEL_MAX_GENERAL = 28;
  const PANEL_MAX_SPECIFIC = 20;
  const FOCUS_RING_MAX = 13; // captured concepts arranged around the selected node
  const FOCUS_RING_RADIUS = 0.3; // normalized orbit radius around the node

  let zoom = 1;
  let camOffsetX = 0;
  let camOffsetY = 0;
  let focusIndex = null;
  // The captured concepts arranged around the focused node (strongest first).
  let focusRingIds = null; // Set of concept ids in the ring
  let focusRingSlots = null; // id -> slot index
  let focusRingCount = 0;

  // Live data (refreshed each state).
  let anchorsByOrder = []; // sorted anchor concept objects
  let numAnchors = 0;
  let liveConcepts = [];
  let transcript = [];
  let activeTheme = null;
  // Live co-occurrence threads between theme nodes (overview only).
  let themeEdges = [];

  // Animated visuals (smoothed toward targets).
  const anchorNodes = new Map(); // anchor_index -> node
  const dots = new Map(); // concept id -> dot

  let dpr = 1;
  let width = 0;
  let height = 0;

  // ---- Speaker colours --------------------------------------------------
  const SPEAKER_COLORS = [
    "#0a66c2", "#b00020", "#1b7a3d", "#8a5a00",
    "#6a1b9a", "#00695c", "#c2185b", "#37474f",
  ];
  function speakerColor(si) {
    if (si == null || si < 0) return "#6f6f6f";
    return SPEAKER_COLORS[si % SPEAKER_COLORS.length];
  }
  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  // ---- Geometry ---------------------------------------------------------
  function resize() {
    dpr = window.devicePixelRatio || 1;
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resize);
  resize();

  function transcriptH() {
    return transcriptEl ? transcriptEl.offsetHeight : 0;
  }
  function availH() {
    return Math.max(120, height - transcriptH());
  }
  function centerY() {
    return availH() / 2;
  }
  function scale() {
    return Math.min(width, availH()) * 0.5 * 0.82 * zoom;
  }
  function project(nx, ny) {
    return [width / 2 + camOffsetX + nx * scale(), centerY() + camOffsetY + ny * scale()];
  }

  function ringPos(index, n) {
    if (index <= 0) return [0, 0];
    const ring = n > 1 ? n - 1 : 1;
    const k = index - 1;
    const ang = -Math.PI / 2 + (k * 2 * Math.PI) / ring;
    return [Math.cos(ang) * RING_RADIUS, Math.sin(ang) * RING_RADIUS];
  }

  function bestSpecific(weights) {
    let best = -Infinity;
    let idx = 1;
    for (let i = 1; i < weights.length; i++) {
      if (weights[i] > best) {
        best = weights[i];
        idx = i;
      }
    }
    return idx;
  }

  // Weighted position toward the specific themes (centre excluded), so a phrase
  // settles toward the theme(s) it belongs to; broadly-relevant ones sit middle.
  function dotTarget(weights) {
    let nx = 0;
    let ny = 0;
    let sum = 0;
    for (let i = 0; i < numAnchors; i++) {
      const ai = anchorsByOrder[i].anchor_index;
      if (ai === CREATIVITY) continue;
      const w = weights[i] || 0;
      const [ax, ay] = ringPos(ai, numAnchors);
      nx += w * ax;
      ny += w * ay;
      sum += w;
    }
    if (sum > 0) {
      nx /= sum;
      ny /= sum;
    }
    return [nx, ny];
  }

  // ---- Websocket --------------------------------------------------------
  async function resolveWsUrl() {
    const override = new URLSearchParams(location.search).get("ws");
    if (override) return override;
    if (window.LIVE_WS_URL) return window.LIVE_WS_URL;
    try {
      const r = await fetch("ws.txt", { cache: "no-store" });
      if (r.ok) {
        const t = (await r.text()).trim();
        if (t) return t.split(/\s+/)[0];
      }
    } catch (_) {
      /* no ws.txt — same origin */
    }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}/ws`;
  }

  function connect(url) {
    const ws = new WebSocket(url);
    ws.onopen = () => {
      statusEl.textContent = "live";
      statusEl.className = "connected";
    };
    ws.onmessage = (ev) => {
      let payload;
      try {
        payload = JSON.parse(ev.data);
      } catch (_) {
        return;
      }
      if (payload && payload.type === "state") applyState(payload);
    };
    ws.onclose = () => {
      statusEl.textContent = "reconnecting…";
      statusEl.className = "disconnected";
      setTimeout(() => connect(url), 1500);
    };
    ws.onerror = () => ws.close();
  }

  // ---- State ingestion --------------------------------------------------
  function applyState(state) {
    const concepts = state.concepts || [];
    anchorsByOrder = concepts
      .filter((c) => c.is_anchor)
      .sort((a, b) => a.anchor_index - b.anchor_index);
    numAnchors = anchorsByOrder.length;
    if (!numAnchors) return;

    // Reconcile anchor nodes.
    const seenAnchors = new Set();
    for (const a of anchorsByOrder) {
      seenAnchors.add(a.anchor_index);
      let node = anchorNodes.get(a.anchor_index);
      const [bx, by] = ringPos(a.anchor_index, numAnchors);
      if (!node) {
        node = { index: a.anchor_index, label: a.label, nx: bx, ny: by, alpha: 0, pulse: 1 };
        anchorNodes.set(a.anchor_index, node);
      } else {
        node.label = a.label;
      }
    }
    for (const k of [...anchorNodes.keys()]) {
      if (!seenAnchors.has(k)) anchorNodes.delete(k);
    }

    // Density cap: keep the strongest phrases (plus the focused theme's ring,
    // so its members always have a dot even if below the global cap).
    liveConcepts = concepts.filter((c) => !c.is_anchor);
    liveConcepts.sort((a, b) => (b.weight || 0) - (a.weight || 0));
    computeFocusRing();

    const toShow = liveConcepts.slice(0, MAX_DOTS);
    const showIds = new Set(toShow.map((c) => c.id));
    if (focusRingIds) {
      for (const c of liveConcepts) {
        if (focusRingIds.has(c.id) && !showIds.has(c.id)) {
          toShow.push(c);
          showIds.add(c.id);
        }
      }
    }

    let maxWeight = 0;
    for (const c of toShow) maxWeight = Math.max(maxWeight, c.weight || 0);

    for (const c of toShow) {
      const [tx, ty] = dotTarget(c.weights || []);
      let dot = dots.get(c.id);
      if (!dot) {
        dot = { id: c.id, nx: tx, ny: ty, alpha: 0 };
        dots.set(c.id, dot);
      }
      dot.tx = tx;
      dot.ty = ty;
      dot.label = c.label;
      dot.t = maxWeight > 0 ? (c.weight || 0) / maxWeight : 0;
      dot.theme = (c.weights || []).length > 1 ? bestSpecific(c.weights) : CREATIVITY;
      dot.live = true;
    }
    for (const [id, dot] of dots) {
      if (!showIds.has(id)) dot.live = false; // fade out, then drop
    }

    transcript = state.transcript || [];
    themeEdges = state.theme_edges || [];
    activeTheme = typeof state.active_theme === "number" ? state.active_theme : null;

    renderTranscript();
    if (focusIndex != null) renderPanel(focusIndex);
  }

  // ---- Transcript -------------------------------------------------------
  function entryKey(e) {
    return String(e.t);
  }

  // Stay pinned to the latest line unless the reader scrolls up or jumps to a
  // specific passage; then leave the transcript where they put it.
  let pinnedToBottom = true;
  let lastRenderSig = "";
  let flashKey = null;

  transcriptScroll.addEventListener("scroll", () => {
    pinnedToBottom =
      transcriptScroll.scrollHeight - transcriptScroll.scrollTop - transcriptScroll.clientHeight < 40;
  });

  function transcriptSig() {
    if (!transcript.length) return "0";
    return `${transcript.length}:${entryKey(transcript[0])}:${entryKey(transcript[transcript.length - 1])}`;
  }

  function renderTranscript() {
    const sig = transcriptSig();
    if (sig === lastRenderSig) return; // unchanged — don't disturb scroll/flash
    lastRenderSig = sig;

    transcriptScroll.innerHTML = transcript
      .map((e) => {
        const who = e.speaker
          ? `<span class="who" style="color:${speakerColor(e.si)}">${escapeHtml(e.speaker)}</span>`
          : "";
        return `<li data-k="${escapeHtml(entryKey(e))}">${who}${escapeHtml(e.text)}</li>`;
      })
      .join("");

    if (flashKey) {
      const again = transcriptScroll.querySelector(`li[data-k="${CSS.escape(flashKey)}"]`);
      if (again) again.classList.add("flash");
    }
    if (pinnedToBottom) transcriptScroll.scrollTop = transcriptScroll.scrollHeight;
  }

  function jumpToTranscript(label, themeIdx) {
    // Key-phrases can be multi-word and need not appear verbatim in one line,
    // so score lines by how many of the phrase's words they contain (most
    // recent wins ties; a line filed under this theme gets a small nudge).
    const tokens = (label || "").toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 2);
    if (!tokens.length) return;
    let match = null;
    let bestScore = 0;
    for (let i = transcript.length - 1; i >= 0; i--) {
      const e = transcript[i];
      const text = e.text.toLowerCase();
      let hits = 0;
      for (const t of tokens) if (text.includes(t)) hits++;
      if (!hits) continue;
      const themed = themeIdx == null || themeIdx === CREATIVITY || e.theme === themeIdx ? 0.5 : 0;
      const total = hits + themed;
      if (total > bestScore) {
        bestScore = total;
        match = e;
      }
    }
    if (!match) return;
    flashKey = entryKey(match);
    pinnedToBottom = false; // don't let live updates yank us back down
    for (const el of transcriptScroll.querySelectorAll("li.flash")) el.classList.remove("flash");
    const li = transcriptScroll.querySelector(`li[data-k="${CSS.escape(flashKey)}"]`);
    if (!li) return;
    li.classList.add("flash");
    // Centre it directly (reliable regardless of smooth-scroll timing).
    const target = li.offsetTop - transcriptScroll.clientHeight / 2 + li.offsetHeight / 2;
    transcriptScroll.scrollTop = Math.max(0, target);
  }

  // ---- Theme panel ------------------------------------------------------
  function conceptsForTheme(idx) {
    const out = [];
    for (const c of liveConcepts) {
      if (idx === CREATIVITY) {
        out.push(c);
      } else {
        const w = c.weights || [];
        if (w.length > 1 && bestSpecific(w) === idx) out.push(c);
      }
    }
    out.sort((a, b) => (b.weight || 0) - (a.weight || 0));
    return out.slice(0, idx === CREATIVITY ? PANEL_MAX_GENERAL : PANEL_MAX_SPECIFIC);
  }

  // The strongest concepts for the focused theme, placed in a ring around its
  // node. The panel still lists more; this keeps the canvas clean.
  function computeFocusRing() {
    focusRingIds = null;
    focusRingSlots = null;
    focusRingCount = 0;
    if (focusIndex == null) return;
    const list = conceptsForTheme(focusIndex).slice(0, FOCUS_RING_MAX);
    focusRingIds = new Set(list.map((c) => c.id));
    focusRingSlots = new Map(list.map((c, i) => [c.id, i]));
    focusRingCount = list.length;
  }

  function ringOffset(slot, total) {
    const ang = -Math.PI / 2 + (slot / Math.max(1, total)) * 2 * Math.PI;
    return [Math.cos(ang) * FOCUS_RING_RADIUS, Math.sin(ang) * FOCUS_RING_RADIUS];
  }

  function renderPanel(idx) {
    const node = anchorNodes.get(idx);
    panelTitle.textContent = node ? node.label : "";
    const list = conceptsForTheme(idx);
    panelCount.textContent = list.length ? `${list.length} captured` : "";
    panelEmpty.hidden = list.length > 0;
    panelList.innerHTML = list
      .map(
        (c) =>
          `<li data-label="${escapeHtml(c.label)}"><span class="phrase">${escapeHtml(
            c.label
          )}</span>${c.count > 1 ? `<span class="count">×${c.count}</span>` : ""}</li>`
      )
      .join("");
  }

  panelList.addEventListener("click", (e) => {
    const li = e.target.closest("li[data-label]");
    if (!li) return;
    jumpToTranscript(li.dataset.label, focusIndex);
  });

  function setFocus(index) {
    focusIndex = index;
    computeFocusRing();
    if (index != null) {
      renderPanel(index);
      panel.classList.add("visible");
      panel.setAttribute("aria-hidden", "false");
      hintEl.textContent = "click empty space to return";
    } else {
      panel.classList.remove("visible");
      panel.setAttribute("aria-hidden", "true");
      hintEl.textContent = "click a theme to explore";
    }
  }

  function nodeAt(sx, sy) {
    let best = null;
    let bestD = 46;
    for (const node of anchorNodes.values()) {
      if (focusIndex != null && node.index !== focusIndex) continue;
      if (node.alpha < 0.2) continue;
      const [x, y] = project(node.nx, node.ny);
      const d = Math.hypot(sx - x, sy - y);
      if (d < bestD) {
        bestD = d;
        best = node;
      }
    }
    return best;
  }

  // A ring concept near the pointer (focus mode only), for click-to-transcript.
  function conceptAt(sx, sy) {
    if (focusIndex == null || !focusRingIds) return null;
    let best = null;
    let bestD = 34;
    for (const dot of dots.values()) {
      if (!focusRingIds.has(dot.id) || dot.alpha < 0.3) continue;
      const [x, y] = project(dot.nx, dot.ny);
      const d = Math.hypot(sx - x, sy - y);
      if (d < bestD) {
        bestD = d;
        best = dot;
      }
    }
    return best;
  }

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    // A captured concept around the focused node: jump to it in the transcript.
    const concept = conceptAt(e.clientX - rect.left, e.clientY - rect.top);
    if (concept) {
      jumpToTranscript(concept.label, focusIndex);
      return;
    }
    const hit = nodeAt(e.clientX - rect.left, e.clientY - rect.top);
    if (hit) {
      setFocus(focusIndex === hit.index ? null : hit.index);
    } else if (focusIndex != null) {
      setFocus(null);
    }
  });

  // ---- Drawing ----------------------------------------------------------
  function drawSpacedCaps(text, cx, y, tracking) {
    const widths = [];
    let total = 0;
    for (const ch of text) {
      const w = ctx.measureText(ch).width + tracking;
      widths.push(w);
      total += w;
    }
    let x = cx - total / 2;
    ctx.textAlign = "left";
    for (let i = 0; i < text.length; i++) {
      ctx.fillText(text[i], x, y);
      x += widths[i];
    }
    ctx.textAlign = "center";
  }

  // ---- Curated graph (read-only mirror of /presentation) ----------------
  // The user-added nodes + connections, authored on /presentation and stored in
  // notes.php. Polled here and drawn on top of the live map.
  let curatedGraph = { nodes: [], edges: [] };
  let graphUrl = null; // the candidate endpoint that worked
  const GRAPH_CANDIDATES = [
    "/presentation/notes.php?graph=1",
    "presentation/notes.php?graph=1",
    "notes.php?graph=1",
  ];

  async function fetchGraph() {
    const urls = graphUrl ? [graphUrl] : GRAPH_CANDIDATES;
    for (const u of urls) {
      try {
        const r = await fetch(u, { cache: "no-store" });
        if (r.ok && (r.headers.get("content-type") || "").includes("json")) {
          const g = await r.json();
          curatedGraph.nodes = Array.isArray(g.nodes) ? g.nodes : [];
          curatedGraph.edges = Array.isArray(g.edges) ? g.edges : [];
          graphUrl = u;
          return;
        }
      } catch (_) {
        /* try the next candidate */
      }
    }
  }

  // Resolve a graph endpoint id to world coords: a theme (matched by lowercased
  // label) or a custom node.
  function curatedNodePos(id) {
    for (const node of anchorNodes.values()) {
      if (node.label && node.label.toLowerCase() === id) return [node.nx, node.ny];
    }
    const cn = curatedGraph.nodes.find((n) => n.id === id);
    return cn ? [cn.x, cn.y] : null;
  }

  function drawGraph() {
    if (!curatedGraph.nodes.length && !curatedGraph.edges.length) return;
    ctx.globalAlpha = 1;

    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1.2;
    for (const e of curatedGraph.edges) {
      const a = curatedNodePos(e.a);
      const b = curatedNodePos(e.b);
      if (!a || !b) continue;
      const [ax, ay] = project(a[0], a[1]);
      const [bx, by] = project(b[0], b[1]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }

    for (const n of curatedGraph.nodes) {
      const [x, y] = project(n.x, n.y);
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = "#0a0a0a";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      if (n.label) {
        ctx.fillStyle = "#0a0a0a";
        ctx.font = "600 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";
        drawSpacedCaps(n.label.toUpperCase(), x, y + 19, 13 * 0.12);
      }
    }
    ctx.globalAlpha = 1;
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const [cx, cy] = project(0, 0);

    // Edges from the hub out to each theme (overview only).
    if (focusIndex == null) {
      ctx.strokeStyle = "rgba(0,0,0,0.12)";
      ctx.lineWidth = 1;
      for (const node of anchorNodes.values()) {
        if (node.index === CREATIVITY) continue;
        const [x, y] = project(node.nx, node.ny);
        ctx.globalAlpha = node.alpha * 0.9;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // Live co-occurrence threads: themes discussed together, fading over time.
      for (const e of themeEdges) {
        const a = anchorNodes.get(e.a);
        const b = anchorNodes.get(e.b);
        if (!a || !b) continue;
        const s = Math.max(0, Math.min(1, e.strength || 0));
        const [ax, ay] = project(a.nx, a.ny);
        const [bx, by] = project(b.nx, b.ny);
        ctx.globalAlpha = Math.min(a.alpha, b.alpha) * (0.25 + s * 0.55);
        ctx.strokeStyle = "rgba(196, 92, 24, 0.9)";
        ctx.lineWidth = 1 + s * 3;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    } else {
      // Focused: retain the line from this theme back toward the centre, drawn
      // long so it trails off-screen — a sense of place in the network.
      const fn = anchorNodes.get(focusIndex);
      if (fn && fn.alpha > 0.01) {
        const [fx, fy] = project(fn.nx, fn.ny);
        const [ox, oy] = project(0, 0);
        let dx = ox - fx;
        let dy = oy - fy;
        const len = Math.hypot(dx, dy);
        if (len > 0.5) {
          dx /= len;
          dy /= len;
          const far = Math.hypot(width, height);
          ctx.strokeStyle = "rgba(0,0,0,0.16)";
          ctx.lineWidth = 1.2;
          ctx.globalAlpha = fn.alpha;
          ctx.beginPath();
          ctx.moveTo(fx, fy);
          ctx.lineTo(fx + dx * far, fy + dy * far);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
    }

    // Captured-phrase dots. In overview they're quiet dots clustered to their
    // theme; when a theme is focused, its ring members carry readable labels.
    const focusNode = focusIndex != null ? anchorNodes.get(focusIndex) : null;
    for (const dot of dots.values()) {
      if (dot.alpha < 0.01) continue;
      const [x, y] = project(dot.nx, dot.ny);
      const inRing = focusRingIds && focusRingIds.has(dot.id);

      ctx.globalAlpha = dot.alpha * (inRing ? 0.9 : 0.5);
      ctx.fillStyle = "#0a0a0a";
      ctx.beginPath();
      ctx.arc(x, y, inRing ? 3.2 : 1.6 + (dot.t || 0) * 4.5, 0, Math.PI * 2);
      ctx.fill();

      if (inRing && dot.label && focusNode) {
        const [fx, fy] = project(focusNode.nx, focusNode.ny);
        let ddx = x - fx;
        let ddy = y - fy;
        const dl = Math.hypot(ddx, ddy) || 1;
        ddx /= dl;
        ddy /= dl;
        ctx.globalAlpha = dot.alpha;
        ctx.fillStyle = "#0a0a0a";
        ctx.font = "400 12.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";
        ctx.textAlign = ddx >= 0 ? "left" : "right";
        ctx.fillText(dot.label, x + ddx * 9, y + ddy * 9);
        ctx.textAlign = "center";
      }
    }
    ctx.globalAlpha = 1;

    drawGraph();

    // Theme nodes + labels.
    for (const node of anchorNodes.values()) {
      if (node.alpha < 0.01) continue;
      const [x, y] = project(node.nx, node.ny);
      const focused = node.index === focusIndex;
      const isCentre = node.index === CREATIVITY;
      const isActive = node.index === activeTheme;
      // Focused node kept small (a quiet marker); the captured phrases live in
      // the side panel, so the diagram steps back when zoomed in.
      const radius = (focused ? 9 : isCentre ? 9 : 6) + node.pulse * 4;

      ctx.globalAlpha = node.alpha;
      ctx.fillStyle = "#0a0a0a";
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      if (focused || isCentre || isActive) {
        ctx.globalAlpha = node.alpha * (isActive && !focused ? 0.8 : 0.5);
        ctx.strokeStyle = "#0a0a0a";
        ctx.lineWidth = isActive && !focused ? 1.5 : 1;
        ctx.beginPath();
        ctx.arc(x, y, radius + 7, 0, Math.PI * 2);
        ctx.stroke();
      }

      const fontSize = focused ? 15 : isCentre ? 17 : 14;
      ctx.globalAlpha = node.alpha;
      ctx.fillStyle = "#0a0a0a";
      ctx.font = `600 ${fontSize}px ${"-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"}`;
      drawSpacedCaps(node.label.toUpperCase(), x, y + radius + fontSize, fontSize * 0.14);
    }
    ctx.globalAlpha = 1;
  }

  function tick() {
    const zoomTarget = focusIndex != null ? FOCUS_ZOOM : 1;
    zoom += (zoomTarget - zoom) * 0.08;

    // Pan so the focused theme tucks into the open area left of the panel
    // (small), its line trailing off toward the off-screen centre. In overview
    // the camera is centred.
    let panX = 0;
    let panY = 0;
    const fn = focusIndex != null ? anchorNodes.get(focusIndex) : null;
    if (fn) {
      const panelW = Math.min(440, width * 0.42); // keep in sync with view.css
      const anchorX = (width - panelW) * 0.46;
      const s = scale();
      panX = anchorX - width / 2 - fn.nx * s;
      panY = -fn.ny * s; // keep it vertically centred in the map area
    }
    camOffsetX += (panX - camOffsetX) * 0.08;
    camOffsetY += (panY - camOffsetY) * 0.08;

    for (const node of anchorNodes.values()) {
      // Every theme holds its ring position; focusing pans the camera to the
      // node rather than collapsing it onto the centre, so the line linking it
      // back to 'creativity' stays visible.
      const [tx, ty] = ringPos(node.index, numAnchors);
      node.nx += (tx - node.nx) * 0.12;
      node.ny += (ty - node.ny) * 0.12;
      const hidden = focusIndex != null && node.index !== focusIndex;
      node.alpha += ((hidden ? 0 : 1) - node.alpha) * 0.12;
      node.pulse *= 0.9;
    }

    const focusNode = focusIndex != null ? anchorNodes.get(focusIndex) : null;
    for (const [id, dot] of dots) {
      const inRing = focusRingIds && focusRingIds.has(id);
      let tx;
      let ty;
      let visible;
      if (inRing && focusNode) {
        // Arrange this theme's captured concepts in a ring around its node.
        const [ox, oy] = ringOffset(focusRingSlots.get(id), focusRingCount);
        tx = focusNode.nx + ox;
        ty = focusNode.ny + oy;
        visible = true;
      } else {
        tx = dot.tx ?? dot.nx;
        ty = dot.ty ?? dot.ny;
        visible = focusIndex == null && dot.live; // plain dots only in overview
      }
      dot.nx += (tx - dot.nx) * 0.1;
      dot.ny += (ty - dot.ny) * 0.1;
      dot.alpha += ((visible ? 1 : 0) - dot.alpha) * 0.1;
      if (!dot.live && !inRing && dot.alpha < 0.02) dots.delete(id);
    }

    draw();
    requestAnimationFrame(tick);
  }

  // ---- Boot -------------------------------------------------------------
  function syncTranscriptVar() {
    document.documentElement.style.setProperty("--transcript-h", transcriptH() + "px");
  }
  window.addEventListener("resize", syncTranscriptVar);
  syncTranscriptVar();

  resolveWsUrl().then((url) => connect(url));
  fetchGraph();
  setInterval(fetchGraph, 5000); // curated graph refresh (independent of the WS)
  requestAnimationFrame(tick);
})();
