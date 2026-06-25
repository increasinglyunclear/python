/* Presentation — black-on-white interactive diagram with editable notes.
 *
 * Loads the theme list (GET /themes) and lays the themes out as a network
 * around 'creativity'. Click a theme to zoom into it (same feel as the local
 * presentation mode) and read its prepared notes (GET /content/<theme>); the
 * notes are editable and saved with PUT /content/<theme>. Click empty space to
 * zoom back out.
 *
 * The API base is the same origin when served by the app (locally or via the
 * tunnel). On the static website, set window.PRESENTATION_API or an api.txt
 * file to the tunnel base so reads/saves reach the running server. */

(() => {
  "use strict";

  const params = new URLSearchParams(location.search);
  const canvas = document.getElementById("map");
  const ctx = canvas.getContext("2d");

  const panel = document.getElementById("panel");
  const panelTitle = document.getElementById("panel-title");
  const panelBody = document.getElementById("panel-body");
  const panelEdit = document.getElementById("panel-edit");
  const panelStatus = document.getElementById("panel-status");
  const editBtn = document.getElementById("edit-btn");
  const cancelBtn = document.getElementById("cancel-btn");
  const saveBtn = document.getElementById("save-btn");
  const hintEl = document.getElementById("hint");

  // ---- Layout / camera (mirrors the local map) --------------------------
  const RING_RADIUS = 0.62;
  const FOCUS_ZOOM = 1.7;
  let zoom = 1;
  let camOffsetX = 0;
  let camOffsetY = 0;

  let apiBase = "";
  let usePhp = false; // website standalone store (notes.php) vs the app
  let themes = []; // [creativity, ...others]
  const notes = new Map(); // theme -> last saved text (cached on first open)
  let nodes = []; // { index, label, nx, ny, tx, ty, alpha, pulse }
  let focusIndex = null; // focused theme index, or null
  let focusCustom = null; // focused custom node object, or null
  let currentKey = null; // notes store key for whatever the panel shows
  let editing = false;

  let dpr = 1;
  let width = 0;
  let height = 0;

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

  function scale() {
    return Math.min(width, height) * 0.5 * 0.82 * zoom;
  }
  function project(nx, ny) {
    return [width / 2 + camOffsetX + nx * scale(), height / 2 + camOffsetY + ny * scale()];
  }

  function ringPos(index, n) {
    if (index <= 0) return [0, 0];
    const ring = n > 1 ? n - 1 : 1;
    const k = index - 1;
    const ang = -Math.PI / 2 + (k * 2 * Math.PI) / ring;
    return [Math.cos(ang) * RING_RADIUS, Math.sin(ang) * RING_RADIUS];
  }

  function targetPos(node) {
    // Every theme holds its ring position. Focusing pans/zooms the camera to the
    // chosen node instead of collapsing it onto the centre, so the line linking
    // it back to 'creativity' stays visible (trailing off-screen).
    return ringPos(node.index, nodes.length);
  }

  function isHidden(node) {
    if (focusCustom) return true; // a custom node is focused → themes step back
    return focusIndex != null && node.index !== focusIndex;
  }

  // ---- API --------------------------------------------------------------
  async function resolveApiBase() {
    const o = params.get("api");
    if (o) return o.replace(/\/+$/, "");
    if (window.PRESENTATION_API) return String(window.PRESENTATION_API).replace(/\/+$/, "");
    try {
      const r = await fetch("api.txt", { cache: "no-store" });
      if (r.ok) {
        const t = (await r.text()).trim();
        if (t) return t.split(/\s+/)[0].replace(/\/+$/, "");
      }
    } catch (_) {
      /* no api.txt — same origin */
    }
    return "";
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function paragraphs(text) {
    return text
      .split(/\n\s*\n/)
      .map((p) => p.trim())
      .filter(Boolean);
  }

  // Where notes are read/written: the website's own PHP store when present
  // (works with the laptop off), otherwise the running app (local or tunnel).
  function contentUrl(theme) {
    return usePhp
      ? `notes.php?theme=${encodeURIComponent(theme)}`
      : `${apiBase}/content/${encodeURIComponent(theme)}`;
  }

  async function loadNote(theme) {
    if (notes.has(theme)) return notes.get(theme);
    try {
      const r = await fetch(contentUrl(theme), { cache: "no-store" });
      const text = r.ok ? await r.text() : "";
      notes.set(theme, text);
      return text;
    } catch (_) {
      return "";
    }
  }

  // ---- Notes panel ------------------------------------------------------
  function renderBody(text) {
    panelBody.innerHTML = text
      ? paragraphs(text)
          .map((p) => `<p>${escapeHtml(p).replace(/\n/g, " ")}</p>`)
          .join("")
      : `<p class="empty">No notes yet.</p>`;
  }

  function setStatus(msg) {
    panelStatus.textContent = msg || "";
  }

  function leaveEdit() {
    editing = false;
    panelEdit.hidden = true;
    panelBody.hidden = false;
    editBtn.hidden = false;
    cancelBtn.hidden = true;
    saveBtn.hidden = true;
  }

  // `key` is the notes store key (a theme name or a custom node id); `title` is
  // what's shown in the panel header.
  async function openPanel(key, title) {
    leaveEdit();
    setStatus("");
    currentKey = key;
    panelTitle.textContent = title;
    panelBody.innerHTML = "";
    panel.classList.add("visible");
    panel.setAttribute("aria-hidden", "false");
    const text = await loadNote(key);
    if (currentKey === key) renderBody(text);
  }

  function closePanel() {
    panel.classList.remove("visible");
    panel.setAttribute("aria-hidden", "true");
    leaveEdit();
  }

  function enterEdit() {
    editing = true;
    panelEdit.value = notes.get(currentKey) || "";
    panelEdit.hidden = false;
    panelBody.hidden = true;
    editBtn.hidden = true;
    cancelBtn.hidden = false;
    saveBtn.hidden = false;
    setStatus("editing");
    panelEdit.focus();
  }

  async function saveEdit() {
    const key = currentKey;
    const val = panelEdit.value;
    if (val === (notes.get(key) || "")) {
      leaveEdit();
      renderBody(val);
      setStatus("");
      return;
    }
    setStatus("saving…");
    try {
      const r = await fetch(contentUrl(key), {
        method: usePhp ? "POST" : "PUT",
        headers: { "Content-Type": "text/plain" },
        body: val,
      });
      if (r.ok) {
        notes.set(key, val);
        leaveEdit();
        renderBody(val);
        setStatus("saved");
        setTimeout(() => setStatus(""), 2500);
      } else {
        setStatus("save failed");
      }
    } catch (_) {
      setStatus("save failed — is the server reachable?");
    }
  }

  editBtn.addEventListener("click", enterEdit);
  cancelBtn.addEventListener("click", () => {
    leaveEdit();
    renderBody(notes.get(currentKey) || "");
    setStatus("");
  });
  saveBtn.addEventListener("click", saveEdit);

  // ---- References (editable side panel, saved to the same store) ---------
  const refsToggle = document.getElementById("refs-toggle");
  const refs = document.getElementById("refs");
  const refsBody = document.getElementById("refs-body");
  const refsEdit = document.getElementById("refs-edit");
  const refsEditBtn = document.getElementById("refs-edit-btn");
  const refsCancelBtn = document.getElementById("refs-cancel-btn");
  const refsSaveBtn = document.getElementById("refs-save-btn");
  const refsStatus = document.getElementById("refs-status");
  let refsOpen = false;
  let refsCache = null; // last loaded/saved text

  // References live only online, in the same PHP store as the theme notes. The
  // toggle is shown only when that store is present (see load()).
  function refsUrl() {
    return "notes.php?theme=references";
  }

  function renderRefs(text) {
    const lines = (text || "")
      .split(/\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    refsBody.innerHTML = lines.length
      ? lines.map((l) => `<p>${escapeHtml(l)}</p>`).join("")
      : `<p class="empty">No references yet.</p>`;
  }

  function setRefsStatus(msg) {
    refsStatus.textContent = msg || "";
  }

  function leaveRefsEdit() {
    refsEdit.hidden = true;
    refsBody.hidden = false;
    refsEditBtn.hidden = false;
    refsCancelBtn.hidden = true;
    refsSaveBtn.hidden = true;
  }

  async function loadRefs() {
    if (refsCache != null) return refsCache;
    try {
      const r = await fetch(refsUrl(), { cache: "no-store" });
      refsCache = r.ok ? await r.text() : "";
    } catch (_) {
      refsCache = "";
    }
    return refsCache;
  }

  async function openRefs() {
    refsOpen = true;
    refsToggle.classList.add("active");
    refs.classList.add("visible");
    refs.setAttribute("aria-hidden", "false");
    leaveRefsEdit();
    setRefsStatus("");
    const text = await loadRefs();
    if (refsOpen) renderRefs(text);
  }

  function closeRefs() {
    refsOpen = false;
    refsToggle.classList.remove("active");
    refs.classList.remove("visible");
    refs.setAttribute("aria-hidden", "true");
    leaveRefsEdit();
  }

  function enterRefsEdit() {
    refsEdit.value = refsCache || "";
    refsEdit.hidden = false;
    refsBody.hidden = true;
    refsEditBtn.hidden = true;
    refsCancelBtn.hidden = false;
    refsSaveBtn.hidden = false;
    setRefsStatus("editing");
    refsEdit.focus();
  }

  async function saveRefs() {
    const val = refsEdit.value;
    if (val === (refsCache || "")) {
      leaveRefsEdit();
      renderRefs(val);
      setRefsStatus("");
      return;
    }
    setRefsStatus("saving…");
    try {
      const r = await fetch(refsUrl(), {
        method: usePhp ? "POST" : "PUT",
        headers: { "Content-Type": "text/plain" },
        body: val,
      });
      if (r.ok) {
        refsCache = val;
        leaveRefsEdit();
        renderRefs(val);
        setRefsStatus("saved");
        setTimeout(() => setRefsStatus(""), 2500);
      } else {
        setRefsStatus("save failed");
      }
    } catch (_) {
      setRefsStatus("save failed — is the server reachable?");
    }
  }

  // Wire the references panel only when its markup is present. After an
  // incremental deploy a browser can run this (new) script against a stale,
  // cached presentation.html that lacks these elements; guarding here keeps the
  // diagram alive instead of throwing on a null element reference.
  if (refsToggle && refs && refsEditBtn && refsCancelBtn && refsSaveBtn) {
    refsToggle.addEventListener("click", () => {
      if (refsOpen) closeRefs();
      else openRefs();
    });
    refsEditBtn.addEventListener("click", enterRefsEdit);
    refsCancelBtn.addEventListener("click", () => {
      leaveRefsEdit();
      renderRefs(refsCache || "");
      setRefsStatus("");
    });
    refsSaveBtn.addEventListener("click", saveRefs);
  }

  // ---- Curated graph: user-added nodes + connections --------------------
  // Persisted online in notes.php (?graph=1); /live and /display read the same
  // store and draw these on top of the live map. Editing is online-only.
  let graph = { nodes: [], edges: [] }; // nodes: {id,label,x,y}; edges: {a,b}
  const addNodeBtn = document.getElementById("add-node-btn");
  const nodeInput = document.getElementById("node-label-input");
  let drag = null; // {mode:'move'|'connect'|'click', ...}
  const pointer = { x: 0, y: 0 }; // last canvas-space pointer (rubber-band)
  let labelling = null; // { node, isNew } while the inline editor is open
  const HANDLE_DX = 14;
  const HANDLE_DY = -14;

  function worldFromScreen(sx, sy) {
    const s = scale();
    return [(sx - width / 2 - camOffsetX) / s, (sy - height / 2 - camOffsetY) / s];
  }

  // Resolve any node id (a theme name or a custom node id) to world coords.
  function nodePosById(id) {
    const ti = themes.indexOf(id);
    if (ti >= 0) return nodes[ti] ? [nodes[ti].nx, nodes[ti].ny] : null;
    const cn = graph.nodes.find((n) => n.id === id);
    return cn ? [cn.x, cn.y] : null;
  }

  function customNodeAt(sx, sy) {
    let best = null;
    let bestD = 22;
    for (const n of graph.nodes) {
      const [x, y] = project(n.x, n.y);
      const d = Math.hypot(sx - x, sy - y);
      if (d < bestD) {
        bestD = d;
        best = n;
      }
    }
    return best;
  }

  // The endpoint id (theme name or custom node id) whose link handle is under
  // the cursor, or null. Core theme handles are offered only in overview (where
  // every theme is visible); a focused custom node exposes only its own handle.
  function handleAt(sx, sy) {
    if (focusIndex == null && !focusCustom) {
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        if (n.alpha < 0.2) continue;
        const [x, y] = project(n.nx, n.ny);
        if (Math.hypot(sx - (x + HANDLE_DX), sy - (y + HANDLE_DY)) < 9) return themes[i];
      }
    }
    for (const n of graph.nodes) {
      if (focusCustom && n !== focusCustom) continue;
      const [x, y] = project(n.x, n.y);
      if (Math.hypot(sx - (x + HANDLE_DX), sy - (y + HANDLE_DY)) < 9) return n.id;
    }
    return null;
  }

  // Nearest node of any kind (connection target), excluding one id.
  function anyNodeIdAt(sx, sy, excludeId) {
    let bestId = null;
    let bestD = 30;
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      if (n.alpha < 0.2 || themes[i] === excludeId) continue;
      const [x, y] = project(n.nx, n.ny);
      const d = Math.hypot(sx - x, sy - y);
      if (d < bestD) {
        bestD = d;
        bestId = themes[i];
      }
    }
    for (const n of graph.nodes) {
      if (n.id === excludeId) continue;
      const [x, y] = project(n.x, n.y);
      const d = Math.hypot(sx - x, sy - y);
      if (d < bestD) {
        bestD = d;
        bestId = n.id;
      }
    }
    return bestId;
  }

  function distToSeg(px, py, ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    const l2 = dx * dx + dy * dy;
    let t = l2 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  function edgeAt(sx, sy) {
    for (let i = 0; i < graph.edges.length; i++) {
      const a = nodePosById(graph.edges[i].a);
      const b = nodePosById(graph.edges[i].b);
      if (!a || !b) continue;
      const [ax, ay] = project(a[0], a[1]);
      const [bx, by] = project(b[0], b[1]);
      if (distToSeg(sx, sy, ax, ay, bx, by) < 6) return i;
    }
    return -1;
  }

  // -- persistence (debounced) --
  let saveTimer = null;
  async function saveGraph() {
    try {
      await fetch("notes.php?graph=1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(graph),
      });
    } catch (_) {
      /* best-effort */
    }
  }
  function scheduleSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(saveGraph, 600);
  }

  async function loadGraph() {
    try {
      const r = await fetch("notes.php?graph=1", { cache: "no-store" });
      if (r.ok) {
        const g = await r.json();
        graph.nodes = Array.isArray(g.nodes) ? g.nodes : [];
        graph.edges = Array.isArray(g.edges) ? g.edges : [];
      }
    } catch (_) {
      /* none yet */
    }
  }

  // -- adding / labelling --
  function addNode() {
    const s = scale();
    // Drop near the current view centre, fanned out by a small spiral so
    // successive nodes don't stack on the centre node.
    const ang = graph.nodes.length * 1.1 - Math.PI / 2;
    const r = 0.22;
    const node = {
      id: "n" + Date.now().toString(36),
      label: "",
      x: -camOffsetX / s + Math.cos(ang) * r,
      y: -camOffsetY / s + Math.sin(ang) * r,
    };
    graph.nodes.push(node);
    startLabel(node, true);
  }

  function startLabel(node, isNew) {
    labelling = { node, isNew };
    const [x, y] = project(node.x, node.y);
    nodeInput.value = node.label;
    nodeInput.style.left = x + "px";
    nodeInput.style.top = y + "px";
    nodeInput.hidden = false;
    nodeInput.focus();
    nodeInput.select();
  }

  function commitLabel() {
    if (!labelling) return;
    const { node, isNew } = labelling;
    const val = nodeInput.value.trim();
    labelling = null;
    nodeInput.hidden = true;
    if (!val) {
      if (isNew) graph.nodes = graph.nodes.filter((n) => n !== node);
    } else {
      node.label = val;
      if (focusCustom === node) panelTitle.textContent = val;
      scheduleSave();
    }
  }

  function cancelLabel() {
    if (!labelling) return;
    const { node, isNew } = labelling;
    labelling = null;
    nodeInput.hidden = true;
    if (isNew) graph.nodes = graph.nodes.filter((n) => n !== node);
  }

  function deleteNode(node) {
    graph.nodes = graph.nodes.filter((n) => n !== node);
    graph.edges = graph.edges.filter((e) => e.a !== node.id && e.b !== node.id);
    scheduleSave();
  }

  function addEdge(a, b) {
    if (a === b) return;
    if (graph.edges.some((e) => (e.a === a && e.b === b) || (e.a === b && e.b === a))) return;
    graph.edges.push({ a, b });
    scheduleSave();
  }

  if (addNodeBtn) addNodeBtn.addEventListener("click", addNode);
  if (nodeInput) {
    nodeInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        commitLabel();
      } else if (e.key === "Escape") {
        e.preventDefault();
        cancelLabel();
      }
    });
    nodeInput.addEventListener("blur", commitLabel);
  }

  // ---- Focus / interaction ----------------------------------------------
  function setFocus(index) {
    focusIndex = index;
    focusCustom = null;
    if (index != null) {
      openPanel(themes[index], themes[index]);
      hintEl.textContent = "click empty space to return";
    } else {
      closePanel();
      hintEl.textContent = "click a theme to read · drag a node's ⊕ to connect";
    }
  }

  // Focus a user-added node: zoom to it and open its notes (stored under its id,
  // exactly like a theme's notes).
  function focusCustomNode(node) {
    focusIndex = null;
    focusCustom = node;
    openPanel(node.id, node.label || "untitled");
    hintEl.textContent = "click empty space to return · double-click to rename";
  }

  function nodeAt(sx, sy) {
    let best = null;
    let bestD = 44; // generous hit radius (px)
    for (const node of nodes) {
      if (isHidden(node) || node.alpha < 0.2) continue;
      const [x, y] = project(node.nx, node.ny);
      const d = Math.hypot(sx - x, sy - y);
      if (d < bestD) {
        bestD = d;
        best = node;
      }
    }
    return best;
  }

  function canvasXY(e) {
    const rect = canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  function capture(id) {
    try {
      canvas.setPointerCapture(id);
    } catch (_) {
      /* pointer capture is best-effort */
    }
  }

  // Delete a custom node or connection under the cursor. Returns true if it hit
  // something. Called from both pointerdown (ctrl/⌘/right) and contextmenu, since
  // Firefox on macOS routes ctrl-click to contextmenu rather than pointerdown.
  function tryDelete(sx, sy) {
    const delNode = customNodeAt(sx, sy);
    if (delNode) {
      if (focusCustom === delNode) setFocus(null);
      deleteNode(delNode);
      return true;
    }
    const ei = edgeAt(sx, sy);
    if (ei >= 0) {
      graph.edges.splice(ei, 1);
      scheduleSave();
      return true;
    }
    return false;
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (editing || labelling) return; // don't disturb an open editor
    const [sx, sy] = canvasXY(e);
    pointer.x = sx;
    pointer.y = sy;

    // ctrl/⌘/right-click: delete a custom node or connection.
    if (e.ctrlKey || e.metaKey || e.button === 2) {
      tryDelete(sx, sy);
      return;
    }

    // A node's link handle → start drawing a connection (from a theme or a
    // custom node; the target can be any node).
    const fromId = handleAt(sx, sy);
    if (fromId) {
      drag = { mode: "connect", from: fromId };
      capture(e.pointerId);
      return;
    }

    // A custom node body → move it.
    const cn = customNodeAt(sx, sy);
    if (cn) {
      drag = { mode: "move", node: cn, moved: false, sx, sy };
      capture(e.pointerId);
      return;
    }

    // Otherwise a potential theme click, resolved on pointerup.
    drag = { mode: "click", sx, sy };
  });

  canvas.addEventListener("pointermove", (e) => {
    const [sx, sy] = canvasXY(e);
    pointer.x = sx;
    pointer.y = sy;
    if (drag && drag.mode === "move") {
      const [wx, wy] = worldFromScreen(sx, sy);
      drag.node.x = wx;
      drag.node.y = wy;
      drag.moved = true;
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    const [sx, sy] = canvasXY(e);
    if (d.mode === "connect") {
      const target = anyNodeIdAt(sx, sy, d.from);
      if (target) addEdge(d.from, target);
      return;
    }
    if (d.mode === "move") {
      if (d.moved) scheduleSave();
      else focusCustomNode(d.node); // a tap (no drag) opens its notes
      return;
    }
    // click → theme focus, only if the pointer barely moved
    if (editing || Math.hypot(sx - d.sx, sy - d.sy) > 4) return;
    const hit = nodeAt(sx, sy);
    if (hit) setFocus(focusIndex === hit.index ? null : hit.index);
    else if (focusIndex != null || focusCustom) setFocus(null);
  });

  // Double-click a custom node to rename it.
  canvas.addEventListener("dblclick", (e) => {
    if (editing || labelling) return;
    const [sx, sy] = canvasXY(e);
    const cn = customNodeAt(sx, sy);
    if (cn) startLabel(cn, false);
  });

  canvas.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    const [sx, sy] = canvasXY(e);
    tryDelete(sx, sy);
  });

  // ---- Drawing ----------------------------------------------------------
  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const [cx, cy] = project(0, 0);

    // Edges from the centre out to each theme (overview only).
    if (focusIndex == null && !focusCustom) {
      ctx.strokeStyle = "rgba(0,0,0,0.12)";
      ctx.lineWidth = 1;
      for (const node of nodes) {
        if (node.index === 0) continue;
        const [x, y] = project(node.nx, node.ny);
        ctx.globalAlpha = node.alpha * 0.9;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    } else if (focusIndex != null) {
      // Focused: retain just the line from this theme back toward the centre,
      // drawn long so it trails off-screen — a quiet sense of place in the
      // network while the notes take over.
      const fn = nodes[focusIndex];
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

    for (const node of nodes) {
      if (node.alpha < 0.01) continue;
      const [x, y] = project(node.nx, node.ny);
      const focused = node.index === focusIndex;
      const isCentre = node.index === 0;
      // Focused node is kept small (a quiet marker) — the bulk of the content is
      // the notes panel, so the diagram steps back when zoomed in.
      const radius = (focused ? 9 : isCentre ? 9 : 6) + node.pulse * 4;

      ctx.globalAlpha = node.alpha;
      ctx.fillStyle = "#0a0a0a";
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      // Ring around the focused / centre node for emphasis.
      if (focused || isCentre) {
        ctx.globalAlpha = node.alpha * 0.5;
        ctx.strokeStyle = "#0a0a0a";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, radius + 7, 0, Math.PI * 2);
        ctx.stroke();
      }

      const fontSize = focused ? 15 : isCentre ? 17 : 14;
      ctx.globalAlpha = node.alpha;
      ctx.fillStyle = "#0a0a0a";
      ctx.font = `600 ${fontSize}px ${
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
      }`;
      // Letter-spaced caps to match the editorial style.
      drawSpacedCaps(node.label.toUpperCase(), x, y + radius + fontSize, fontSize * 0.14);

      // In overview, every theme gets a link handle too, so core concepts can
      // be connected to each other (and to added nodes), not just the new ones.
      if (focusIndex == null && !focusCustom) drawHandle(x, y);
    }

    drawGraph();
    ctx.globalAlpha = 1;
  }

  // A small open ring with a plus, upper-right of a node — the grab point for
  // drawing a new connection. Shared by theme and custom nodes.
  function drawHandle(x, y) {
    const hx = x + HANDLE_DX;
    const hy = y + HANDLE_DY;
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(hx, hy, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(hx - 2.5, hy);
    ctx.lineTo(hx + 2.5, hy);
    ctx.moveTo(hx, hy - 2.5);
    ctx.lineTo(hx, hy + 2.5);
    ctx.stroke();
  }

  const GRAPH_SANS =
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";

  // The curated layer: user-added nodes (hollow markers) and their connections,
  // each with a small link handle for drawing new edges.
  function drawGraph() {
    ctx.globalAlpha = 1;

    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1.2;
    for (const e of graph.edges) {
      // When a custom node is focused, only draw its own connections.
      if (focusCustom && e.a !== focusCustom.id && e.b !== focusCustom.id) continue;
      const a = nodePosById(e.a);
      const b = nodePosById(e.b);
      if (!a || !b) continue;
      const [ax, ay] = project(a[0], a[1]);
      const [bx, by] = project(b[0], b[1]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }

    if (drag && drag.mode === "connect") {
      const a = nodePosById(drag.from);
      if (a) {
        const [ax, ay] = project(a[0], a[1]);
        ctx.strokeStyle = "rgba(0,0,0,0.5)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(pointer.x, pointer.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    for (const n of graph.nodes) {
      // When a custom node is focused, only draw that node.
      if (focusCustom && n !== focusCustom) continue;
      const [x, y] = project(n.x, n.y);
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = "#0a0a0a";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      if (n === focusCustom) {
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(x, y, 13, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      if (n.label) {
        ctx.fillStyle = "#0a0a0a";
        ctx.font = `600 13px ${GRAPH_SANS}`;
        drawSpacedCaps(n.label.toUpperCase(), x, y + 19, 13 * 0.12);
      }

      // Link handle (open ring with a plus), upper-right of the node.
      drawHandle(x, y);
    }
    ctx.globalAlpha = 1;
  }

  function drawSpacedCaps(text, cx, y, tracking) {
    const widths = [];
    let total = 0;
    for (const ch of text) {
      const w = ctx.measureText(ch).width + tracking;
      widths.push(w);
      total += w;
    }
    let x = cx - total / 2;
    const prevAlign = ctx.textAlign;
    ctx.textAlign = "left";
    for (let i = 0; i < text.length; i++) {
      ctx.fillText(text[i], x, y);
      x += widths[i];
    }
    ctx.textAlign = prevAlign;
  }

  function tick() {
    const focused = focusIndex != null || focusCustom != null;
    const zoomTarget = focused ? FOCUS_ZOOM : 1;
    zoom += (zoomTarget - zoom) * 0.08;

    // Pan so the focused node tucks into the open area left of the notes panel
    // (small). In overview the camera is centred and unzoomed.
    let panX = 0;
    let panY = 0;
    let fnx = null;
    let fny = null;
    if (focusIndex != null && nodes[focusIndex]) {
      fnx = nodes[focusIndex].nx;
      fny = nodes[focusIndex].ny;
    } else if (focusCustom) {
      fnx = focusCustom.x;
      fny = focusCustom.y;
    }
    if (fnx != null) {
      // Keep in sync with #panel width in presentation.css. The node tucks into
      // the narrow strip left of the (wide) notes panel.
      const panelW = Math.min(600, width * 0.54);
      const anchorX = (width - panelW) * 0.46;
      const anchorY = height * 0.52;
      const s = scale();
      panX = anchorX - width / 2 - fnx * s;
      panY = anchorY - height / 2 - fny * s;
    }
    camOffsetX += (panX - camOffsetX) * 0.08;
    camOffsetY += (panY - camOffsetY) * 0.08;

    for (const node of nodes) {
      const [tx, ty] = targetPos(node);
      node.nx += (tx - node.nx) * 0.12;
      node.ny += (ty - node.ny) * 0.12;
      const targetAlpha = isHidden(node) ? 0 : 1;
      node.alpha += (targetAlpha - node.alpha) * 0.12;
      node.pulse *= 0.9;
    }

    draw();
    requestAnimationFrame(tick);
  }

  // ---- Boot -------------------------------------------------------------
  async function fetchThemes() {
    // Prefer the standalone PHP store on the website (so the page works with the
    // laptop off); fall back to the app's /themes (served locally or via tunnel).
    try {
      const r = await fetch("notes.php?themes=1", { cache: "no-store" });
      if (r.ok && (r.headers.get("content-type") || "").includes("json")) {
        usePhp = true;
        return (await r.json()).themes || [];
      }
    } catch (_) {
      /* no PHP store here — use the app */
    }
    apiBase = await resolveApiBase();
    const r = await fetch(`${apiBase}/themes`, { cache: "no-store" });
    return (await r.json()).themes || [];
  }

  async function load() {
    try {
      themes = await fetchThemes();
    } catch (_) {
      hintEl.textContent = "could not reach the server";
      return;
    }
    // References and the curated graph are online-only features backed by
    // notes.php; hide their controls when served by the app (no PHP store).
    if (!usePhp) {
      if (refsToggle) refsToggle.style.display = "none";
      if (refs) refs.style.display = "none";
      if (addNodeBtn) addNodeBtn.style.display = "none";
    } else {
      loadGraph();
    }
    nodes = themes.map((label, index) => {
      const [nx, ny] = ringPos(index, themes.length);
      return { index, label, nx, ny, alpha: 0, pulse: 1 };
    });
  }

  load();
  requestAnimationFrame(tick);
})();
