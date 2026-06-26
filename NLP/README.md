# Weaving Live Semantic Vector Map
Kevin Walker June 2026,
Developed for https://the-making-of-creativity.com/

A shared web app for live events: participants' browsers transcribe their speech
locally and stream the text to a small server, which surfaces key concepts with
a transformer embedding model and weaves them into an evolving 2D vector map that
everyone sees in real time.

Each browser captures its own microphone (so multiple people can contribute from
their own devices) and the server is the single source of truth for the concept
map and the editable files.

```
browser mic ─▶ Web Speech API (per client) ─▶ recognized text
            ─▶ websocket ─▶ server:
                 KeyBERT key phrases ─▶ MiniLM embeddings
                 ─▶ living concept store (merge · weight · decay)
                 ─▶ anchor-affinity layout + similarity threads
            ─▶ websocket ─▶ all browsers: woven vector map + transcript ribbon
```

## Requirements

- Python 3.11 (3.10+ should work) for the server
- A modern browser for the capture window. Speech transcription uses the **Web
  Speech API** (Chrome / Edge; Safari/Firefox lack it and fall back to the
  optional server-side microphone).
- No microphone, PortAudio or Whisper needed on the server — transcription
  happens in the browser.

> Privacy note: the Web Speech API in Chrome sends microphone audio to the
> browser's cloud speech service. Only recognized **text** reaches our server.

## Setup

```bash
cd workshop
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads one small model (the `all-MiniLM-L6-v2` embedder ~90MB).
Do this once while online; afterwards the embedder runs from the local cache.

## Running

```bash
python -m backend.app
```

There are two distinct front-ends, served by the same app:

- **Capture + projection (or monitor** — <http://127.0.0.1:8000/display>. The single live
  window on the capture machine: white-on-black vector map that records the room
  and projects the result. It is **keyboard-driven** (no on-screen controls):

  | Key     | Action                                  |
  | ------- | --------------------------------------- |
  | `Space` | record / pause the microphone           |
  | `F`     | toggle fullscreen                       |
  | `C`     | clear the map (start a fresh session)   |

  It auto-zooms to whichever theme is being discussed, then drifts back out after
  a quiet stretch, and overlays the curated graph authored on `/presentation`. A
  small dot in the bottom-left corner shows the recording state.
- **Online viewer** — <http://127.0.0.1:8000/>. A black-on-white, editorial
  read-only page (styled after the workshop site) that simply arranges the
  uploaded concepts under the seven category headings as they arrive — no map,
  no canvas, no controls. This is the public-facing version. Read-only is
  enforced on the server: only the capture window may add or change anything.

On `/display`, press **Space** to start transcribing. Concepts are extracted,
embedded and filed under the category each relates to most strongly, and appear
live on every connected online viewer.

The seven categories are the seed anchors (`backend/config.py` →
`anchor_terms`): creativity, desire, process, skill, materiality, communication,
technology.

### True fullscreen (no menu/title bar)

- Press **`F`** on `/display` to enter the browser Fullscreen API, which hides
  all browser and OS chrome.
- For a totally chrome-free display from launch, start Chrome in kiosk mode:

  ```bash
  # macOS — the capture/projection window:
  open -na "Google Chrome" --args --kiosk --app=http://127.0.0.1:8000/display
  # the viewer screens:
  open -na "Google Chrome" --args --kiosk --app=http://<host-ip>:8000/
  ```

### Hosting the viewer statically (e.g. on the workshop site)

The online viewer is plain static files and can be dropped into any folder on a
normal web server (Apache/nginx) — e.g. `https://your-site/live/`. Upload just:

```
live/
├── index.html   (frontend/index.html — the viewer)
├── view.css     (frontend/view.css)
└── view.js      (frontend/view.js)
```

Its asset paths are relative, so it works under a subpath like `/live/`. It does
**not** include the Python server, so point it at wherever the server is running
by editing one line in `index.html`:

```html
<script>
  window.LIVE_WS_URL = "wss://your-server.example/ws";
</script>
```

(or append `?ws=wss://your-server.example/ws` to the URL). Leave it `""` only
when the Python app itself serves the page. The websocket server must be
reachable over `wss://` from the visitor's browser — a static folder alone
cannot provide it (see options below).

## Deploying for multiple clients

- Bind to all interfaces so other devices can reach it: `WW_HOST=0.0.0.0 python -m backend.app`.
- **Serve over HTTPS.** Browsers only allow the Web Speech API / microphone in a
  *secure context*. `http://localhost` is exempt (so the host machine itself is
  fine on localhost), but any other host (an IP or domain) must be HTTPS, e.g.
  behind a TLS-terminating reverse proxy (Caddy, nginx, a tunnel like
  Cloudflare/ngrok, or your host's TLS). The websocket URL is derived from the
  page, so it upgrades to `wss://` automatically. Viewers don't need the mic, so
  plain HTTP is fine for read-only screens.
- All clients share one concept map and transcript (the server holds the state).
  The capture window records and drives the map; every other browser is a
  read-only viewer. The curated graph and notes are authored separately on the
  `/presentation` page and stored online (see `DEPLOY.md`).

## Deploying online (free, via a Cloudflare tunnel)

The simplest free public setup: your **laptop** runs everything (the Python app
+ the ML), a free **Cloudflare quick tunnel** gives it a temporary public
`https`/`wss` address, and the public viewer page is static files on your website
that connect back to the laptop.

```
visitors → your-site.com/live/  ──wss:// (tunnel URL)──▶  laptop: python -m backend.app
                                                            ▲ http://localhost:8000/display
                                                              (you: Space=record, F=fullscreen, C=clear)
```

**One-time:** install the tunnel tool and put the viewer files on your website.

```bash
brew install cloudflared            # the tunnel tool
```

Upload the two folders from the prebuilt `dist/` bundle (see `dist/UPLOAD.md`):
`dist/live/` → your site's `live/` folder, and `dist/presentation/` → its
`presentation/` folder. The viewer reads the laptop's current address from a
one-line `live/ws.txt`.

### The easy way — one command (auto-uploads the URL)

`deploy/run_session.py` starts the app, opens the tunnel, grabs the fresh URL,
and uploads it into `ws.txt` on your website over FTP — nothing to copy by hand.

```bash
cp deploy/infomaniak.example.json deploy/infomaniak.json   # one-time: add FTP creds
python deploy/run_session.py
```

Then open <http://localhost:8000/display> (press `Space` to record) and send
people to `https://your-site.com/live/`. Ctrl-C stops the app and the tunnel.

### The manual way — getting and copying the Cloudflare URL

If you don't use the launcher, do it by hand:

1. **Start the app** on the laptop: `python -m backend.app` (leave it running).
2. **Start the tunnel** in a second terminal:

   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```

   It prints a line with a random URL — **copy it**:

   ```
   https://tiny-forest-1234.trycloudflare.com
   ```

   Keep this terminal open for the whole session (closing it kills the tunnel).
   The URL is **new every run** — this is inherent to free quick tunnels.
3. **Convert it to a websocket URL**: change `https://` → `wss://` and add `/ws`:

   ```
   wss://tiny-forest-1234.trycloudflare.com/ws
   ```

4. **Paste that one line into `live/ws.txt`** on your website (overwrite the old
   contents) and upload it. This is the only file that changes between sessions —
   never the HTML. To test without re-uploading, append it to the viewer URL:
   `https://your-site.com/live/?ws=wss://tiny-forest-1234.trycloudflare.com/ws`.

The capture window stays on `http://localhost:8000/display` (localhost is a
secure context, so the mic works without HTTPS); only the public viewer goes
through the tunnel. Full details, the `/presentation` PHP store, and a stable
named-tunnel option are in [`DEPLOY.md`](DEPLOY.md) and [`dist/UPLOAD.md`](dist/UPLOAD.md).

## Tuning the feel

All knobs live in [`backend/config.py`](backend/config.py) and can be overridden
with environment variables:

| Variable             | Default | Meaning                                              |
| -------------------- | ------- | ---------------------------------------------------- |
| `WW_HOST`            | `127.0.0.1` | Bind address (`0.0.0.0` to expose on a network)  |
| `WW_PORT`            | `8000`  | Server port                                          |
| `WW_MERGE_THRESHOLD` | `0.8`   | Cosine sim above which two phrases are one concept   |
| `WW_DECAY`           | `0.985` | Per-tick weight decay (lower = faster forgetting)    |
| `WW_MAX_CONCEPTS`    | `60`    | Hard cap on concepts on screen                       |
| `WW_EDGE_THRESHOLD`  | `0.35`  | Min similarity to draw a thread between concepts     |
| `WW_ANCHOR_TEMP`     | `0.12`  | Anchor-clustering tightness (lower = tighter)        |
| `WW_BROADCAST`       | `1.0`   | Seconds between map updates                          |

### Legacy local-microphone mode

The original single-machine version captured the mic and ran Whisper on the
server. Those modules remain ([`backend/audio.py`](backend/audio.py),
[`backend/transcribe.py`](backend/transcribe.py)) but are no longer wired in.
To use them you'd re-add the capture loop in `session.py` and install the
optional deps (`sounddevice`, `faster-whisper`) commented out in
`requirements.txt`.

## Presentation (the curated graph)

Curated presentation lives **online only**, on the `/presentation` page — a
black-on-white editor where you arrange the seven themes, add your own nodes,
connect them, and write notes for each. It is stored server-side by `notes.php`
and rendered, read-only, on `/live` and overlaid on the local `/display`. See
`DEPLOY.md` for the page and its `notes.php` store.

The live capture window (`/display`) itself has no presentation mode — it just
records and projects the living map, auto-zooming to the theme under discussion.

## References

References live **online only**, in the `/presentation` page. Open it on the
website and click **References** (bottom-left) to read/edit the list; it is
stored server-side by `notes.php` (under the `references` key) and persists
between sessions. See `DEPLOY.md`. There is no references panel in the local
capture app.

## Session archives

When the app exits it writes a JSON archive of the full transcript and final
concept graph to `data/session-<timestamp>.json` (disable with `WW_EXPORT=0`).
Fitting for a research/art project — each workshop leaves a trace.

## Project layout

```
workshop/
├── requirements.txt
├── README.md
├── backend/
│   ├── config.py       # all tunables
│   ├── concepts.py     # KeyBERT + MiniLM + living concept store
│   ├── projection.py   # anchor-affinity layout + similarity threads
│   ├── session.py      # shared state, ingest text, broadcast, archival
│   ├── app.py          # FastAPI websocket server + file endpoints
│   ├── audio.py        # (legacy) local mic capture — not wired in
│   └── transcribe.py   # (legacy) faster-whisper STT — not wired in
├── frontend/
│   ├── index.html      # online viewer (light, categorized) — served at / (static-host friendly)
│   ├── view.js         # groups uploaded concepts by category, live
│   ├── view.css        # black-on-white editorial theme
│   ├── capture.html    # capture + projection page (dark vector map) — served at /display
│   ├── main.js         # Web Speech capture + Canvas 2D map (no external deps)
│   ├── presentation.html / .js / .css  # curated graph + notes editor (online)
│   └── style.css
├── content/            # legacy per-concept notes (served by /content endpoints)
└── data/               # session exports
```

## Troubleshooting

### `OMP: Error #15: ... libiomp5.dylib already initialized` (macOS)

Seen when launching inside the Anaconda `base` environment (your prompt shows
`(base)`) alongside the venv — two copies of the OpenMP runtime get loaded and
PyTorch aborts. The app now sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically (see
[`backend/__init__.py`](backend/__init__.py)), which resolves it for inference
workloads like this. Just re-run:

```bash
python -m backend.app
```

A cleaner alternative is to keep conda out of the way entirely:

```bash
conda deactivate          # leave the (base) env; keep only the venv active
source .venv/bin/activate
python -m backend.app
```

To opt out of the automatic workaround, set `WW_ALLOW_OMP_DUPLICATE=0`.

### First run seems stuck

On first run the server downloads the embedder (~90MB) once. A moving progress
bar / byte counter means it's working; a true hang shows no output and no network
activity. After caching, startup is quick.

### Pressing Space does nothing / "no mic"

The Web Speech API isn't available or is blocked. Use Chrome or Edge, allow
microphone access when prompted, and make sure the page is on `http://localhost`
or an `https://` URL (not a plain `http://` IP). On browsers without Web Speech
(Firefox/Safari) the capture window falls back to the server's own microphone if
the optional STT deps are installed (`WW_SERVER_MIC`).

## Notes

- The front end uses the Canvas 2D API directly (no D3/CDN).
- All participants share one collective map. Per-speaker colouring would require
  attributing utterances to clients — a natural next step.
