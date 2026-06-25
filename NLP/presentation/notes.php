<?php
/*
 * Standalone notes store for the /presentation page on static hosting.
 *
 * Lets the editable presentation work entirely on the website — no laptop app
 * required — by reading/writing one Markdown file per theme on the server.
 * Upload this next to presentation.html in the /presentation/ folder.
 *
 *   GET  notes.php?themes=1        -> {"themes": [...]}            (JSON)
 *   GET  notes.php?theme=skill     -> the saved notes, or the prepared default
 *   POST notes.php?theme=skill     -> save the request body, returns "ok"
 *   GET  notes.php?graph=1         -> {"nodes": [...], "edges": [...]}  (JSON)
 *   POST notes.php?graph=1         -> save the curated graph, returns "ok"
 *   GET  notes.php?transcript=<id> -> the saved audio transcript for a session
 *   POST notes.php?transcript=<id> -> archive that session's transcript, "ok"
 *
 * The prepared notes are embedded below as defaults, so the page shows the full
 * text out of the box. The first edit to a theme writes ./notes/<theme>.md,
 * which then takes precedence and persists between sessions. Same-origin; no
 * auth (the prepared theme notes aren't secret). Theme names are validated.
 *
 * The presentation's references list reuses this store under the reserved key
 * 'references' (GET/POST notes.php?theme=references) — it is not a theme node.
 */

$THEMES = array(
    'creativity', 'desire', 'process', 'skill',
    'materiality', 'communication', 'technology',
);
$DIR = __DIR__ . '/notes';

// --- Prepared default notes (migrated from content/*.md) --------------------
$DEFAULTS = array();

$DEFAULTS['creativity'] = <<<'MD'
Creativity is a process of expressing intentionality through competence over the material of one’s domain.
MD;

$DEFAULTS['desire'] = <<<'MD'
Is creativity a type of desire? Does AI dull desire?

Is there too much mediation in the way one engages with AI, or does it help extract what this desire is about?
MD;

$DEFAULTS['process'] = <<<'MD'
Is the creative process part of identity, attitude, style (a process of non-process)?

Does AI make creativity too procedural?

In other words, does process take precedence when engaging with AI, and "attitude" / or style recede?
MD;

$DEFAULTS['skill'] = <<<'MD'
Is skill a necessary condition for creativity?

What about openness – breaking it open, being free from the constraints of the "domain"?

What happens when the "skill" is farmed out to AI?

Is the result up-skilling, de-skilling, or something that requires an entirely new vocabulary?
MD;

$DEFAULTS['materiality'] = <<<'MD'
Is the vulnerability of the material – its mortality, epigenetic atmosphere, material historicity – intrinsically connected to creativity?

Does the probabilistic nature of AI prevent it from being thought of as a material?
MD;

$DEFAULTS['communication'] = <<<'MD'
Is a desire to communicate an inalienable aspect of creativity?

Communicate with whom – the self, the divine, the other, a community?

How does communicating with or for a mechanistic logic change desire, process, confidence in one's domain, and the materiality of thinking?
MD;

$DEFAULTS['technology'] = '';

// The references list lives in the same store under the 'references' key. It is
// not a theme node on the diagram — it surfaces as the editable side panel.
$DEFAULTS['references'] = <<<'MD'
Baum, A., Leahy, R. & Walker, R. (2016). Weaving Worlds. Royal College of Art.
Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space (word2vec).
MD;

function fail($code, $msg) {
    http_response_code($code);
    header('Content-Type: text/plain; charset=utf-8');
    echo $msg;
    exit;
}

// --- CORS ------------------------------------------------------------------
// The projected /display window runs from the laptop (a different origin) and
// reads the curated graph from here, so allow cross-origin reads/writes. The
// stored data isn't secret.
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, PUT, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(204);
    exit;
}

// --- Themes list -----------------------------------------------------------
if (isset($_GET['themes'])) {
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode(array('themes' => $THEMES));
    exit;
}

// --- Curated graph (user-added nodes + connections) ------------------------
// GET  notes.php?graph=1  -> {"nodes":[...],"edges":[...]}  (empty if unset)
// POST notes.php?graph=1  -> save the JSON body, returns "ok"
// Nodes: {id,label,x,y} in normalized world coords. Edges: {a,b} where each end
// is a theme key or a node id. This is the curated layer /live and /display draw
// on top of the live map.
if (isset($_GET['graph'])) {
    $graphPath = $DIR . '/graph.json';
    $method = $_SERVER['REQUEST_METHOD'];
    if ($method === 'GET') {
        header('Content-Type: application/json; charset=utf-8');
        if (is_file($graphPath)) {
            echo file_get_contents($graphPath);
        } else {
            echo '{"nodes":[],"edges":[]}';
        }
        exit;
    }
    if ($method === 'POST' || $method === 'PUT') {
        $body = file_get_contents('php://input');
        $data = json_decode($body, true);
        if (!is_array($data) || !isset($data['nodes']) || !isset($data['edges'])) {
            fail(400, 'invalid graph');
        }
        if (strlen($body) > 200000) {
            fail(413, 'graph too large');
        }
        if (!is_dir($DIR) && !@mkdir($DIR, 0775, true) && !is_dir($DIR)) {
            fail(500, 'cannot create notes directory');
        }
        if (file_put_contents($graphPath, $body) === false) {
            fail(500, 'cannot write graph');
        }
        header('Content-Type: text/plain; charset=utf-8');
        echo 'ok';
        exit;
    }
    fail(405, 'method not allowed');
}

// --- Transcript archive (audio session log) -------------------------------
// GET  notes.php?transcript=<id>  -> the saved markdown for that session (or "")
// POST notes.php?transcript=<id>  -> overwrite notes/transcript-<id>.md, "ok"
// The capture window (/display) posts the growing session transcript here every
// few seconds (one timestamped file per session), so the audio log is archived
// online alongside the curated notes — independent of the laptop staying up.
if (isset($_GET['transcript'])) {
    $id = strtolower(trim($_GET['transcript']));
    if ($id === '' || !preg_match('/^[a-z0-9_-]+$/', $id)) {
        fail(400, 'invalid transcript id');
    }
    $tpath = $DIR . '/transcript-' . $id . '.md';
    $method = $_SERVER['REQUEST_METHOD'];
    if ($method === 'GET') {
        header('Content-Type: text/plain; charset=utf-8');
        echo is_file($tpath) ? file_get_contents($tpath) : '';
        exit;
    }
    if ($method === 'POST' || $method === 'PUT') {
        $body = file_get_contents('php://input');
        if (strlen($body) > 2000000) {
            fail(413, 'transcript too large');
        }
        if (!is_dir($DIR) && !@mkdir($DIR, 0775, true) && !is_dir($DIR)) {
            fail(500, 'cannot create notes directory');
        }
        if (file_put_contents($tpath, $body) === false) {
            fail(500, 'cannot write transcript');
        }
        header('Content-Type: text/plain; charset=utf-8');
        echo 'ok';
        exit;
    }
    fail(405, 'method not allowed');
}

// --- Validate the theme ----------------------------------------------------
$theme = isset($_GET['theme']) ? strtolower(trim($_GET['theme'])) : '';
if ($theme === '' || !preg_match('/^[a-z0-9_-]+$/', $theme)) {
    fail(400, 'invalid theme');
}
$path = $DIR . '/' . $theme . '.md';
$method = $_SERVER['REQUEST_METHOD'];

// --- Read (saved edit wins, else the prepared default) ---------------------
if ($method === 'GET') {
    header('Content-Type: text/plain; charset=utf-8');
    if (is_file($path)) {
        echo file_get_contents($path);
    } elseif (isset($DEFAULTS[$theme])) {
        echo $DEFAULTS[$theme];
    } else {
        echo '';
    }
    exit;
}

// --- Write (POST or PUT) ---------------------------------------------------
if ($method === 'POST' || $method === 'PUT') {
    if (!is_dir($DIR) && !@mkdir($DIR, 0775, true) && !is_dir($DIR)) {
        fail(500, 'cannot create notes directory');
    }
    $body = file_get_contents('php://input');
    if (file_put_contents($path, $body) === false) {
        fail(500, 'cannot write note');
    }
    header('Content-Type: text/plain; charset=utf-8');
    echo 'ok';
    exit;
}

fail(405, 'method not allowed');
