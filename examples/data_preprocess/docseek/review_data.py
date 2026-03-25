#!/usr/bin/env python3
"""
DocSeek Data Quality Viewer — Web-based visual inspection for training data.
Displays document images with bbox overlays, question/GT, and labeling controls.

Usage:
    python review_data.py --parquet data/docseek/v2/train.parquet --port 8899

Then open http://<server>:8899 in your browser.
Keyboard: ← → navigate | R random | B mark bad | Space next
"""
from __future__ import annotations
import argparse
import json
import mimetypes
import os
import re
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
SAMPLES: list[dict] = []
LABELS: dict[int, dict] = {}
LABELS_PATH: str = ""


def load_parquet(path: str) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(path)
    samples = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        prompt = row.get("prompt", [])
        extra = row.get("extra_info", {})
        reward = row.get("reward_model", {})

        question = ""
        try:
            if len(prompt) > 1:
                question = prompt[1].get("content", "")
        except (TypeError, KeyError, IndexError):
            pass

        images = row.get("images", [])
        image_path = ""
        try:
            if len(images) > 0:
                item = images[0]
                if isinstance(item, dict):
                    image_path = item.get("image", "")
                elif isinstance(item, str):
                    image_path = item
        except (TypeError, KeyError):
            pass

        gt = reward.get("ground_truth", "") if isinstance(reward, dict) else ""
        task_type = extra.get("task_type", "unknown") if isinstance(extra, dict) else "unknown"
        element_type = extra.get("element_type", "") if isinstance(extra, dict) else ""
        doc_id = extra.get("doc_id", "") if isinstance(extra, dict) else ""
        data_source = row.get("data_source", "")

        # Extract bbox
        bbox = None
        if task_type == "gnd" and isinstance(gt, str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", gt)
            if len(nums) >= 4:
                bbox = [int(float(n)) for n in nums[:4]]
        elif task_type == "ocr":
            match = re.search(r"region\s*\[([^\]]+)\]", question)
            if match:
                nums = [n.strip() for n in match.group(1).split(",")]
                if len(nums) >= 4:
                    try:
                        bbox = [int(float(n)) for n in nums[:4]]
                    except ValueError:
                        pass

        # Clean question for display
        q_display = question.replace("<image>\n", "").strip()
        parts = q_display.split("Guidelines:")
        q_short = parts[0].strip()
        guidelines = parts[1].strip() if len(parts) > 1 else ""

        samples.append({
            "index": i,
            "data_source": data_source,
            "task_type": task_type,
            "element_type": element_type or "",
            "doc_id": doc_id or "",
            "question": q_short,
            "guidelines": guidelines,
            "ground_truth": gt if isinstance(gt, str) else json.dumps(gt, ensure_ascii=False),
            "image_path": image_path,
            "bbox": bbox,
            "label": "",
        })
    return samples


def load_labels(path: str) -> dict[int, dict]:
    labels = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    labels[item["index"]] = item
    return labels


def save_label(index: int, label: str, corrected_bbox=None, note: str = ""):
    entry = {
        "index": index, "label": label,
        "corrected_bbox": corrected_bbox, "note": note,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    LABELS[index] = entry
    with open(LABELS_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocSeek Quality Studio</title>
<style>
:root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
    --text: #c9d1d9; --text2: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #d29922;
    --border: #30363d;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }
.header {
    background: var(--bg2); border-bottom: 1px solid var(--border);
    padding: 12px 24px; display: flex; align-items: center; gap: 20px;
    flex-wrap: wrap; position: sticky; top: 0; z-index: 100;
}
.header h1 { font-size: 18px; color: var(--accent); white-space: nowrap; }
.stats { font-size: 13px; color: var(--text2); }
.stats b { color: var(--text); }
.filters {
    background: var(--bg2); border-bottom: 1px solid var(--border);
    padding: 10px 24px; display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
}
.filters label { font-size: 12px; color: var(--text2); }
.filters select, .filters input {
    background: var(--bg3); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 8px; font-size: 13px;
}
.nav {
    background: var(--bg2); border-bottom: 1px solid var(--border);
    padding: 8px 24px; display: flex; gap: 8px; align-items: center;
}
.nav button {
    background: var(--bg3); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 13px;
}
.nav button:hover { background: var(--border); }
.nav .pos { font-size: 13px; color: var(--text2); margin: 0 8px; }
.btn-bad {
    background: var(--red) !important; color: #fff !important;
    border: none !important; font-weight: 600; padding: 6px 18px !important;
}
.btn-bad:hover { opacity: 0.85; }
.btn-good {
    background: var(--green) !important; color: #fff !important;
    border: none !important; font-weight: 600; padding: 6px 18px !important;
}
.btn-good:hover { opacity: 0.85; }
.btn-undo {
    background: var(--bg3) !important; color: var(--green) !important;
    border: 1px solid var(--green) !important;
}
.badge-good { background: rgba(63,185,80,0.25); color: var(--green); }
.main { display: flex; height: calc(100vh - 140px); }
.left-panel { flex: 0 0 55%; max-width: 55%; overflow: auto; padding: 16px; }
.right-panel { flex: 1; overflow: auto; padding: 16px; border-left: 1px solid var(--border); }
.img-container {
    position: relative; display: inline-block; max-width: 100%;
    background: #000; border-radius: 8px; overflow: hidden;
}
.img-container img { max-width: 100%; height: auto; display: block; }
#bbox-canvas {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 10;
}
#draw-canvas {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    z-index: 20; cursor: crosshair;
}
.meta {
    margin: 12px 0; padding: 10px; background: var(--bg2);
    border-radius: 8px; font-size: 13px; display: grid;
    grid-template-columns: auto 1fr; gap: 4px 12px;
}
.meta .key { color: var(--text2); }
.meta .val { color: var(--text); }
.badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600;
}
.badge-vqa { background: rgba(88,166,255,0.15); color: var(--accent); }
.badge-ocr { background: rgba(210,153,34,0.15); color: var(--orange); }
.badge-gnd { background: rgba(63,185,80,0.15); color: var(--green); }
.badge-bad { background: rgba(248,81,73,0.25); color: var(--red); }
.question-box {
    background: var(--bg3); border-radius: 8px; padding: 12px 16px;
    margin-bottom: 12px; font-size: 14px; border-left: 3px solid var(--accent);
    white-space: pre-wrap; word-break: break-word;
}
.gt-box {
    background: var(--bg2); border-radius: 8px; padding: 12px 16px;
    margin-bottom: 12px; font-size: 14px; border-left: 3px solid var(--green);
    font-family: 'Fira Code', monospace; white-space: pre-wrap; word-break: break-word;
    color: var(--green);
}
#draw-info {
    margin: 8px 0; padding: 8px 12px; background: var(--bg3);
    border-radius: 6px; font-size: 13px; font-family: monospace; display: none;
}
.help-bar {
    margin: 6px 0; padding: 6px 12px; background: var(--bg2);
    border-radius: 6px; font-size: 12px; color: var(--text2);
}
.help-bar b { color: var(--accent); }
@media (max-width: 1000px) {
    .main { flex-direction: column; height: auto; }
    .left-panel, .right-panel { max-width: 100%; flex: auto; }
}
</style>
</head>
<body>
<div class="header">
    <h1>🔍 DocSeek Quality Studio</h1>
    <div class="stats" id="stats"></div>
</div>
<div class="filters">
    <label>Source:</label>
    <select id="f-source"><option value="">All</option></select>
    <label>Task:</label>
    <select id="f-task"><option value="">All</option></select>
    <label>Element:</label>
    <select id="f-elem"><option value="">All</option></select>
    <label>Label:</label>
    <select id="f-label">
        <option value="">All</option>
        <option value="unlabeled">Unlabeled</option>
        <option value="good">Good</option>
        <option value="bad">Bad</option>
    </select>
    <label>Search:</label>
    <input id="f-search" type="text" placeholder="question / GT..." style="width:160px">
    <button onclick="applyFilters()" style="background:var(--accent);color:#fff;border:none;border-radius:6px;padding:4px 12px;cursor:pointer">Filter</button>
</div>
<div class="nav">
    <button onclick="go(-10)">⏪ -10</button>
    <button onclick="go(-1)">◀ Prev</button>
    <span class="pos" id="pos">0 / 0</span>
    <button onclick="go(1)">Next ▶</button>
    <button onclick="go(10)">+10 ⏩</button>
    <button onclick="goRandom()" style="margin-left:12px">🎲 Random</button>
    <button id="btn-good" class="btn-good" onclick="markGood()" style="margin-left:auto">✅ OK (Space)</button>
    <button id="btn-bad" class="btn-bad" onclick="markBad()">❌ BAD (B)</button>
</div>
<div class="main">
    <div class="left-panel">
        <div class="img-container" id="img-container">
            <img id="doc-img" src="" alt="Document Image">
            <canvas id="bbox-canvas"></canvas>
            <canvas id="draw-canvas"></canvas>
        </div>
        <div id="draw-info">
            <span style="color:var(--orange)">Your bbox [0-1000]:</span>
            <span id="draw-coords" style="color:#fff;font-weight:bold"></span>
            <button onclick="saveBbox()" style="margin-left:12px;background:var(--green);color:#fff;border:none;border-radius:6px;padding:4px 14px;cursor:pointer;font-weight:600">Save bbox (S)</button>
            <button onclick="clearDraw()" style="margin-left:6px;background:var(--bg3);color:var(--text2);border:1px solid var(--border);border-radius:6px;padding:4px 10px;cursor:pointer">Clear</button>
        </div>
        <div class="help-bar">
            Draw bbox on image to correct | <b>←→</b> navigate | <b>R</b> random | <b>B</b> bad | <b>S</b> save bbox | <b>Space</b> next
            <span id="bad-count" style="float:right;color:var(--red);font-weight:600"></span>
        </div>
        <div class="meta" id="meta"></div>
    </div>
    <div class="right-panel">
        <h3 style="margin:0 0 8px;font-size:14px;color:var(--text2)">Question</h3>
        <div class="question-box" id="question-box"></div>
        <h3 style="margin:0 0 8px;font-size:14px;color:var(--text2)">Ground Truth</h3>
        <div class="gt-box" id="gt-box"></div>
        <div id="guidelines-section" style="display:none">
            <h3 style="margin:12px 0 8px;font-size:14px;color:var(--text2);cursor:pointer" onclick="toggleGuidelines()">▸ Guidelines</h3>
            <div id="guidelines-box" style="display:none;background:var(--bg2);border-radius:8px;padding:12px;font-size:12px;color:var(--text2)"></div>
        </div>
    </div>
</div>
<script>
let totalCount = 0;
let filteredIndices = [];
let ci = 0;  // position in filteredIndices
let currentSample = null;
let userBbox = null;
let drawing = false;
let drawStart = null;
let badCount = 0;

async function init() {
    const r = await fetch('/api/meta');
    const meta = await r.json();
    totalCount = meta.total;
    const add = (id, vals) => { const el = document.getElementById(id); vals.forEach(v => { const o = document.createElement('option'); o.value = v; o.textContent = v; el.appendChild(o); }); };
    add('f-source', meta.sources);
    add('f-task', meta.tasks);
    add('f-elem', meta.elements);
    applyFilters();
}

async function applyFilters() {
    const p = new URLSearchParams();
    const src = document.getElementById('f-source').value;
    const task = document.getElementById('f-task').value;
    const elem = document.getElementById('f-elem').value;
    const label = document.getElementById('f-label').value;
    const search = document.getElementById('f-search').value;
    if (src) p.set('source', src);
    if (task) p.set('task', task);
    if (elem) p.set('element', elem);
    if (label) p.set('label', label);
    if (search) p.set('q', search);
    const r = await fetch('/api/filtered?' + p.toString());
    const data = await r.json();
    filteredIndices = data.indices;
    document.getElementById('stats').innerHTML =
        `Showing <b>${filteredIndices.length}</b> / ${totalCount}`;
    ci = 0;
    loadCurrent();
}

function go(d) { if (!filteredIndices.length) return; ci = Math.max(0, Math.min(filteredIndices.length-1, ci+d)); loadCurrent(); }
function goRandom() { if (!filteredIndices.length) return; ci = Math.floor(Math.random()*filteredIndices.length); loadCurrent(); }

async function loadCurrent() {
    if (!filteredIndices.length) {
        document.getElementById('pos').textContent = '0 / 0';
        return;
    }
    const idx = filteredIndices[ci];
    const r = await fetch('/api/sample?index=' + idx);
    currentSample = await r.json();
    render();
}

function render() {
    const s = currentSample;
    if (!s) return;
    document.getElementById('pos').textContent = `${ci+1} / ${filteredIndices.length}  (#${s.index})`;

    // Image
    const img = document.getElementById('doc-img');
    img.src = '/api/image?path=' + encodeURIComponent(s.image_path);
    img.onload = () => { drawBbox(s); initDraw(); clearDraw(); };

    // Meta
    const tb = `<span class="badge badge-${s.task_type}">${s.task_type}</span>`;
    const lb = s.label === 'bad' ? ' <span class="badge badge-bad">BAD</span>' :
               s.label === 'good' ? ' <span class="badge badge-good">OK</span>' : '';
    const bb = s.bbox ? `[${s.bbox.join(', ')}]` : '—';
    document.getElementById('meta').innerHTML = `
        <span class="key">Index</span><span class="val">#${s.index}</span>
        <span class="key">Source</span><span class="val">${s.data_source}</span>
        <span class="key">Task</span><span class="val">${tb}${lb}</span>
        <span class="key">Element</span><span class="val">${s.element_type || '—'}</span>
        <span class="key">Doc</span><span class="val">${s.doc_id || '—'}</span>
        <span class="key">Bbox</span><span class="val">${bb}</span>
    `;

    // Question & GT
    document.getElementById('question-box').textContent = s.question;
    document.getElementById('gt-box').textContent = s.ground_truth;

    // Guidelines
    if (s.guidelines) {
        document.getElementById('guidelines-section').style.display = 'block';
        document.getElementById('guidelines-box').textContent = s.guidelines;
    } else {
        document.getElementById('guidelines-section').style.display = 'none';
    }

    // Button states
    const btnGood = document.getElementById('btn-good');
    const btnBad = document.getElementById('btn-bad');
    if (s.label === 'bad' || s.label === 'good') {
        btnGood.textContent = '↩️ Undo';
        btnGood.className = 'btn-undo';
        btnGood.onclick = undoLabel;
        btnBad.style.display = 'none';
    } else {
        btnGood.textContent = '✅ OK (Space)';
        btnGood.className = 'btn-good';
        btnGood.onclick = markGood;
        btnBad.textContent = '❌ BAD (B)';
        btnBad.className = 'btn-bad';
        btnBad.onclick = markBad;
        btnBad.style.display = '';
    }

    document.getElementById('bad-count').textContent = `Bad: ${badCount}`;
}

function drawBbox(s) {
    const img = document.getElementById('doc-img');
    const c = document.getElementById('bbox-canvas');
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    if (!s.bbox) return;
    let [x1,y1,x2,y2] = s.bbox;
    x1 = x1/1000*c.width; y1 = y1/1000*c.height;
    x2 = x2/1000*c.width; y2 = y2/1000*c.height;
    // Dim outside
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.fillRect(0,0,c.width,y1);
    ctx.fillRect(0,y1,x1,y2-y1);
    ctx.fillRect(x2,y1,c.width-x2,y2-y1);
    ctx.fillRect(0,y2,c.width,c.height-y2);
    // Border
    const color = s.task_type === 'gnd' ? '#3fb950' : '#58a6ff';
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(3, c.width/200);
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    // Label
    const fs = Math.max(16, c.width/40);
    ctx.font = `bold ${fs}px sans-serif`;
    const label = s.task_type.toUpperCase();
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1-fs-8, tw+12, fs+6);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1+6, y1-10);
}

async function markGood() {
    if (!currentSample) return;
    const r = await fetch('/api/label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({index: currentSample.index, label: 'good'})
    });
    if ((await r.json()).ok) {
        currentSample.label = 'good';
        go(1);
    }
}

async function markBad() {
    if (!currentSample) return;
    const r = await fetch('/api/label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({index: currentSample.index, label: 'bad'})
    });
    if ((await r.json()).ok) {
        currentSample.label = 'bad';
        badCount++;
        go(1);
    }
}

async function undoLabel() {
    if (!currentSample) return;
    const r = await fetch('/api/label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({index: currentSample.index, label: ''})
    });
    if ((await r.json()).ok) {
        if (currentSample.label === 'bad') badCount = Math.max(0, badCount - 1);
        currentSample.label = '';
        render();
    }
}

async function saveBbox() {
    if (!userBbox || !currentSample) return;
    const r = await fetch('/api/save_bbox', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({index: currentSample.index, bbox: userBbox})
    });
    if ((await r.json()).ok) {
        currentSample.bbox = userBbox;
        clearDraw();
        render();
    }
}

function toggleGuidelines() {
    const el = document.getElementById('guidelines-box');
    el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

// --- Drawing ---
function getDC() { return document.getElementById('draw-canvas'); }
function xy(e) {
    const c = getDC(), r = c.getBoundingClientRect();
    return { x: (e.clientX-r.left)/r.width, y: (e.clientY-r.top)/r.height };
}
function initDraw() {
    const dc = getDC(), img = document.getElementById('doc-img');
    dc.width = img.naturalWidth; dc.height = img.naturalHeight;
    dc.getContext('2d').clearRect(0,0,dc.width,dc.height);
}
function clearDraw() {
    const dc = getDC();
    dc.getContext('2d').clearRect(0,0,dc.width,dc.height);
    document.getElementById('draw-info').style.display = 'none';
    userBbox = null; drawing = false;
}

getDC().addEventListener('mousedown', e => { if(e.button===2){clearDraw();return;} drawing=true; drawStart=xy(e); });
getDC().addEventListener('mousemove', e => {
    if (!drawing) return;
    const cur = xy(e), dc = getDC(), ctx = dc.getContext('2d');
    ctx.clearRect(0,0,dc.width,dc.height);
    const x1=Math.min(drawStart.x,cur.x)*dc.width, y1=Math.min(drawStart.y,cur.y)*dc.height;
    const x2=Math.max(drawStart.x,cur.x)*dc.width, y2=Math.max(drawStart.y,cur.y)*dc.height;
    ctx.strokeStyle='#ff8800'; ctx.lineWidth=Math.max(2,dc.width/250);
    ctx.setLineDash([6,3]); ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.fillStyle='rgba(255,136,0,0.12)'; ctx.fillRect(x1,y1,x2-x1,y2-y1);
    const c = [Math.round(Math.min(drawStart.x,cur.x)*1000), Math.round(Math.min(drawStart.y,cur.y)*1000),
               Math.round(Math.max(drawStart.x,cur.x)*1000), Math.round(Math.max(drawStart.y,cur.y)*1000)];
    document.getElementById('draw-coords').textContent = `[${c.join(', ')}]`;
    document.getElementById('draw-info').style.display = 'block';
});
getDC().addEventListener('mouseup', e => {
    if (!drawing) return; drawing = false;
    const end = xy(e), dc = getDC(), ctx = dc.getContext('2d');
    ctx.clearRect(0,0,dc.width,dc.height);
    const x1=Math.min(drawStart.x,end.x), y1=Math.min(drawStart.y,end.y);
    const x2=Math.max(drawStart.x,end.x), y2=Math.max(drawStart.y,end.y);
    const px1=x1*dc.width,py1=y1*dc.height,px2=x2*dc.width,py2=y2*dc.height;
    ctx.strokeStyle='#ff8800'; ctx.lineWidth=Math.max(3,dc.width/200);
    ctx.setLineDash([]); ctx.strokeRect(px1,py1,px2-px1,py2-py1);
    ctx.fillStyle='rgba(255,136,0,0.15)'; ctx.fillRect(px1,py1,px2-px1,py2-py1);
    const c = [Math.round(x1*1000),Math.round(y1*1000),Math.round(x2*1000),Math.round(y2*1000)];
    const fs = Math.max(14,dc.width/50);
    ctx.font=`bold ${fs}px sans-serif`;
    const lbl=`[${c.join(',')}]`;
    const tw=ctx.measureText(lbl).width;
    ctx.fillStyle='rgba(255,136,0,0.85)';
    ctx.fillRect(px1,py1-fs-8,tw+12,fs+6);
    ctx.fillStyle='#fff'; ctx.fillText(lbl,px1+6,py1-10);
    userBbox = c;
    document.getElementById('draw-coords').textContent = `[${c.join(', ')}]`;
    document.getElementById('draw-info').style.display = 'block';
});
getDC().addEventListener('contextmenu', e => { e.preventDefault(); clearDraw(); });

// Keyboard
document.addEventListener('keydown', e => {
    if (e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
    if (e.key==='ArrowLeft'||e.key==='a') go(-1);
    else if (e.key==='ArrowRight'||e.key==='d') go(1);
    else if (e.key==='r'||e.key==='R') goRandom();
    else if (e.key==='b'||e.key==='B') markBad();
    else if (e.key==='s'||e.key==='S') saveBbox();
    else if (e.key===' ') { e.preventDefault(); markGood(); }
});

init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and "200" in str(args[0]):
            return
        super().log_message(fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        p = parsed.path
        params = parse_qs(parsed.query)
        if p == "/" or p == "/index.html":
            self._html()
        elif p == "/api/meta":
            # Lightweight: only unique filter values + total count
            sources = sorted(set(s["data_source"] for s in SAMPLES))
            tasks = sorted(set(s["task_type"] for s in SAMPLES))
            elems = sorted(set(s["element_type"] for s in SAMPLES if s["element_type"]))
            self._json({"total": len(SAMPLES), "sources": sources, "tasks": tasks, "elements": elems})
        elif p == "/api/sample":
            # Single sample by index
            idx = int(params.get("index", [0])[0])
            if 0 <= idx < len(SAMPLES):
                self._json(SAMPLES[idx])
            else:
                self._json({"error": "out of range"})
        elif p == "/api/filtered":
            # Server-side filtering, return indices only
            src = params.get("source", [""])[0]
            task = params.get("task", [""])[0]
            elem = params.get("element", [""])[0]
            label = params.get("label", [""])[0]
            kw = params.get("q", [""])[0].lower()
            indices = []
            for s in SAMPLES:
                if src and s["data_source"] != src:
                    continue
                if task and s["task_type"] != task:
                    continue
                if elem and s["element_type"] != elem:
                    continue
                if label == "bad" and s["label"] != "bad":
                    continue
                if label == "good" and s["label"] != "good":
                    continue
                if label == "unlabeled" and s["label"] in ("bad", "good"):
                    continue
                if kw and kw not in s["question"].lower() and kw not in s["ground_truth"].lower():
                    continue
                indices.append(s["index"])
            self._json({"indices": indices, "count": len(indices)})
        elif p == "/api/image":
            self._image(params)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        if parsed.path == "/api/label":
            idx = body["index"]
            label = body["label"]
            save_label(idx, label)
            SAMPLES[idx]["label"] = label
            print(f"  [LABEL] #{idx} → {label or 'cleared'}")
            self._json({"ok": True})
        elif parsed.path == "/api/save_bbox":
            idx = body["index"]
            bbox = body["bbox"]
            save_label(idx, "corrected", corrected_bbox=bbox)
            SAMPLES[idx]["bbox"] = bbox
            print(f"  [BBOX] #{idx} → {bbox}")
            self._json({"ok": True})
        else:
            self.send_error(404)

    def _html(self):
        data = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _image(self, params):
        path = params.get("path", [""])[0]
        if not path or not os.path.isfile(path):
            self.send_error(404, f"Not found: {path}")
            return
        mime, _ = mimetypes.guess_type(path)
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DocSeek Data Quality Viewer")
    parser.add_argument("--parquet", required=True, help="Training parquet file")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    global SAMPLES, LABELS, LABELS_PATH
    LABELS_PATH = str(Path(args.parquet).parent / "quality_labels.jsonl")

    print(f"Loading {args.parquet}...")
    SAMPLES = load_parquet(args.parquet)

    # Load existing labels
    LABELS = load_labels(LABELS_PATH)
    for idx, lb in LABELS.items():
        if idx < len(SAMPLES):
            SAMPLES[idx]["label"] = lb.get("label", "")
            if lb.get("corrected_bbox"):
                SAMPLES[idx]["bbox"] = lb["corrected_bbox"]

    tc = {}
    for s in SAMPLES:
        tc[s["task_type"]] = tc.get(s["task_type"], 0) + 1
    bad = sum(1 for s in SAMPLES if s["label"] == "bad")

    print(f"\n{'='*50}")
    print(f"  DocSeek Quality Studio")
    print(f"  Samples: {len(SAMPLES)}")
    print(f"  Tasks: {tc}")
    print(f"  Bad: {bad}")
    print(f"  Labels: {LABELS_PATH}")
    print(f"\n  Open: http://{args.host}:{args.port}")
    print(f"  Keys: ← → navigate | R random | B bad | S save bbox")
    print(f"{'='*50}\n")

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
