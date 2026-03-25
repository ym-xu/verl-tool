"""
DocSeek Training Data Quality Studio.
Run:
    streamlit run review_data.py -- --parquet data/docseek/v2/train.parquet
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


# ─── Page ────────────────────────────────────────────────────────────────────

def init_page():
    st.set_page_config(page_title="DocSeek Quality Studio", page_icon="🔍",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp {
    background: radial-gradient(1200px 400px at 0% 0%, rgba(255,186,104,0.16), transparent 55%),
                radial-gradient(1000px 400px at 100% 0%, rgba(110,186,255,0.16), transparent 55%),
                linear-gradient(180deg, #f6f8fb 0%, #eef2f7 100%);
}
h1,h2,h3,h4,h5,h6,body,.stMarkdown,.stText {font-family:"Space Grotesk",sans-serif!important}
code,pre {font-family:"IBM Plex Mono",monospace!important}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0f1729,#132a45)}
[data-testid="stSidebar"] * {color:#f2f6ff!important}
.hero-card {
    background:linear-gradient(145deg,rgba(9,31,57,0.93),rgba(20,74,122,0.92));
    border:1px solid rgba(255,255,255,0.08); border-radius:18px;
    padding:16px 18px; color:#f6fbff;
    box-shadow:0 12px 32px rgba(10,25,45,0.18); margin-bottom:1rem;
}
.label-bad {background:#dc2626;color:white;padding:2px 10px;border-radius:6px;font-weight:600}
</style>""", unsafe_allow_html=True)


# ─── Data ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data...")
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def extract_fields(row) -> dict:
    prompt = row.get("prompt", [])
    extra = row.get("extra_info", {})
    reward = row.get("reward_model", {})

    question = ""
    if isinstance(prompt, list) and len(prompt) > 1:
        question = prompt[1].get("content", "")

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

    return dict(data_source=row.get("data_source", ""), task_type=task_type,
                element_type=element_type, doc_id=doc_id, question=question,
                ground_truth=gt, image_path=image_path)


def extract_bbox(fields: dict) -> Optional[list[int]]:
    bbox = None
    if fields["task_type"] == "gnd":
        gt = fields["ground_truth"]
        if isinstance(gt, str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", gt)
            if len(nums) >= 4:
                bbox = [int(float(n)) for n in nums[:4]]
    elif fields["task_type"] == "ocr":
        match = re.search(r"region\s*\[([^\]]+)\]", fields["question"])
        if match:
            nums = [n.strip() for n in match.group(1).split(",")]
            if len(nums) >= 4:
                try:
                    bbox = [int(float(n)) for n in nums[:4]]
                except ValueError:
                    pass
    return bbox


def draw_bbox_on_image(image: Image.Image, bbox: list[int], color: str,
                       label: str = "") -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x1, y1 = int(bbox[0]/1000*w), int(bbox[1]/1000*h)
    x2, y2 = int(bbox[2]/1000*w), int(bbox[3]/1000*h)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if label:
        ty = max(0, y1 - 22)
        draw.rectangle([x1, ty, x1+len(label)*9+8, ty+20], fill=color)
        draw.text((x1+4, ty+2), label, fill="white")
    return img


# ─── Labels ──────────────────────────────────────────────────────────────────

def labels_path(parquet_path: str) -> Path:
    return Path(parquet_path).parent / "quality_labels.jsonl"


def load_labels(parquet_path: str) -> dict[int, dict]:
    lp = labels_path(parquet_path)
    out = {}
    if lp.exists():
        for line in lp.read_text().strip().split("\n"):
            if line.strip():
                item = json.loads(line)
                out[item["index"]] = item
    return out


def save_label(parquet_path: str, index: int, label: str,
               corrected_bbox=None, note: str = ""):
    with open(labels_path(parquet_path), "a") as f:
        f.write(json.dumps(dict(index=index, label=label,
                corrected_bbox=corrected_bbox, note=note,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")),
                ensure_ascii=False) + "\n")


# ─── App ─────────────────────────────────────────────────────────────────────

def main():
    init_page()
    st.markdown("""<div class="hero-card">
<h2 style="margin:0">🔍 DocSeek Quality Studio</h2>
<p style="margin:6px 0 0;opacity:.88">Browse, inspect bbox, flag bad samples.</p>
</div>""", unsafe_allow_html=True)

    # --- Sidebar ---
    default_parquet = "data/docseek/v2/train.parquet"
    if "--parquet" in sys.argv:
        i = sys.argv.index("--parquet")
        if i+1 < len(sys.argv):
            default_parquet = sys.argv[i+1]

    with st.sidebar:
        st.header("Data Source")
        parquet_path = st.text_input("Parquet", value=default_parquet)

    if not Path(parquet_path).exists():
        st.error(f"Not found: {parquet_path}")
        st.stop()

    df = load_parquet(parquet_path)
    labels = load_labels(parquet_path)
    total = len(df)

    # Build lightweight index
    records = []
    for i in range(total):
        row = df.iloc[i].to_dict()
        records.append(dict(
            idx=i,
            data_source=row.get("data_source", ""),
            task_type=(row.get("extra_info") or {}).get("task_type", ""),
            element_type=(row.get("extra_info") or {}).get("element_type", "") or "",
            label=labels.get(i, {}).get("label", ""),
        ))
    index_df = pd.DataFrame(records)

    # --- Sidebar filters ---
    with st.sidebar:
        st.header("Filters")
        sel_sources = st.multiselect("Data Source",
            sorted(index_df["data_source"].unique()), default=sorted(index_df["data_source"].unique()))
        sel_tasks = st.multiselect("Task Type",
            sorted(index_df["task_type"].unique()), default=sorted(index_df["task_type"].unique()))
        etypes = sorted([e for e in index_df["element_type"].unique() if e])
        sel_etypes = st.multiselect("Element Type", etypes, default=etypes) if etypes else []
        sel_label = st.multiselect("Label", ["unlabeled", "bad"],  default=["unlabeled", "bad"])
        kw = st.text_input("Keyword")

    # Filter
    filt = index_df.copy()
    filt = filt[filt["data_source"].isin(sel_sources)]
    filt = filt[filt["task_type"].isin(sel_tasks)]
    if sel_etypes:
        filt = filt[(filt["element_type"].isin(sel_etypes)) | (filt["element_type"] == "")]
    lmask = filt["label"].isin(sel_label) | ((filt["label"] == "") & ("unlabeled" in sel_label))
    filt = filt[lmask]
    if kw.strip():
        # need to check question/gt — do a quick pass
        kw_lower = kw.strip().lower()
        keep_idx = set()
        for idx_val in filt["idx"]:
            row = df.iloc[idx_val].to_dict()
            f = extract_fields(row)
            if kw_lower in f["question"].lower() or kw_lower in str(f["ground_truth"]).lower():
                keep_idx.add(idx_val)
        filt = filt[filt["idx"].isin(keep_idx)]

    kept = len(filt)

    # Stats
    with st.sidebar:
        st.markdown("---")
        st.metric("Total", total)
        st.metric("Showing", kept)
        st.metric("Bad", sum(1 for v in labels.values() if v.get("label") == "bad"))

    # --- Metrics ---
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total", total)
    m2.metric("Showing", kept)
    m3.metric("VQA", len(filt[filt["task_type"] == "vqa"]))
    m4.metric("GND", len(filt[filt["task_type"] == "gnd"]))
    m5.metric("OCR", len(filt[filt["task_type"] == "ocr"]))

    if kept == 0:
        st.warning("No samples match.")
        st.stop()

    # --- Table ---
    st.subheader("Filtered Table")
    st.dataframe(
        filt[["idx", "data_source", "task_type", "element_type", "label"]],
        use_container_width=True, hide_index=True, height=250,
    )

    # --- Sample selector ---
    st.subheader("Sample Detail")
    options = filt["idx"].tolist()

    last = st.session_state.get("sel_idx")
    if last not in options:
        last = options[0]
    default_pos = options.index(last)

    selected = st.selectbox(
        "Select sample",
        options=options,
        index=default_pos,
        format_func=lambda i: (
            f"#{i}  |  {index_df.iloc[i]['data_source']}  |  "
            f"{index_df.iloc[i]['task_type']}  |  "
            f"{index_df.iloc[i]['element_type']}"
            f"{'  🚫' if labels.get(i, {}).get('label') == 'bad' else ''}"
        ),
    )
    st.session_state["sel_idx"] = selected

    # Load full sample
    row = df.iloc[selected].to_dict()
    fields = extract_fields(row)
    bbox = extract_bbox(fields)
    corrected = labels.get(selected, {}).get("corrected_bbox")
    if corrected:
        bbox = corrected

    # --- Detail ---
    c1, c2 = st.columns([1.25, 1.0])

    with c1:
        ip = fields["image_path"]
        if ip and Path(ip).exists():
            image = Image.open(ip).convert("RGB")

            show_bbox = st.checkbox("Show bbox", value=True)
            disp = image
            if bbox and show_bbox:
                color = "#16a34a" if fields["task_type"] == "gnd" else "#0ea5e9"
                disp = draw_bbox_on_image(image, bbox, color, fields["task_type"].upper())

            st.image(disp, use_container_width=True)
            st.caption(f"{Path(ip).name}  |  {image.size[0]}×{image.size[1]}")

            # Bbox editor
            if bbox:
                with st.expander("✏️ Edit bbox"):
                    bc = st.columns(4)
                    nb = [
                        bc[0].number_input("x1", 0, 1000, int(bbox[0]), key="bx1"),
                        bc[1].number_input("y1", 0, 1000, int(bbox[1]), key="by1"),
                        bc[2].number_input("x2", 0, 1000, int(bbox[2]), key="bx2"),
                        bc[3].number_input("y2", 0, 1000, int(bbox[3]), key="by2"),
                    ]
                    if nb != list(bbox):
                        st.image(draw_bbox_on_image(image, nb, "#f59e0b", "EDIT"),
                                 use_container_width=True)
                        if st.button("💾 Save bbox"):
                            save_label(parquet_path, selected, "corrected", corrected_bbox=nb)
                            st.rerun()
        else:
            st.error(f"Image not found: {ip}")

    with c2:
        # Bad button
        cur_label = labels.get(selected, {}).get("label", "")
        if cur_label == "bad":
            st.markdown('<span class="label-bad">BAD</span>', unsafe_allow_html=True)
            if st.button("↩️ Undo", use_container_width=True):
                save_label(parquet_path, selected, "")
                st.rerun()
        else:
            if st.button("❌ Mark BAD", use_container_width=True, type="primary"):
                save_label(parquet_path, selected, "bad")
                # Auto advance
                pos = options.index(selected)
                if pos + 1 < len(options):
                    st.session_state["sel_idx"] = options[pos + 1]
                st.rerun()

        note = st.text_input("Note")
        if note and st.button("Save"):
            save_label(parquet_path, selected, cur_label, note=note)
            st.success("Saved")

        # Question
        st.markdown("---")
        st.markdown("**Question**")
        q = fields["question"].replace("<image>\n", "").strip()
        parts = q.split("Guidelines:")
        st.code(parts[0].strip()[:400], language=None, wrap_lines=True)
        if len(parts) > 1:
            with st.expander("Guidelines"):
                st.caption(parts[1].strip())

        # Ground Truth
        st.markdown("**Ground Truth**")
        gt = fields["ground_truth"]
        if isinstance(gt, list):
            st.code(json.dumps(gt, ensure_ascii=False), language="json", wrap_lines=True)
        else:
            st.code(str(gt), language=None, wrap_lines=True)

        # Info
        st.markdown("**Info**")
        st.json(dict(
            data_source=fields["data_source"], task_type=fields["task_type"],
            element_type=fields["element_type"], doc_id=fields["doc_id"],
            bbox=bbox,
        ))

    # --- Export ---
    st.markdown("---")
    bad_list = sorted(i for i, lb in labels.items() if lb.get("label") == "bad")
    st.metric("Total bad", len(bad_list))
    if bad_list:
        st.download_button("Download bad indices", json.dumps(bad_list),
                           "bad_indices.json", "application/json")


if __name__ == "__main__":
    main()
