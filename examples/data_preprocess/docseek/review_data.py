"""
DocSeek Training Data Quality Studio.
Interactive viewer for inspecting, labeling, and correcting training data.

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


# ─── Page Setup ──────────────────────────────────────────────────────────────

def init_page():
    st.set_page_config(
        page_title="DocSeek Quality Studio",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp {
    background:
        radial-gradient(1200px 400px at 0% 0%, rgba(255,186,104,0.16), transparent 55%),
        radial-gradient(1000px 400px at 100% 0%, rgba(110,186,255,0.16), transparent 55%),
        linear-gradient(180deg, #f6f8fb 0%, #eef2f7 100%);
    color: #102238;
}
h1, h2, h3, h4, h5, h6, body, .stMarkdown, .stText {
    font-family: "Space Grotesk", sans-serif !important;
}
code, pre {
    font-family: "IBM Plex Mono", monospace !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #132a45 100%);
}
[data-testid="stSidebar"] * {
    color: #f2f6ff !important;
}
.hero-card {
    background: linear-gradient(145deg, rgba(9,31,57,0.93), rgba(20,74,122,0.92));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px 18px;
    color: #f6fbff;
    box-shadow: 0 12px 32px rgba(10, 25, 45, 0.18);
    margin-bottom: 1rem;
}
.hero-sub { opacity: 0.88; font-size: 0.95rem; }
.label-bad { background: #dc2626; color: white; padding: 2px 10px; border-radius: 6px; font-weight: 600; }
.source-tag {
    display: inline-block;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.25);
    padding: 2px 8px; border-radius: 6px;
    font-size: 0.85rem; margin-right: 4px;
}
</style>
    """, unsafe_allow_html=True)


# ─── Data Loading ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data...")
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def extract_fields(row) -> dict:
    """Extract display fields from a parquet row."""
    prompt = row.get("prompt", [])
    extra = row.get("extra_info", {})
    reward = row.get("reward_model", {})

    question = ""
    if isinstance(prompt, list) and len(prompt) > 1:
        question = prompt[1].get("content", "")

    images = row.get("images", [])
    image_path = ""
    if isinstance(images, list) and images:
        if isinstance(images[0], dict):
            image_path = images[0].get("image", "")
        elif isinstance(images[0], str):
            image_path = images[0]

    gt = reward.get("ground_truth", "") if isinstance(reward, dict) else ""
    task_type = extra.get("task_type", "unknown") if isinstance(extra, dict) else "unknown"
    element_type = extra.get("element_type", "") if isinstance(extra, dict) else ""
    doc_id = extra.get("doc_id", "") if isinstance(extra, dict) else ""

    return {
        "data_source": row.get("data_source", ""),
        "task_type": task_type,
        "element_type": element_type,
        "doc_id": doc_id,
        "question": question,
        "ground_truth": gt,
        "image_path": image_path,
    }


def parse_bbox_from_gt(gt: Any) -> Optional[list[int]]:
    """Parse bbox from ground truth string like '[100, 200, 300, 400]'."""
    if isinstance(gt, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", gt)
        if len(nums) >= 4:
            return [int(float(n)) for n in nums[:4]]
    if isinstance(gt, list) and len(gt) >= 4:
        return [int(float(n)) for n in gt[:4]]
    return None


def extract_bbox_for_sample(fields: dict) -> Optional[list[int]]:
    """Extract bbox from GND ground truth or OCR question region."""
    bbox = None
    if fields["task_type"] == "gnd":
        bbox = parse_bbox_from_gt(fields["ground_truth"])
    elif fields["task_type"] == "ocr":
        q = fields["question"]
        match = re.search(r"region\s*\[([^\]]+)\]", q)
        if match:
            nums = [n.strip() for n in match.group(1).split(",")]
            if len(nums) >= 4:
                try:
                    bbox = [int(float(n)) for n in nums[:4]]
                except ValueError:
                    pass
    return bbox


# ─── Drawing ─────────────────────────────────────────────────────────────────

def draw_bbox_on_image(image: Image.Image, bbox: list[int], color: str = "#16a34a",
                       label: str = "", width: int = 3) -> Image.Image:
    """Draw bbox on image. bbox is in 0-1000 coords."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)

    # Draw rectangle with slight transparency effect via double outline
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    draw.rectangle([x1-1, y1-1, x2+1, y2+1], outline=color, width=1)

    if label:
        text_y = max(0, y1 - 22)
        label_w = len(label) * 9 + 8
        draw.rectangle([x1, text_y, x1 + label_w, text_y + 20], fill=color)
        draw.text((x1 + 4, text_y + 2), label, fill="white")

    return img


# ─── Labels Management ───────────────────────────────────────────────────────

def get_labels_path(parquet_path: str) -> Path:
    return Path(parquet_path).parent / "quality_labels.jsonl"


def load_labels(parquet_path: str) -> dict[int, dict]:
    labels_path = get_labels_path(parquet_path)
    labels = {}
    if labels_path.exists():
        for line in labels_path.read_text().strip().split("\n"):
            if line.strip():
                item = json.loads(line)
                labels[item["index"]] = item
    return labels


def save_label(parquet_path: str, index: int, label: str,
               corrected_bbox: Optional[list[int]] = None, note: str = ""):
    labels_path = get_labels_path(parquet_path)
    entry = {
        "index": index,
        "label": label,
        "corrected_bbox": corrected_bbox,
        "note": note,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(labels_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    init_page()

    st.markdown("""
<div class="hero-card">
  <h2 style="margin:0;">🔍 DocSeek Quality Studio</h2>
  <p class="hero-sub" style="margin:6px 0 0 0;">
    Visualize training samples, inspect bbox quality, and flag bad data.
  </p>
</div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.header("📂 Data Source")
        default_parquet = "data/docseek/v2/train.parquet"
        if "--parquet" in sys.argv:
            idx = sys.argv.index("--parquet")
            if idx + 1 < len(sys.argv):
                default_parquet = sys.argv[idx + 1]
        parquet_path = st.text_input("Parquet path", value=default_parquet)

    if not Path(parquet_path).exists():
        st.error(f"File not found: {parquet_path}")
        st.stop()

    df = load_parquet(parquet_path)
    labels = load_labels(parquet_path)
    total = len(df)

    # Build display records
    display_records = []
    for i in range(total):
        row = df.iloc[i].to_dict()
        fields = extract_fields(row)
        fields["index"] = i
        fields["label"] = labels.get(i, {}).get("label", "")
        display_records.append(fields)
    display_df = pd.DataFrame(display_records)

    # ── Sidebar: Filters ──
    with st.sidebar:
        st.header("🔎 Filters")

        sources = sorted(display_df["data_source"].unique().tolist())
        selected_sources = st.multiselect("Data Source", sources, default=sources)

        tasks = sorted(display_df["task_type"].unique().tolist())
        selected_tasks = st.multiselect("Task Type", tasks, default=tasks)

        etypes = sorted([e for e in display_df["element_type"].unique().tolist() if e])
        if etypes:
            selected_etypes = st.multiselect("Element Type", etypes, default=etypes)
        else:
            selected_etypes = []

        label_options = ["unlabeled", "bad"]
        selected_labels = st.multiselect("Label", label_options, default=label_options)

        keyword = st.text_input("Keyword search")

    # Apply filters
    filtered = display_df.copy()
    if selected_sources:
        filtered = filtered[filtered["data_source"].isin(selected_sources)]
    if selected_tasks:
        filtered = filtered[filtered["task_type"].isin(selected_tasks)]
    if selected_etypes:
        filtered = filtered[
            (filtered["element_type"].isin(selected_etypes)) |
            (filtered["element_type"] == "")
        ]
    if selected_labels:
        label_mask = filtered["label"].isin(selected_labels) | \
                     ((filtered["label"] == "") & ("unlabeled" in selected_labels))
        filtered = filtered[label_mask]
    if keyword.strip():
        kw = keyword.strip().lower()
        filtered = filtered[
            filtered["question"].str.lower().str.contains(kw, na=False) |
            filtered["ground_truth"].astype(str).str.lower().str.contains(kw, na=False)
        ]

    kept = len(filtered)

    # ── Sidebar: Stats ──
    with st.sidebar:
        st.markdown("---")
        st.header("📊 Stats")
        st.metric("Total", total)
        st.metric("Filtered", kept)
        bad_count = sum(1 for v in labels.values() if v.get("label") == "bad")
        st.metric("Marked Bad", bad_count)

    # ── Metrics row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total", total)
    m2.metric("Showing", kept)
    m3.metric("VQA", len(filtered[filtered["task_type"] == "vqa"]))
    m4.metric("GND", len(filtered[filtered["task_type"] == "gnd"]))
    m5.metric("OCR", len(filtered[filtered["task_type"] == "ocr"]))

    if kept == 0:
        st.warning("No samples match filters.")
        st.stop()

    # ── Navigation ──
    filtered_indices = filtered["index"].tolist()

    if "current_pos" not in st.session_state:
        st.session_state.current_pos = 0
    # Clamp
    st.session_state.current_pos = min(st.session_state.current_pos, kept - 1)

    nav1, nav2, nav3, nav4 = st.columns([1, 1, 2, 1])
    with nav1:
        if st.button("⬅️ Prev", use_container_width=True):
            st.session_state.current_pos = max(0, st.session_state.current_pos - 1)
            st.rerun()
    with nav2:
        if st.button("➡️ Next", use_container_width=True):
            st.session_state.current_pos = min(kept - 1, st.session_state.current_pos + 1)
            st.rerun()
    with nav3:
        pos = st.number_input(f"Position (0-{kept-1})", 0, kept - 1,
                              value=st.session_state.current_pos, step=1)
        st.session_state.current_pos = pos
    with nav4:
        if st.button("🎲 Random", use_container_width=True):
            import random
            st.session_state.current_pos = random.randint(0, kept - 1)
            st.rerun()

    current_idx = filtered_indices[st.session_state.current_pos]
    row = df.iloc[current_idx].to_dict()
    fields = extract_fields(row)

    # Extract bbox
    bbox = extract_bbox_for_sample(fields)
    # Check for corrected bbox
    corrected = labels.get(current_idx, {}).get("corrected_bbox")
    if corrected:
        bbox = corrected

    # ── Header bar ──
    current_label = labels.get(current_idx, {}).get("label", "")
    label_html = ""
    if current_label == "bad":
        label_html = ' <span class="label-bad">BAD</span>'

    st.markdown(
        f'**Sample {current_idx}** / {total} &nbsp;'
        f'<span class="source-tag">{fields["data_source"]}</span>'
        f'<span class="source-tag">{fields["task_type"]}</span>'
        f'<span class="source-tag">{fields.get("element_type", "")}</span>'
        f'{label_html}',
        unsafe_allow_html=True
    )

    # ── Detail layout ──
    c1, c2 = st.columns([1.25, 1.0])

    with c1:
        image_path = fields["image_path"]
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")

            # Bbox controls
            b1, b2 = st.columns(2)
            with b1:
                show_bbox = st.checkbox("Show bbox", value=True)
            with b2:
                edit_bbox = st.checkbox("✏️ Edit bbox", value=False)

            display_image = image
            if bbox and show_bbox:
                color = "#16a34a" if fields["task_type"] == "gnd" else "#0ea5e9"
                task_label = fields["task_type"].upper()
                display_image = draw_bbox_on_image(image, bbox, color, task_label)

            # Bbox editor
            if edit_bbox and bbox:
                st.caption(f"Bbox (0-1000): {bbox}")
                bc1, bc2, bc3, bc4 = st.columns(4)
                with bc1:
                    new_x1 = st.number_input("x1", 0, 1000, int(bbox[0]), key="bx1")
                with bc2:
                    new_y1 = st.number_input("y1", 0, 1000, int(bbox[1]), key="by1")
                with bc3:
                    new_x2 = st.number_input("x2", 0, 1000, int(bbox[2]), key="bx2")
                with bc4:
                    new_y2 = st.number_input("y2", 0, 1000, int(bbox[3]), key="by2")

                new_bbox = [new_x1, new_y1, new_x2, new_y2]
                if new_bbox != list(bbox):
                    display_image = draw_bbox_on_image(image, new_bbox, "#f59e0b", "EDITED")
                    if st.button("💾 Save corrected bbox"):
                        save_label(parquet_path, current_idx, "corrected",
                                   corrected_bbox=new_bbox)
                        st.success("Saved!")
                        st.rerun()

            st.image(display_image, use_container_width=True)
            st.caption(f"Image: {Path(image_path).name} | "
                       f"Size: {image.size[0]}×{image.size[1]}")
        else:
            st.error(f"Image not found: {image_path}")

    with c2:
        # ── Bad button (prominent) ──
        if st.button("❌ Mark as BAD & Next", use_container_width=True,
                      type="primary" if current_label != "bad" else "secondary"):
            save_label(parquet_path, current_idx, "bad")
            st.session_state.current_pos = min(kept - 1, st.session_state.current_pos + 1)
            st.rerun()

        if current_label == "bad":
            if st.button("↩️ Undo bad label", use_container_width=True):
                save_label(parquet_path, current_idx, "")  # clear label
                st.rerun()

        note = st.text_input("Note")
        if note and st.button("Save note"):
            save_label(parquet_path, current_idx, current_label or "", note=note)
            st.success("Saved")

        # ── Question ──
        st.markdown("---")
        st.markdown("**Question**")
        q_display = fields["question"].replace("<image>\n", "").strip()
        # Separate guidelines
        parts = q_display.split("Guidelines:")
        if len(parts) > 1:
            st.code(parts[0].strip(), language=None, wrap_lines=True)
            with st.expander("Guidelines", expanded=False):
                st.caption(parts[1].strip())
        else:
            st.code(q_display[:500], language=None, wrap_lines=True)

        # ── Ground Truth ──
        st.markdown("**Ground Truth**")
        gt = fields["ground_truth"]
        if isinstance(gt, list):
            st.code(json.dumps(gt, ensure_ascii=False), language="json", wrap_lines=True)
        else:
            st.code(str(gt), language=None, wrap_lines=True)

        # ── Info ──
        st.markdown("**Info**")
        st.json({
            "data_source": fields["data_source"],
            "task_type": fields["task_type"],
            "element_type": fields.get("element_type", ""),
            "doc_id": fields.get("doc_id", ""),
            "bbox": bbox,
        })

    # ── Export ──
    st.markdown("---")
    ex1, ex2 = st.columns(2)
    with ex1:
        bad_indices = [idx for idx, lb in labels.items() if lb.get("label") == "bad"]
        st.metric("Total marked bad", len(bad_indices))
        if bad_indices and st.button("Export bad indices (JSON)"):
            st.code(json.dumps(sorted(bad_indices)), language="json")
    with ex2:
        fix_items = [(idx, lb) for idx, lb in labels.items() if lb.get("corrected_bbox")]
        st.metric("Bbox corrections", len(fix_items))
        if fix_items and st.button("Export corrections"):
            st.code(json.dumps([{"index": idx, "bbox": lb["corrected_bbox"]}
                                for idx, lb in fix_items], indent=2), language="json")


if __name__ == "__main__":
    main()
