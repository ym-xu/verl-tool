"""
DocSeek Training Data Quality Studio.
Interactive viewer for inspecting, labeling, and correcting training data.

Run:
    streamlit run review_data.py -- --parquet data/docseek/v2/train.parquet

Features:
    - Browse samples by data_source, task_type
    - View images with bbox overlay (GND/OCR)
    - Mark samples as good/bad/needs_fix
    - Edit bbox coordinates with live preview
    - Export labeled/corrected data
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


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
.stApp {
    background: linear-gradient(180deg, #f6f8fb 0%, #eef2f7 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #132a45 100%);
}
[data-testid="stSidebar"] * {
    color: #f2f6ff !important;
}
.hero-card {
    background: linear-gradient(145deg, rgba(9,31,57,0.93), rgba(20,74,122,0.92));
    border-radius: 18px;
    padding: 16px 18px;
    color: #f6fbff;
    box-shadow: 0 12px 32px rgba(10, 25, 45, 0.18);
    margin-bottom: 1rem;
}
.label-good { background: #16a34a; color: white; padding: 2px 8px; border-radius: 4px; }
.label-bad { background: #dc2626; color: white; padding: 2px 8px; border-radius: 4px; }
.label-fix { background: #f59e0b; color: white; padding: 2px 8px; border-radius: 4px; }
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

    # Extract question from prompt
    question = ""
    if isinstance(prompt, list) and len(prompt) > 1:
        content = prompt[1].get("content", "")
        question = content

    # Extract image path
    images = row.get("images", [])
    image_path = ""
    if isinstance(images, list) and images:
        if isinstance(images[0], dict):
            image_path = images[0].get("image", "")
        elif isinstance(images[0], str):
            image_path = images[0]

    # Extract ground truth
    gt = reward.get("ground_truth", "") if isinstance(reward, dict) else ""

    # Extract task type and element type
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


# ─── Drawing ─────────────────────────────────────────────────────────────────

def draw_bbox_on_image(image: Image.Image, bbox: list[int], color: str = "#16a34a",
                       label: str = "", width: int = 3) -> Image.Image:
    """Draw bbox on image. bbox is in 0-1000 coords, convert to pixel coords."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Convert 0-1000 to pixel
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)

    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    if label:
        text_y = max(0, y1 - 20)
        draw.rectangle([x1, text_y, x1 + len(label) * 8 + 6, text_y + 18], fill=color)
        draw.text((x1 + 3, text_y + 1), label, fill="white")

    return img


# ─── Labels Management ───────────────────────────────────────────────────────

def get_labels_path(parquet_path: str) -> Path:
    """Labels stored alongside parquet."""
    return Path(parquet_path).parent / "quality_labels.jsonl"


def load_labels(parquet_path: str) -> dict[int, dict]:
    """Load quality labels from file."""
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
    """Append a quality label to file."""
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

    # Header
    st.markdown("""
<div class="hero-card">
  <h2 style="margin:0;">🔍 DocSeek Quality Studio</h2>
  <p style="margin:6px 0 0 0; opacity:0.88;">
    Browse, label, and correct training data. Labels saved to quality_labels.jsonl.
  </p>
</div>
    """, unsafe_allow_html=True)

    # ── Sidebar: Data source & filters ──
    with st.sidebar:
        st.header("📂 Data Source")
        import sys
        # Get parquet path from command line or default
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

    # Build display dataframe
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

        # Data source filter
        sources = sorted(display_df["data_source"].unique().tolist())
        selected_sources = st.multiselect("Data Source", sources, default=sources)

        # Task type filter
        tasks = sorted(display_df["task_type"].unique().tolist())
        selected_tasks = st.multiselect("Task Type", tasks, default=tasks)

        # Element type filter (for GND/OCR)
        etypes = sorted([e for e in display_df["element_type"].unique().tolist() if e])
        if etypes:
            selected_etypes = st.multiselect("Element Type", etypes, default=etypes)
        else:
            selected_etypes = []

        # Label filter
        label_options = ["unlabeled", "good", "bad", "needs_fix"]
        selected_labels = st.multiselect("Label", label_options, default=label_options)

        # Keyword search
        keyword = st.text_input("Question keyword")

        st.markdown("---")
        st.header("📊 Stats")

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

    # Stats in sidebar
    with st.sidebar:
        st.metric("Total", total)
        st.metric("Filtered", kept)
        labeled_count = sum(1 for v in labels.values() if v.get("label"))
        st.metric("Labeled", labeled_count)
        for src in sources:
            cnt = len(filtered[filtered["data_source"] == src])
            if cnt > 0:
                st.caption(f"{src}: {cnt}")

    # ── Metrics row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total", total)
    m2.metric("Filtered", kept)
    vqa_cnt = len(filtered[filtered["task_type"] == "vqa"])
    gnd_cnt = len(filtered[filtered["task_type"] == "gnd"])
    ocr_cnt = len(filtered[filtered["task_type"] == "ocr"])
    m3.metric("VQA", vqa_cnt)
    m4.metric("GND", gnd_cnt)
    m5.metric("OCR", ocr_cnt)

    if kept == 0:
        st.warning("No samples match filters.")
        st.stop()

    # ── Navigation ──
    filtered_indices = filtered["index"].tolist()

    if "current_pos" not in st.session_state:
        st.session_state.current_pos = 0

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 1])
    with nav_col1:
        if st.button("⬅️ Prev", use_container_width=True):
            st.session_state.current_pos = max(0, st.session_state.current_pos - 1)
    with nav_col2:
        if st.button("➡️ Next", use_container_width=True):
            st.session_state.current_pos = min(kept - 1, st.session_state.current_pos + 1)
    with nav_col3:
        pos = st.number_input("Position", 0, kept - 1,
                              value=st.session_state.current_pos, step=1)
        st.session_state.current_pos = pos
    with nav_col4:
        if st.button("🎲 Random", use_container_width=True):
            import random
            st.session_state.current_pos = random.randint(0, kept - 1)

    current_idx = filtered_indices[st.session_state.current_pos]
    row = df.iloc[current_idx].to_dict()
    fields = extract_fields(row)

    # ── Sample Detail ──
    st.markdown(f"### Sample {current_idx} / {total}  "
                f"`{fields['data_source']}` | `{fields['task_type']}` | "
                f"`{fields.get('element_type', '')}` | `{fields.get('doc_id', '')}`")

    c1, c2 = st.columns([1.3, 1.0])

    with c1:
        image_path = fields["image_path"]
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")

            # Parse bbox for GND tasks
            gt = fields["ground_truth"]
            bbox = parse_bbox_from_gt(gt) if fields["task_type"] == "gnd" else None

            # Also parse bbox from OCR question (region [...])
            if fields["task_type"] == "ocr":
                q = fields["question"]
                match = re.search(r"region\s*\[([^\]]+)\]", q)
                if match:
                    nums = [n.strip() for n in match.group(1).split(",")]
                    if len(nums) >= 4:
                        try:
                            bbox = [int(float(n)) for n in nums[:4]]
                        except ValueError:
                            pass

            # Corrected bbox from labels
            corrected = labels.get(current_idx, {}).get("corrected_bbox")
            if corrected:
                bbox = corrected

            # Bbox editing
            show_bbox = st.checkbox("Show bbox", value=True)
            edit_bbox = st.checkbox("✏️ Edit bbox", value=False)

            display_image = image
            if bbox and show_bbox:
                color = "#16a34a" if fields["task_type"] == "gnd" else "#0ea5e9"
                label = f"{fields['task_type'].upper()}"
                display_image = draw_bbox_on_image(image, bbox, color, label)

            if edit_bbox and bbox:
                st.caption(f"Current bbox (0-1000): {bbox}")
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
                if new_bbox != bbox:
                    color = "#f59e0b"
                    display_image = draw_bbox_on_image(image, new_bbox, color, "EDITED")
                    if st.button("💾 Save corrected bbox"):
                        save_label(parquet_path, current_idx, "needs_fix",
                                   corrected_bbox=new_bbox)
                        st.success(f"Saved corrected bbox for sample {current_idx}")
                        st.rerun()

            st.image(display_image, use_container_width=True)
        else:
            st.error(f"Image not found: {image_path}")

    with c2:
        # Current label
        current_label = labels.get(current_idx, {}).get("label", "")
        if current_label:
            color_map = {"good": "label-good", "bad": "label-bad", "needs_fix": "label-fix"}
            css = color_map.get(current_label, "")
            st.markdown(f'Current label: <span class="{css}">{current_label}</span>',
                        unsafe_allow_html=True)

        # Label buttons
        st.markdown("**Quality Label**")
        lb1, lb2, lb3 = st.columns(3)
        with lb1:
            if st.button("✅ Good", use_container_width=True, type="primary"):
                save_label(parquet_path, current_idx, "good")
                st.session_state.current_pos = min(kept - 1, st.session_state.current_pos + 1)
                st.rerun()
        with lb2:
            if st.button("❌ Bad", use_container_width=True):
                save_label(parquet_path, current_idx, "bad")
                st.session_state.current_pos = min(kept - 1, st.session_state.current_pos + 1)
                st.rerun()
        with lb3:
            if st.button("⚠️ Fix", use_container_width=True):
                save_label(parquet_path, current_idx, "needs_fix")
                st.session_state.current_pos = min(kept - 1, st.session_state.current_pos + 1)
                st.rerun()

        note = st.text_input("Note (optional)")
        if note and st.button("Save note"):
            save_label(parquet_path, current_idx, current_label or "needs_fix", note=note)
            st.success("Note saved")

        # Metadata
        st.markdown("---")
        st.markdown("**Question**")
        q_text = fields["question"]
        # Clean up for display
        q_display = q_text.replace("<image>\n", "").strip()
        if len(q_display) > 500:
            q_display = q_display[:500] + "..."
        st.code(q_display, language=None, wrap_lines=True)

        st.markdown("**Ground Truth**")
        gt = fields["ground_truth"]
        if isinstance(gt, list):
            st.code(json.dumps(gt, ensure_ascii=False, indent=2), language="json")
        else:
            st.code(str(gt), language=None, wrap_lines=True)

        st.markdown("**Info**")
        st.json({
            "data_source": fields["data_source"],
            "task_type": fields["task_type"],
            "element_type": fields.get("element_type", ""),
            "doc_id": fields.get("doc_id", ""),
            "image_path": fields["image_path"],
            "bbox": bbox if bbox else None,
        })

    # ── Export ──
    st.markdown("---")
    st.subheader("Export")
    ex1, ex2 = st.columns(2)
    with ex1:
        bad_indices = [idx for idx, lb in labels.items() if lb.get("label") == "bad"]
        st.metric("Marked as bad", len(bad_indices))
        if bad_indices and st.button("Export bad indices"):
            st.code(json.dumps(bad_indices), language="json")
    with ex2:
        fix_items = [(idx, lb) for idx, lb in labels.items()
                     if lb.get("corrected_bbox")]
        st.metric("Bbox corrections", len(fix_items))
        if fix_items and st.button("Export corrections"):
            st.code(json.dumps([{"index": idx, "bbox": lb["corrected_bbox"]}
                                for idx, lb in fix_items], indent=2), language="json")


if __name__ == "__main__":
    main()
