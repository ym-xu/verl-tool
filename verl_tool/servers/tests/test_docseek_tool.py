#!/usr/bin/env python
"""Test DocSeek zoom tool via HTTP API or directly."""
import json
import requests
import fire
import logging
import io
import base64
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dummy_document(width=2000, height=3000):
    """Create a dummy document image with text-like content for testing."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    # Draw grid and text placeholders
    for y in range(0, height, 100):
        draw.line([(0, y), (width, y)], fill='lightgray')
    for x in range(0, width, 200):
        draw.line([(x, 0), (x, height)], fill='lightgray')
    # Draw some "text" regions
    draw.rectangle([100, 100, 900, 200], fill='black')
    draw.rectangle([100, 250, 600, 290], fill='gray')
    draw.rectangle([100, 320, 800, 360], fill='gray')
    draw.rectangle([100, 390, 700, 430], fill='gray')
    # Draw a "table"
    for row in range(5):
        for col in range(4):
            x1, y1 = 100 + col * 400, 500 + row * 80
            draw.rectangle([x1, y1, x1 + 380, y1 + 60], outline='black', width=2)
    return img


def encode_image_url(img):
    buffered = io.BytesIO()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"


def test_zoom_api(
    url: str = "http://localhost:5500/get_observation",
    trajectory_id: str = "test-docseek-001",
):
    """Test DocSeek zoom via HTTP API (requires running tool server)."""
    img = create_dummy_document()
    encoded = encode_image_url(img)

    # Test 1: Valid zoom action
    print("--- Test 1: Valid zoom_in ---")
    action = '<tool_call>{"name": "zoom_in", "arguments": {"bbox_2d": [0.05, 0.03, 0.5, 0.15], "target_image": 1}}</tool_call>'
    result = _send_request(url, trajectory_id, action, {"images": [encoded]})
    if result and "observations" in result:
        obs = result["observations"][0]
        if isinstance(obs, dict) and 'image' in obs:
            print(f"  OK: Got cropped image, obs text: {obs['obs']}")
        else:
            print(f"  Got observation: {str(obs)[:200]}")

    # Test 2: Invalid bbox
    print("\n--- Test 2: Invalid bbox ---")
    action2 = '<tool_call>{"name": "zoom_in", "arguments": {"bbox_2d": [0.1, 0.2], "target_image": 1}}</tool_call>'
    result2 = _send_request(url, f"{trajectory_id}-2", action2, {"images": [encoded]})
    if result2:
        print(f"  Observation: {result2.get('observations', [''])[0]}")
        print(f"  Valid: {result2.get('valids', [None])[0]}")

    # Test 3: Unknown tool name
    print("\n--- Test 3: Unknown tool name ---")
    action3 = '<tool_call>{"name": "select_frames", "arguments": {"target_frames": [1]}}</tool_call>'
    result3 = _send_request(url, f"{trajectory_id}-3", action3, {"images": [encoded]})
    if result3:
        print(f"  Valid: {result3.get('valids', [None])[0]} (should be False)")


def test_zoom_direct():
    """Test DocSeek zoom tool directly without HTTP server."""
    from verl_tool.servers.tools.docseek import DocSeekTool

    tool = DocSeekTool(num_workers=1)
    img = create_dummy_document()
    encoded = encode_image_url(img)

    # Test zoom_in
    action = '<tool_call>{"name": "zoom_in", "arguments": {"bbox_2d": [0.05, 0.03, 0.5, 0.15], "target_image": 1}}</tool_call>'
    obs, done, valid = tool.conduct_action("test-direct-001", action, {"images": [encoded]})
    print(f"Direct test - Valid: {valid}, Done: {done}")
    if isinstance(obs, dict):
        print(f"  Obs: {obs['obs']}")
        print(f"  Has image: {'image' in obs}")
    else:
        print(f"  Obs: {obs}")


def _send_request(url, trajectory_id, action, extra_fields):
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [extra_fields],
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Request error: {e}")
        return None


def main():
    fire.Fire({
        "api": test_zoom_api,
        "direct": test_zoom_direct,
    })


if __name__ == "__main__":
    main()
