from .base import BaseTool, register_tool
import regex as re
import json
import asyncio
import concurrent.futures
from typing import Tuple, Union, List, Dict, Any
import os
import logging

import base64
import io
from PIL import Image
from pathlib import Path
from verl_tool.agent_loop.vision_utils import process_image

logger = logging.getLogger(__file__)


def crop(str_image, bbox_2d, padding=(0.1, 0.1)):
    """Crop the image based on the bounding box coordinates."""
    if isinstance(str_image, list):
        str_image = str_image[0]
    if isinstance(str_image, Path) and str_image.exists() or \
        isinstance(str_image, str) and os.path.exists(str_image):
        image = Image.open(str_image)
    elif isinstance(str_image, Image.Image):
        image = str_image
    else:
        image = decode_image_url(str_image)
    img_x, img_y = image.size
    padding_tr = (600.0 / img_x, 600.0 / img_y)
    padding = (min(padding[0], padding_tr[0]), min(padding[1], padding_tr[1]))

    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0]) - padding[0], float(bbox_2d[1]) - padding[1],
                              float(bbox_2d[2]) + padding[0], float(bbox_2d[3]) + padding[1])
    else:
        normalized_bbox_2d = (float(bbox_2d[0]) / img_x - padding[0], float(bbox_2d[1]) / img_y - padding[1],
                              float(bbox_2d[2]) / img_x + padding[0], float(bbox_2d[3]) / img_y + padding[1])
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 = min(max(0, normalized_x1), 1)
    normalized_y1 = min(max(0, normalized_y1), 1)
    normalized_x2 = min(max(0, normalized_x2), 1)
    normalized_y2 = min(max(0, normalized_y2), 1)
    cropped_img = image.crop((int(normalized_x1 * img_x), int(normalized_y1 * img_y),
                              int(normalized_x2 * img_x), int(normalized_y2 * img_y)))
    return cropped_img


def encode_image(img: Image.Image) -> str:
    buffered = io.BytesIO()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def decode_image(img_str):
    img_data = base64.b64decode(img_str)
    return Image.open(io.BytesIO(img_data))


def encode_image_url(img: Image.Image) -> str:
    return f"data:image/jpeg;base64,{encode_image(img)}"


def decode_image_url(img_str):
    if img_str.startswith("data:image/jpeg;base64,"):
        img_str = img_str.split("data:image/jpeg;base64,")[1]
    return decode_image(img_str)


@register_tool
class DocSeekTool(BaseTool):
    tool_type = "docseek"

    stop_tokens = ["</tool_call>"]
    valid_mcp_func_names = ['zoom_in', 'crop_image']

    def __init__(self, num_workers=1):
        super().__init__(num_workers)
        self.image_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4),
            thread_name_prefix="docseek_image_processor"
        )

    def get_usage_inst(self):
        return ""

    def parse_action(self, action: str) -> Tuple[str, bool]:
        """Parse the raw action string into an action dict and validity flag."""
        try:
            call = json.loads(action.split('<tool_call>')[1].split('</tool_call>')[0])
            name = call.get('name', '')
            if name not in self.valid_mcp_func_names:
                return "", False
        except:
            return "", False
        return call, True

    def load_env(self, trajectory_id):
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {"turns": 0},
                "previous_obs": [],
                "images": None,
                "temporary_images": [],
                "temporary_image_folder": Path(f"tmp/crop_images/{trajectory_id}"),
            }
            env['temporary_image_folder'].mkdir(parents=True, exist_ok=True)
        return env

    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        if isinstance(observation, dict) and 'image' in observation:
            if isinstance(observation['image'], str):
                env['images'].append(self.save_image_to_env(trajectory_id, observation['image']))
            elif isinstance(observation['image'], list):
                env['images'].extend([self.save_image_to_env(trajectory_id, img) for img in observation['image']])
        env["metadata"]["turns"] += 1
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })

    def delete_env(self, trajectory_id):
        self.env_cache.pop(trajectory_id, None)

    def save_image_to_env(self, trajectory_id, image: Union[Image.Image, str]) -> str:
        env = self.load_env(trajectory_id)
        env['temporary_images'].append(image)
        return image

    async def _process_single_image(self, img_source, bbox_2d):
        """Process a single image crop operation asynchronously."""
        def _crop_and_process():
            cropped_img = crop(img_source, bbox_2d)
            return process_image({"image": cropped_img})

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.image_executor, _crop_and_process)

    async def conduct_zoom_in_action_async(self, parameters, env):
        """Execute the zoom-in action asynchronously."""
        valid = False
        missing_parameters = []
        if 'bbox_2d' not in parameters:
            missing_parameters.append('bbox_2d')
        if 'target_image' not in parameters:
            missing_parameters.append('target_image')
        try:
            parameters['target_image'] = int(parameters['target_image'])
        except:
            pass
        if missing_parameters:
            observation = f"Missing parameters: {', '.join(missing_parameters)}"
        elif not isinstance(parameters['bbox_2d'], list) or len(parameters['bbox_2d']) != 4:
            observation = "Invalid bbox_2d format. It should be a list of four numbers."
        elif not isinstance(parameters['target_image'], int) or parameters['target_image'] <= 0 or parameters['target_image'] > len(env['images']):
            observation = f"Invalid target_image index. It should be an integer between 1 and the number of previous images ({len(env['images'])})."
        else:
            try:
                previous_images = env['images']
                img_to_crop = previous_images[parameters['target_image'] - 1]

                # Get original image size before cropping
                if isinstance(img_to_crop, (str, Path)) and not str(img_to_crop).startswith("data:"):
                    orig_img = Image.open(img_to_crop)
                elif isinstance(img_to_crop, Image.Image):
                    orig_img = img_to_crop
                else:
                    orig_img = decode_image_url(img_to_crop)
                orig_w, orig_h = orig_img.size

                processed_img = await self._process_single_image(img_to_crop, parameters['bbox_2d'])
                encoded_cropped_img = encode_image_url(processed_img)
                crop_w, crop_h = processed_img.size
                observation = {
                    'obs': f"Here is the cropped image. (Original: {orig_w}x{orig_h}, Cropped: {crop_w}x{crop_h})\n<image>",
                    'image': encoded_cropped_img,
                }
                valid = True
            except Exception as e:
                observation = f"Error processing image: {str(e)}"
                logger.error(f"Error processing zoom-in action: {str(e)}; parameters: {parameters}")
        return observation, valid

    async def aget_observations(self, trajectory_ids: List[str], actions: List[str], extra_fields: List[Dict[str, Any]]):
        """Async version of get_observations for concurrent processing."""
        tasks = [
            self._conduct_action_async(trajectory_id, action, extra_field)
            for trajectory_id, action, extra_field in zip(trajectory_ids, actions, extra_fields)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        observations, dones, valids = [], [], []
        for result in results:
            if isinstance(result, Exception):
                observations.append(f"Processing error: {str(result)}")
                dones.append(False)
                valids.append(False)
            else:
                obs, done, valid = result
                observations.append(obs)
                dones.append(done)
                valids.append(valid)
        return observations, dones, valids

    async def _conduct_action_async(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]):
        """Execute the parsed action asynchronously."""
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        if env['images'] is None:
            env['images'] = [Path(x) if not x.startswith("data:image") else decode_image_url(x) for x in extra_field.get('images', [])]

        if not is_valid:
            observation = ""
            done = False
            valid = False
        else:
            done = False
            valid = True
            if 'arguments' not in parsed_action:
                observation = "Missing 'arguments' in the tool call."
                valid = False
            elif not isinstance(parsed_action['arguments'], dict):
                observation = f"'arguments' should be a dictionary of parameters key-value pairs, got {type(parsed_action['arguments'])}."
                valid = False
            elif parsed_action['name'] in ['zoom_in', 'crop_image']:
                try:
                    observation, valid = await self.conduct_zoom_in_action_async(parsed_action['arguments'], env)
                except Exception as e:
                    observation = f"Error processing {parsed_action['name']} action: {str(e)}"
                    valid = False
                    logger.error(f"Error processing {parsed_action['name']} action: {str(e)}; parameters: {parsed_action['arguments']}")
            else:
                observation = "Unknown action name."
                valid = False

        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        return observation, done, valid

    def conduct_action(self, trajectory_id, action, extra_field):
        """Synchronous wrapper for backward compatibility."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._conduct_action_async(trajectory_id, action, extra_field))
        finally:
            loop.close()

    def __del__(self):
        if hasattr(self, 'image_executor'):
            self.image_executor.shutdown(wait=False)
