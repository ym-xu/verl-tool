import os
import torch
import json
import numpy as np
import logging
from collections import defaultdict
from verl.workers.reward_manager import register
from verl import DataProto
from .reward_score.doc_metrics import (
    extract_boxed_answer,
    compute_anls,
    compute_iou,
    compute_ned_similarity,
    parse_bbox_from_text,
)
from .reward_score.eval_benchmarks.docvqa_anls import anls_score
from .reward_score.eval_benchmarks.textvqa_accuracy import textvqa_accuracy_score
from .reward_score.eval_benchmarks.ocrbench import ocrbench_score
from .reward_score.eval_benchmarks.ocrbench_v2 import ocrbench_v2_score
from .reward_score.eval_benchmarks.visa_eval import visa_answer_score
from .reward_score.eval_benchmarks.wilddoc_eval import (
    wilddoc_docvqa_score,
    wilddoc_chartqa_score,
    wilddoc_tablevqa_fintabnet_score,
    wilddoc_tablevqa_vtabfact_score,
    wilddoc_tablevqa_vwtq_score,
)

logger = logging.getLogger(__file__)


def docseek_vqa_score(response_str: str, ground_truth) -> float:
    """Compute VQA score using ANLS."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        return 0.0
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    return compute_anls(answer, ground_truth)


def docseek_gnd_score(response_str: str, ground_truth) -> float:
    """Compute grounding score using IoU."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        return 0.0
    pred_bbox = parse_bbox_from_text(answer)
    if pred_bbox is None:
        return 0.0
    if isinstance(ground_truth, str):
        gt_bbox = parse_bbox_from_text(ground_truth)
    elif isinstance(ground_truth, list) and len(ground_truth) == 4:
        gt_bbox = [float(x) for x in ground_truth]
    else:
        return 0.0
    if gt_bbox is None:
        return 0.0
    iou = compute_iou(pred_bbox, gt_bbox)
    return iou


def docseek_ocr_score(response_str: str, ground_truth) -> float:
    """Compute OCR score using normalized edit distance similarity."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        return 0.0
    if isinstance(ground_truth, list):
        return max(compute_ned_similarity(answer, gt) for gt in ground_truth)
    return compute_ned_similarity(answer, ground_truth)


# --- Benchmark-specific eval functions (official metrics) ---

def eval_docvqa(response_str: str, ground_truth) -> float:
    """Official DocVQA / InfographicsVQA ANLS evaluation."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        return 0.0
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    return anls_score(answer, ground_truth)


def eval_textvqa(response_str: str, ground_truth) -> float:
    """Official TextVQA accuracy evaluation (10-annotator soft accuracy)."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        answer = response_str.strip()  # fallback: use raw response
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    return textvqa_accuracy_score(answer, ground_truth)


def eval_ocrbench(response_str: str, ground_truth, is_hme100k: bool = False) -> float:
    """Official OCRBench v1 substring containment evaluation."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        answer = response_str.strip()
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]
    return float(ocrbench_score(answer, ground_truth, is_hme100k=is_hme100k))


def eval_ocrbench_v2(response_str: str, ground_truth, task_type: str = "document_vqa") -> float:
    """Official OCRBench v2 evaluation (task-type specific metrics)."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        answer = response_str.strip()
    return ocrbench_v2_score(answer, ground_truth, task_type=task_type)


def eval_visa(response_str: str, ground_truth) -> float:
    """Official VISA token subsequence matching evaluation."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        answer = response_str.strip()
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    return visa_answer_score(answer, ground_truth)


def eval_wilddoc_docvqa(response_str: str, ground_truth) -> float:
    """Official WildDoc-DocVQA ANLS evaluation."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        return 0.0
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    return wilddoc_docvqa_score(answer, ground_truth)


def eval_wilddoc_chartqa(response_str: str, ground_truth) -> float:
    """Official WildDoc-ChartQA relaxed accuracy evaluation."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        return 0.0
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]
    return wilddoc_chartqa_score(answer, ground_truth)


def eval_wilddoc_tablevqa(response_str: str, ground_truth, subset: str = '') -> float:
    """Official WildDoc-TableVQA evaluation (routes by subset)."""
    answer = extract_boxed_answer(response_str)
    if answer is None:
        return 0.0
    if 'fintabnet' in subset:
        gt = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
        return wilddoc_tablevqa_fintabnet_score(answer, gt)
    elif 'vtabfact' in subset:
        gt = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
        return wilddoc_tablevqa_vtabfact_score(answer, gt)
    elif 'vwtq' in subset:
        return wilddoc_tablevqa_vwtq_score(answer, ground_truth)
    # Default: ANLS
    return eval_docvqa(response_str, ground_truth)


@register("docseek")
class DocSeekRewardManager:
    """Reward manager for DocSeek: multi-task document understanding (VQA, GND, OCR)."""

    name = "docseek"

    # Training reward: route by task_type
    TASK_SCORE_FN = {
        "vqa": docseek_vqa_score,
        "gnd": docseek_gnd_score,
        "ocr": docseek_ocr_score,
    }

    # Eval: route by data_source to official benchmark metrics
    EVAL_SCORE_FN = {
        "docvqa": eval_docvqa,
        "infovqa": eval_docvqa,          # same ANLS as DocVQA
        "infographicsvqa": eval_docvqa,   # alias
        "textvqa": eval_textvqa,
        "ocrbench": eval_ocrbench,
        "ocrbench_v2": eval_ocrbench_v2,  # v2 has task-specific metrics
        "visa_paper": eval_visa,
        "visa_wiki": eval_visa,
        "paper_visa": eval_visa,          # alias
        "wiki_visa": eval_visa,           # alias
        "wilddoc_docvqa": eval_wilddoc_docvqa,
        "wilddoc_chartqa": eval_wilddoc_chartqa,
        "wilddoc_tablevqa": eval_wilddoc_tablevqa,
    }

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        # Tool-use penalties (from pixel_reasoner)
        self.add_curiousity_penalty = True
        self.add_action_redundancy_penalty = True
        self.group_tool_call_rate_lower_bound = 0.3  # H in the paper
        self.action_redundancy_limit = 1  # n_{vo}
        self.alpha = 0.0  # Disabled: curiosity penalty suppresses zoom learning
        self.beta = 0.05

    def get_group_info(self, data: DataProto):
        group_info = {}
        for i in range(len(data)):
            data_item = data[i]
            tool_interact_info = data_item.non_tensor_batch.get('tool_interact_info', None)
            num_turn = len(tool_interact_info) if tool_interact_info is not None else 0
            num_valid_action = sum(1 for t in tool_interact_info if t.get('valid_action', False)) if tool_interact_info is not None else 0
            if "tool_interact_info" in data_item.non_tensor_batch:
                uid = data_item.non_tensor_batch.get('uid', i)
                if uid not in group_info:
                    group_info[uid] = {'num_turns': [], 'num_valid_actions': []}
                group_info[uid]['num_turns'].append(num_turn)
                group_info[uid]['num_valid_actions'].append(num_valid_action)
        for uid, info in group_info.items():
            info['num_turns'] = np.array(info['num_turns'])
            info['num_valid_actions'] = np.array(info['num_valid_actions'])
            info['group_tool_call_rate'] = np.mean([1 if n > 0 else 0 for n in info['num_valid_actions']])
            info['tool_call_total'] = info['num_valid_actions'].sum()
        return group_info

    def add_additional_penalties(self, response: str, data_i, scores_i: dict, group_info: dict):
        if "tool_interact_info" in data_i.non_tensor_batch:
            tool_interact_info = data_i.non_tensor_batch.get('tool_interact_info', None)
            num_valid_action = sum(1 for t in tool_interact_info if t.get('valid_action', False)) if tool_interact_info is not None else 0
            if self.add_curiousity_penalty:
                penalty = (num_valid_action != 0) * max(0, self.group_tool_call_rate_lower_bound - group_info.get('group_tool_call_rate', 0))
                penalty *= self.alpha
                scores_i['score'] += penalty
                scores_i['curiousity_penalty'] = penalty
            if self.add_action_redundancy_penalty:
                penalty = min(self.action_redundancy_limit - num_valid_action, 0)
                penalty *= self.beta
                scores_i['score'] += penalty
                scores_i['action_redundancy_penalty'] = penalty
        return scores_i

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        group_info = self.get_group_info(data)

        for i in range(len(data)):
            score = {}
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get('extra_info', {})

            # Route scoring: use official eval metric if data_source matches a benchmark,
            # otherwise fall back to task_type-based training reward
            task_type = extra_info.get('task_type', 'vqa')
            data_source_lower = data_source.lower().replace('-', '_')
            eval_fn = self.EVAL_SCORE_FN.get(data_source_lower, None)
            if eval_fn is not None:
                # OCRBench v2 needs sub-task type from extra_info
                if data_source_lower == 'ocrbench_v2':
                    sub_task = extra_info.get('ocr_task_type', 'document_vqa')
                    task_score = eval_fn(response_str, ground_truth, task_type=sub_task)
                # WildDoc-TableVQA needs subset info
                elif data_source_lower == 'wilddoc_tablevqa':
                    subset = extra_info.get('subset', '')
                    task_score = eval_fn(response_str, ground_truth, subset=subset)
                else:
                    task_score = eval_fn(response_str, ground_truth)
            else:
                score_fn = self.TASK_SCORE_FN.get(task_type, docseek_vqa_score)
                task_score = score_fn(response_str, ground_truth)

            score['accuracy'] = task_score
            score['score'] = task_score
            score['task_type'] = task_type

            # Add tool-use penalties
            score = self.add_additional_penalties(
                response_str, data_item, score,
                group_info.get(data_item.non_tensor_batch.get('uid', i), {})
            )

            if score['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

            # Per-task tracking
            reward_extra_info[f'{task_type}_accuracy'].append(score['accuracy'])

            reward = score["score"]
            for key, value in score.items():
                reward_extra_info[key].append(value)
            if self.num_examine == 1:
                reward = score["accuracy"]  # for validation

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[response] {response_str[:500]}")
                print(f"[ground_truth] {ground_truth}")
                print(f"[task_type] {task_type}")
                for key, value in score.items():
                    print(f"[{key}] {value}")

            # Crop tool_interact_info images for debug storage
            tool_interact_info_i = data_item.non_tensor_batch.get('tool_interact_info', None)
            if tool_interact_info_i is not None:
                for tool_interact in tool_interact_info_i:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50]

        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info['correct_response_length'] else 0.0
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info['wrong_response_length'] else 0.0
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items())),
            }
        else:
            return reward_tensor
