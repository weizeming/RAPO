import argparse
import json
import os
import re
from typing import Optional

from load_dataset import load_data
from utils import (
    BENIGN_JUDGE_SYSTEM_PROMPT,
    REWARD_JUDGE_SYSTEM_PROMPT,
    SAFETY_JUDGE_SYSTEM_PROMPT,
    str_judge,
)

_reward_model = None
_reward_tokenizer = None
_reward_sampling_params = None
_log_path = None
_save_json = []


def _parse_dataset_recipe(s: str):
    if not s.strip():
        return []
    recipe = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        name, count = part.split(":")
        recipe.append((name.strip(), int(count.strip())))
    return recipe


def _extract_tag_number(text: str, tag: str):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    v = m.group(1).strip()
    try:
        return float(v)
    except ValueError:
        digits = re.findall(r"[-+]?\d*\.?\d+", v)
        return float(digits[0]) if digits else None


def _get_general_reward(text: str):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    score = _extract_tag_number(text, "score")
    if score is None:
        digits = re.findall(r"\d", text)
        score = float(digits[0]) if digits else 0.0
    return score * 2.0 - 1.0


def _get_risk_reward(text: str):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    reward = _extract_tag_number(text, "reward")
    if reward is None:
        return 0.0
    return reward if -2.0 < reward < 2.0 else 0.0


def _extract_safety_reasoning(completion_obj):
    raw = completion_obj[0]["content"]
    try:
        thinking, response = raw.split("</think>", 1)
        thinking = thinking.replace("<think>", "").strip()
        response = response.strip()
        safety_reasoning = thinking.split("\n\n", 1)[0].strip()
        return safety_reasoning, thinking, response
    except ValueError:
        head = raw.split("\n\n", 1)[0].strip()
        return head, "", ""


def reward_func(prompts, completions, harmful_label, **kwargs):
    global _save_json
    full_completions = [c[0]["content"].strip() for c in completions]
    extracted = [_extract_safety_reasoning(c) for c in completions]
    safety_reasonings = [x[0] for x in extracted]
    plain_completions = [x[2] for x in extracted]

    risk_inputs = []
    for i, sr in enumerate(safety_reasonings):
        ori_prompt = prompts[i][0]["content"]
        length_hint = (
            "## Hint: The length of the **Original Prompt is "
            f"{ori_prompt.count('. ') + 1} sentences**; The length of the **Safety Reasoning Trace is {sr.count('. ') + 1} sentences**. "
            "Please follow the **sentence length criteria** in the system prompt to judge the level, case and reward."
        )
        risk_user_prompt = f"## [Original Prompt] {ori_prompt}\n\n## [Safety Reasoning Trace] {sr}\n\n{length_hint}"
        risk_message = [
            {"role": "system", "content": REWARD_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": risk_user_prompt},
        ]
        risk_inputs.append(
            _reward_tokenizer.apply_chat_template(
                risk_message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    risk_responses = _reward_model.generate(risk_inputs, _reward_sampling_params)
    risk_texts = [r.outputs[0].text for r in risk_responses]
    risk_rewards = [_get_risk_reward(t) for t in risk_texts]

    general_inputs = []
    for i, completion in enumerate(plain_completions):
        ori_prompt = prompts[i][0]["content"]
        if int(harmful_label[i]) == 1:
            user_prompt = SAFETY_JUDGE_SYSTEM_PROMPT.format(prompt=ori_prompt, response=completion)
        else:
            user_prompt = BENIGN_JUDGE_SYSTEM_PROMPT.format(prompt=ori_prompt, response=completion)
        general_inputs.append(
            _reward_tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    general_responses = _reward_model.generate(general_inputs, _reward_sampling_params)
    general_texts = [r.outputs[0].text for r in general_responses]
    general_rewards = [_get_general_reward(t) for t in general_texts]

    rewards = [risk_rewards[i] + general_rewards[i] for i in range(len(prompts))]

    _save_json.append(
        [
            {
                "prompts": prompts[i],
                "full_completion": full_completions[i],
                "safety_reasoning": safety_reasonings[i],
                "completion": plain_completions[i],
                "harmful_label": int(harmful_label[i]),
                "risk_aware_reward_response": risk_texts[i],
                "general_reward_response": general_texts[i],
                "risk_aware_reward": risk_rewards[i],
                "general_reward": general_rewards[i],
                "reward": rewards[i],
                "str_judge": str_judge(plain_completions[i]),
            }
            for i in range(len(prompts))
        ]
    )
    with open(_log_path, "w") as f:
        json.dump(_save_json, f, indent=2, ensure_ascii=False)

    return rewards


def rl_train(
    model_path: str,
    reward_model_path: str,
    dataset_recipe,
    config,
    save_path: str,
    data_dir: Optional[str],
    log_dir: str,
    reward_tensor_parallel_size: int,
    reward_gpu_memory_utilization: float,
):
    global _reward_model, _reward_tokenizer, _reward_sampling_params, _log_path, _save_json

    try:
        import pandas as pd
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOTrainer
        from vllm import LLM
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependencies. Install `pandas datasets transformers trl vllm` before running RL."
        ) from e

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "recipe.json"), "w") as f:
        json.dump(dataset_recipe, f, indent=2, ensure_ascii=False)

    rl_rows = []
    for dataset_name, n in dataset_recipe:
        data = load_data(dataset_name, n, data_dir=data_dir)
        df = pd.DataFrame(data)
        for _, item in df.iterrows():
            rl_rows.append(
                {
                    "prompt": [{"content": item["prompt"], "role": "user"}],
                    "harmful_label": int(item["harmful_label"]),
                }
            )
    rl_dataset = Dataset.from_pandas(pd.DataFrame(rl_rows).sample(frac=1, random_state=0))

    _reward_model = LLM(
        model=reward_model_path,
        tensor_parallel_size=reward_tensor_parallel_size,
        gpu_memory_utilization=reward_gpu_memory_utilization,
        dtype="auto",
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    _reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path, trust_remote_code=True)

    os.makedirs(log_dir, exist_ok=True)
    _log_path = os.path.join(log_dir, f"{os.path.basename(save_path.rstrip('/'))}.json")
    _save_json = []

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=reward_func,
        train_dataset=rl_dataset,
        args=config,
    )
    trainer.train()


def main():
    try:
        from trl import GRPOConfig
        from vllm import SamplingParams
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependencies. Install `trl vllm` before running RL."
        ) from e

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="wildjailbreak:300,star:100,starbenign:400")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-steps", type=int, default=400)
    parser.add_argument("--log-dir", type=str, default="rl_log")
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--reward-max-tokens", type=int, default=1024)
    parser.add_argument("--reward-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--reward-gpu-memory-utilization", type=float, default=0.2)
    args = parser.parse_args()

    global _reward_sampling_params
    _reward_sampling_params = SamplingParams(
        max_tokens=args.reward_max_tokens,
        temperature=0.0,
        top_p=1.0,
        skip_special_tokens=False,
    )

    dataset_recipe = _parse_dataset_recipe(args.datasets)
    config = GRPOConfig(
        output_dir=args.save_path,
        num_train_epochs=args.epochs,
        save_safetensors=True,
        save_only_model=True,
        bf16=True,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        save_steps=args.save_steps,
    )

    rl_train(
        model_path=args.model_path,
        reward_model_path=args.base_model_path,
        dataset_recipe=dataset_recipe,
        config=config,
        save_path=args.save_path,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        reward_tensor_parallel_size=args.reward_tensor_parallel_size,
        reward_gpu_memory_utilization=args.reward_gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
