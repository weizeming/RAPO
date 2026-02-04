import argparse
import json
import os
from typing import Optional

from load_dataset import load_data
from utils import ADAPTIVE_THINKING_SYSTEM_PROMPT, adaptive_thinking_length


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


def _apply_chat_template(tokenizer, messages, add_generation_prompt: bool, enable_thinking: bool):
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def create_sft_data(
    model_path: str,
    dataset_recipe,
    save_path: str,
    data_dir: Optional[str],
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
):
    try:
        import gc
        import pandas as pd
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependencies. Install `torch pandas datasets transformers vllm` before running SFT."
        ) from e

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "recipe.json"), "w") as f:
        json.dump(dataset_recipe, f, indent=2, ensure_ascii=False)

    is_ds_model = "deepseek" in model_path.lower()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0, skip_special_tokens=False)

    model = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=True,
    )

    sft_dataset = []
    sft_data_prompt = []
    for dataset_name, n in dataset_recipe:
        data = load_data(dataset_name, n, data_dir=data_dir)
        df = pd.DataFrame(data)
        for _, item in df.iterrows():
            level = int(item.get("level", 1))
            level = level if level in adaptive_thinking_length else 1
            messages = [
                {
                    "role": "system",
                    "content": ADAPTIVE_THINKING_SYSTEM_PROMPT.format(
                        level=level, length=adaptive_thinking_length[level]
                    ),
                },
                {"role": "user", "content": f'The user prompt:\n"""{item["prompt"]}"""'},
            ]
            inputs = _apply_chat_template(tokenizer, messages, add_generation_prompt=True, enable_thinking=True)
            sft_dataset.append(
                {
                    "prompt": item["prompt"],
                    "message": messages,
                    "inputs": inputs,
                    "response": item.get("response", ""),
                    "level": level,
                }
            )
            sft_data_prompt.append(inputs)

    initial_outputs = model.generate(sft_data_prompt, sampling_params)
    final_inputs = []
    for idx, output in enumerate(initial_outputs):
        response = output.outputs[0].text or ""
        try:
            safe_reasoning = response.split("</think>", 1)[1].strip("\n").strip()
        except Exception:
            safe_reasoning = ""

        if len(output.outputs[0].token_ids) >= max_tokens:
            safe_reasoning = ""

        if len(safe_reasoning) == 0 or len(safe_reasoning) < 64:
            final_input = "</s>"
        else:
            final_messages = [{"role": "user", "content": sft_dataset[idx]["prompt"]}]
            final_input = _apply_chat_template(tokenizer, final_messages, add_generation_prompt=True, enable_thinking=True)
            if not is_ds_model:
                final_input += "<think>\nOkay, " + safe_reasoning + "\n\n"
            else:
                final_input += "Okay, " + safe_reasoning + "\n\n"

        sft_dataset[idx]["final_input"] = final_input
        sft_dataset[idx]["safe_reasoning"] = safe_reasoning
        sft_dataset[idx]["len_safe_reasoning"] = len(safe_reasoning)
        final_inputs.append(final_input)

    final_outputs = model.generate(final_inputs, sampling_params)
    sft_training_set = []
    for idx, output in enumerate(final_outputs):
        if len(sft_dataset[idx]["final_input"]) < 10:
            continue

        response_ids = output.outputs[0].token_ids
        if len(response_ids) >= max_tokens:
            full_completion = ""
            full_completion_response = ""
        else:
            full_completion = (output.prompt or "") + (output.outputs[0].text or "")

            if is_ds_model:
                marker_start = "<｜Assistant｜>"
                marker_end = ""
            else:
                marker_start = "<|im_start|>assistant"
                marker_end = "<|im_end|>"

            if marker_start in full_completion and marker_end in full_completion:
                start_idx = full_completion.index(marker_start) + len(marker_start)
                full_completion_response = full_completion[start_idx:].strip("\n").replace(marker_end, "")
            else:
                full_completion = ""
                full_completion_response = ""

        sft_dataset[idx]["full_completion"] = full_completion
        sft_dataset[idx]["full_completion_response"] = full_completion_response

        sft_training_item = {
            "prompt": [{"content": sft_dataset[idx]["prompt"], "role": "user"}],
            "completion": [{"content": full_completion_response, "role": "assistant"}],
            "level": sft_dataset[idx]["level"],
            "len_safe_thinking": sft_dataset[idx]["len_safe_reasoning"],
            "raw_prompt": sft_dataset[idx]["prompt"],
            "safe_reasoning": sft_dataset[idx]["safe_reasoning"],
        }
        if len(full_completion_response) > 0:
            sft_training_set.append(sft_training_item)

    with open(os.path.join(save_path, "json_data.json"), "w") as f:
        json.dump(sft_training_set, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(sft_training_set)
    df.to_csv(os.path.join(save_path, "csv_data.csv"), index=False)
    dataset = Dataset.from_pandas(df)

    if is_ds_model:
        def _concat_prompt_completion(example):
            text = _apply_chat_template(
                tokenizer,
                [{"role": "user", "content": example["prompt"][0]["content"]}],
                add_generation_prompt=True,
                enable_thinking=True,
            )
            completion = example["completion"][0]["content"]
            if "<think>\n" in completion:
                completion = completion.split("<think>\n", 1)[1]
            text += completion
            return {"text": text}

        dataset = dataset.map(_concat_prompt_completion, remove_columns=["prompt", "completion"])

    dataset.save_to_disk(save_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def sft_train(base_model_path: str, dataset_path: str, config):
    try:
        from datasets import load_from_disk
        from trl import SFTTrainer
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependencies. Install `datasets trl` before running SFT training."
        ) from e

    ds = load_from_disk(dataset_path)
    trainer = SFTTrainer(model=base_model_path, train_dataset=ds, args=config)
    trainer.train()


def main():
    try:
        from trl import SFTConfig
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `trl`. Install it before running SFT."
        ) from e

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--data-save-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="starbenign:400,stratasword:400")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    args = parser.parse_args()

    dataset_recipe = _parse_dataset_recipe(args.datasets)
    create_sft_data(
        model_path=args.model_path,
        dataset_recipe=dataset_recipe,
        save_path=args.data_save_path,
        data_dir=args.data_dir,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    config = SFTConfig(
        output_dir=args.save_path,
        num_train_epochs=float(args.epochs),
        save_steps=args.save_steps,
        save_safetensors=True,
        save_only_model=True,
        fp16=True,
        per_device_train_batch_size=2,
        packing=False,
    )
    sft_train(args.model_path, args.data_save_path, config)


if __name__ == "__main__":
    main()
