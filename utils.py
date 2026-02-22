import wandb
import os
from datetime import datetime
import hashlib
from datasets import load_dataset
import random
import numpy as np
import torch
import argparse
# code from https://github.com/yaochenzhu/Rank-GRPO (MIT Licence)
from transformers import AutoTokenizer
import re
from datasets import load_from_disk

def set_seed(seed: int):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description="Train a model with RL.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode to run the script in.",
    )
    parser.add_argument(
        "--eval_untrained_model",
        action="store_true",
        help="Evaluate the untrained model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="/shared/public/models/Qwen/Qwen2.5-0.5B-Instruct",
        default="enter/your/path",
        # default="/shared/public/models/DeepSeek-R1-Distill-Qwen-7B",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        # default="HuggingFaceH4/MATH-500",
        default="DAPO-Math-17k",
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="G2RPO",
        help="RL method to use (e.g., PPO, SFT, G2RPO).",
    )
    parser.add_argument(
        "--sft_max_steps",
        type=int,
        default=1000,
        help="Maximum number of training steps for supervised fine-tuning.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.99,
        help="Adam beta2.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--sft_eval_steps",
        type=int,
        default=50,
        help="Number of steps between evaluations during supervised fine-tuning.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Number of steps between evaluations.",
    )
    parser.add_argument(
        "--surprise_weight",
        type=float,
        default=1e-6,
        help="Weight for the surprise bonus.",
    )
    parser.add_argument(
        "--surprise_decay",
        type=float,
        default=0.5,
        help="Decay rate for the surprise bonus.",
    )
    parser.add_argument(
        "--surprise_horizon",
        type=int,
        default=1,
        help="Horizon for the surprise bonus.",
    )
    parser.add_argument(
        "--confidence_weight",
        type=float,
        default=1e-6,
        help="Weight for the confidence bonus.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugging mode.",
    )
    parser.add_argument(
        "--max_completion_length",
        "-mcl",
        type=int,
        default=8000,
        help="Maximum length of the generated completions."
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training.",
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=1,
        help="Number of devices for distributed training.",
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Enable supervised fine-tuning.",
    )
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        default='no_sft',
        help="Path to the checkpoint of the supervised fine-tuned model.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=64,
        help="Steps per generation.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Per device train batch size.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Per device eval batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of generations per prompt for SGRPOLee.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging steps.",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0,
        help="KL beta.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable wandb logging.",
    )
    parser.add_argument(
        "--wandb_project", 
        type=str,
        default="week_10_26_lkn",
        help="Random seed for reproducibility.",
    )
    # argument for baseline SGRPOLee
    parser.add_argument(
        "--alpha",
        type=float,
        default=150,
        help="Alpha parameter for the stochastic length mask.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=341,
        help="K parameter for the stochastic length mask.",
    )
    parser.add_argument(
        "--token_prob",
        type=float,
        default=0.5,
        help="Probability of the token being masked.",
    )
    parser.add_argument(
        "--predict_mode",
        type=str,
        default='next_token',
        choices=['next_token', 'last_token'],
        help="Predict mode for the entropy calculation.",
    )
    # argument for G2RPOCritics
    parser.add_argument(
        "--critics_path",
        type=str,
        default="/shared/user/agent_rl/critics",
        help="Path to the critics model.",
    )
    parser.add_argument(
        "--critics_llm_freezed",
        action="store_true",
        help="Freeze the LLM backbone.",
    )
    # argument for baseline GFPO
    parser.add_argument(
        "--top_num_gen",
        type=int,
        default=6,
        help="Number of shortest completions to select from num_generations. If None, all generations are used.",
    )
    # argument for baseline DAPO
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100,
        help="Buffer size for the buffer.",
    )
    # arguments for vLLM generation acceleration
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for faster generation.",
    )
    parser.add_argument(
        "--vllm_mode",
        type=str,
        default="colocate",
        choices=["server", "colocate"],
        help="vLLM mode: 'server' (separate vLLM server) or 'colocate' (same process).",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.4,
        help="GPU memory utilization for vLLM (only for colocate mode).",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM (only for colocate mode).",
    )
    # only when evaluation
    parser.add_argument(
        "--eval_checkpoint_dir",
        type=str,
        required=False,
        help="Checkpoint directory for the model to evaluate.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def get_dataset(data_name, model_name, mode=None):
    data_handle_dict = {
        "AIME2025": "opencompass/AIME2025",
        "MATH-500": "enter/your/path",
        "DAPO-Math-17k": "enter/your/path",
        "GSM8K":"enter/your/path",
        "CommonsenseQA": "enter/your/path",
    }
    if data_name in ["AIME2025",  "DAPO-Math-17k"]:
        full_dataset = preprocess_dataset(model_name, data_handle_dict, data_name, mode=mode)
    elif data_name == "MATH-500":
        full_dataset = preprocess_dataset(model_name, data_handle_dict, data_name, mode=mode)
    elif data_name == "GSM8K":
        train_dataset = preprocess_dataset(model_name, data_handle_dict, data_name, split='train', mode=mode)
        eval_dataset = preprocess_dataset(model_name, data_handle_dict, data_name, split='test', mode=mode)
        return train_dataset, eval_dataset
    elif data_name == "CommonsenseQA":
        train_dataset = preprocess_dataset(model_name, data_handle_dict, data_name, split='train', mode=mode)
        eval_dataset = preprocess_dataset(model_name, data_handle_dict, data_name, split='validation', mode=mode)
        return train_dataset, eval_dataset
    else:
        raise NotImplementedError
    # Get the actual dataset size dynamically
    dataset_size = len(full_dataset)  # type: ignore
    print(f"Total dataset size: {dataset_size}")
 
    # we don't have data leakage between the main and train_critics.py because they have the same random.state() here.
    all_indices = list(range(dataset_size))
    random.shuffle(all_indices)
    
    train_size = int(0.8 * dataset_size)
    eval_size = dataset_size - train_size
    train_indices = all_indices[:train_size]
    eval_indices = all_indices[train_size:train_size+eval_size]
    eval_indices = eval_indices[:50] if data_name == "DAPO-Math-17k" else eval_indices
    # Create train and eval datasets using select (cast to avoid type checker issues)
    train_dataset = full_dataset.select(train_indices)  # type: ignore
    eval_dataset = full_dataset.select(eval_indices)    # type: ignore
    print(f"Dataset split: {len(train_dataset)} train samples, {len(eval_dataset)} eval samples")
    return train_dataset, eval_dataset
 

def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""

def _stable_wandb_run_id(output_dir, model_name, checkpoint, seed):
    """
    Persist a deterministic run id under output_dir so restarts reuse it.
    """
    rid_file = os.path.join(output_dir, ".wandb_run_id")
    if os.path.exists(rid_file):
        with open(rid_file, "r") as f:
            return f.read().strip()

    raw = f"{os.path.abspath(output_dir)}|{model_name}|sft{checkpoint}|seed{seed}"
    rid = hashlib.sha1(raw.encode()).hexdigest()[:16]
    os.makedirs(output_dir, exist_ok=True)
    with open(rid_file, "w") as f:
        f.write(rid)
    return rid


def setup_wandb(accelerator, output_dir, model_name, sft_checkpoint, seed, project_name, disable_wandb=False):
    """
    Setup wandb logging for training.
    
    Args:
        accelerator: Accelerator instance
        output_dir: Output directory for training
        model_name: Name of the model
        sft_checkpoint: SFT checkpoint path
        seed: Random seed
        project_name: Wandb project name
        disable_wandb: Whether to disable wandb logging
        
    Returns:
        run_name: Name of the wandb run (or None if disabled)
    """
    run_name = f"{output_dir}-sft{sft_checkpoint}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    if disable_wandb:
        # Disable wandb completely
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        print("Wandb logging disabled")
        return run_name
    
    if accelerator is not None:
        if accelerator.is_main_process:
            wandb_id = _stable_wandb_run_id(output_dir, model_name, sft_checkpoint, seed)
            wandb.init(project=project_name, id=wandb_id, resume="allow", name=run_name, reinit=True, mode="offline")
            return run_name
        else:
            os.environ["WANDB_DISABLED"] = "true"
            return None
    else:
        wandb.init(project=project_name, name=run_name, reinit=True, mode="offline")

# code for preprocess GSM8K
def preprocess_dataset(model_name, data_handle_dict, data_name, split=None, mode=None):
    R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
    The assistant first thinks about the reasoning process in the mind and then provides the user
    with the answer. The reasoning process and answer are enclosed within <think> and
    <answer> tags, respectively, i.e., <think> reasoning process here </think>
    <answer> answer here </answer>."""
 
    if data_name in ["GSM8K", "DAPO-Math-17k", "AIME2025"]:
        TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single integer."
    elif data_name == "CommonsenseQA":
        TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single letter (A, B, C, D, or E) corresponding to the correct choice."
    else:
        TASK_SPECIFIC_INSTRUCTIONS = ""

    if data_name == "GSM8K":
        dataset = load_dataset(data_handle_dict[data_name], split=split)
    elif data_name == "CommonsenseQA":
        dataset = load_from_disk(data_handle_dict[data_name])
        if split:
            dataset = dataset[split]
    else:
        dataset = load_from_disk(data_handle_dict[data_name])
    if data_name == "GSM8K":
        def extract_hash_answer(text: str) -> str | None:
            try:
                return text.split("####")[1].strip()
            except IndexError:
                return None
        # # GSM8K has question and answer nested in extra_info column, so flatten it
        # # First remove all columns except extra_info
        # cols_to_remove = [col for col in dataset.column_names if col != 'extra_info']
        # dataset = dataset.remove_columns(cols_to_remove)
        # # Then flatten extra_info by mapping
        # dataset = dataset.map(lambda x: x['extra_info'], remove_columns=['extra_info'])
        # dataset = dataset.rename_column("question", "prompt")
        # dataset = dataset.rename_column("answer", "labels")
    elif data_name == "MATH-500":
        dataset = dataset.rename_column("problem", "prompt")
        dataset = dataset.rename_column("answer", "labels")
        if mode =='critics':
            dataset = dataset.rename_column("solution", "reasoning")
    elif data_name == "DAPO-Math-17k":
        # ADDED on Jan 2, 2026. take only the first 7000 samples
        dataset = dataset.select(range(7000))
        def remove_prefix_suffix(x):
            prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
            suffix = "\n\nRemember to put your answer on its own line after \"Answer:\"."

            result = x
            if result.startswith(prefix):
                result = result[len(prefix):]
            if result.endswith(suffix):
                result = result[:-len(suffix)]

            return result
        def extract_fields(example):
            extracted = {
                "prompt": remove_prefix_suffix(example["prompt"][0]["content"]),
                "labels": example["reward_model"]["ground_truth"]
            }
            if mode == 'critics':
                extracted["reasoning"] = ''
            return extracted
        # Apply mapping
        dataset = dataset.map(extract_fields)
        # Optionally, keep only relevant columns
        columns_to_keep = ["prompt", "labels"]
        if mode == 'critics':
            columns_to_keep.append("reasoning")
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in columns_to_keep]
        )
    elif data_name == "CommonsenseQA":
        # Limit dataset size based on split
        if split == 'train':
            dataset = dataset.select(range(5250))
        elif split == 'validation':
            dataset = dataset.select(range(1221))
        
        def format_commonsense_qa(example):
            # Format question with choices
            question = example["question"]
            choices = example["choices"]
            labels = choices["label"]
            texts = choices["text"]
            
            # Build the formatted question with choices
            choices_str = "\n".join([f"{label}. {text}" for label, text in zip(labels, texts)])
            formatted_prompt = f"{question}\n\n{choices_str}"
            
            extracted = {
                "prompt": formatted_prompt,
                "labels": example["answerKey"]
            }
            if mode == 'critics':
                extracted["reasoning"] = ''
            return extracted
        
        # Apply mapping
        dataset = dataset.map(format_commonsense_qa)
        # Keep only relevant columns
        columns_to_keep = ["prompt", "labels"]
        if mode == 'critics':
            columns_to_keep.append("reasoning")
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in columns_to_keep]
        )
 
    def process(data):
        if data_name == "GSM8K":
            data = data["extra_info"]
            prompt = data['question']
            labels = data['answer']
            if mode == 'critics':
                reasoning = data['answer']
        else:
            prompt = data['prompt']
            labels = data['labels']
            if mode == 'critics':
                reasoning = data['reasoning']
        
        if data_name == "CommonsenseQA":
            # Use a commonsense reasoning example
            prompts = [
                {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
                {'role': 'user', 'content': "Where would you find a dog?\n\nA. park\nB. ocean\nC. space\nD. volcano\nE. cloud"},
                {'role': 'assistant', 'content': "<think>Dogs are domestic animals that live with humans on land. Let me consider each option:\n- A. park: Dogs are commonly found in parks, where owners take them for walks.\n- B. ocean: Dogs cannot live in the ocean as they are land animals.\n- C. space: Dogs cannot survive in space without special equipment.\n- D. volcano: Volcanoes are dangerous and not a place where dogs would be found.\n- E. cloud: Dogs cannot be on clouds as clouds are made of water vapor.\nThe most logical answer is A. park, as it's a common place where dogs are found.</think>\n<answer>A</answer>"},
                {'role': 'user', 'content': prompt}
            ]
        else:
            prompts = [
                {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
                {'role': 'user', 'content': "What is 2+2?"},
                {'role': 'assistant', 'content': "<think>To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.</think>\n<answer>4</answer>"},
                {'role': 'user', 'content': prompt}
            ]
        if mode == 'critics':
            reasoning = prompts + [{'role': 'assistant', 'content': "<think>"+reasoning+"</think>"}]

        return_dict = {
            "prompt": prompts,
            "labels": extract_hash_answer(labels) if data_name == "GSM8K" else labels,
        } 
        if mode == 'critics':
            return_dict["reasoning"] = reasoning
        return return_dict
    # Map over dataset to produce just two columns: prompt and labels
    processed_dataset = dataset.map(process, remove_columns=dataset.column_names)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def apply_chat_template(example):
        try:
            example["prompt"] = tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True
            )
            if mode == 'critics':
                example["reasoning"] = tokenizer.apply_chat_template(
                    example["reasoning"],
                    tokenize=False,
                    add_generation_prompt=True
                )
        except: 
            pass
        return example
    processed_dataset = processed_dataset.map(apply_chat_template)
    return processed_dataset