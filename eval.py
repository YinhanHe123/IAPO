import torch
import os
import math
import csv
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
 
def evaluate_at_k(model, tokenizer, dataset, reward_function,
                  use_vllm=False, vllm_gpu_memory_utilization=0.4,
                  vllm_tensor_parallel_size=1, model_name_or_path=None,
                  seed=42, measure_time=False):
    """
    Evaluate pass@k and average length@k metrics.
 
    Args:
        model: The language model to evaluate (ignored if use_vllm=True)
        tokenizer: The tokenizer for the model
        dataset: The evaluation dataset containing prompts and answers
        reward_function: Function to compute rewards for completions
        use_vllm: Whether to use vLLM for generation
        vllm_gpu_memory_utilization: GPU memory utilization for vLLM
        vllm_tensor_parallel_size: Tensor parallel size for vLLM
        model_name_or_path: Path to model for vLLM initialization
 
    Returns:
        pass_at_k: Dictionary with keys [2, 4, 8, 16, 32] containing pass@k values
        length_at_k: Dictionary with keys [2, 4, 8, 16, 32] containing average lengths
    """
    dataset = dataset.select(range(15)) # for debugging
    k_values = [2, 4, 8, 16, 32]
    max_k = 32
 
    # Store results for each prompt
    all_accuracies = [0 for _ in range(len(k_values))]
    all_lengths = [0 for _ in range(len(k_values))]
    
    # Initialize vLLM if needed
    if use_vllm:
        print(f"Initializing vLLM with tensor_parallel_size={vllm_tensor_parallel_size}, gpu_memory_utilization={vllm_gpu_memory_utilization}") # for debugging
        llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=vllm_tensor_parallel_size,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            trust_remote_code=True,
            max_num_batched_tokens=4096,
            max_model_len=768, # Added on Jan 2, all results before does not have it
        )
        print("vLLM initialized!!")
 
        # presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2048, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None
        sampling_params = SamplingParams(
            n=max_k,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            seed=seed,
            stop=[],
            stop_token_ids=[],  
            bad_words=[],
            include_stop_str_in_output=False,
            ignore_eos=False,
            min_tokens=0,
            logprobs=None,
            prompt_logprobs=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            truncate_prompt_tokens=None,
            guided_decoding=None,
            extra_args=None,
            max_tokens=2048,
        )
    else:
        # not implemented
        raise NotImplementedError("Only vLLM-based evaluation is implemented in this function.")
 
    # Each GPU processes its own subset
    correctness_list, completion_lengths_list, time_list = [], [], []
    rows=[["prompt", "completion_length", "completion", "correctness"]] # JAN 5: Documenting completions to csv
    for example in tqdm(dataset, desc=f"Evaluating"):
        prompt = example['prompt']
        ground_truth = example['labels']
        # Use vLLM for batch generation
        if measure_time:
            start_time = time.time()
        outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        if measure_time:
            end_time = time.time()
            time_list.append(end_time - start_time)
        else:
            time_list.append(None)
        # Extract completions from vLLM output
        completions = []
        completion_lengths = []
        for output_obj in outputs[0].outputs:
            completion = output_obj.text
            completions.append(completion)
            print('output_obj.token_ids:', output_obj.token_ids)
            print('output_obj.text:', output_obj.text)
            completion_lengths.append(len(output_obj.token_ids))

        # Verify we got exactly max_k completions
        if len(completions) != max_k:
            raise ValueError(
                f"Expected {max_k} completions but got {len(completions)} for prompt: {prompt[:100]}..."
            )

        result = reward_function(completions, [ground_truth]*max_k)
        # Extract correctness (0 or 1)
        correctness = [r.get('correctness', 0.0) for r in result]

        correctness_list.append(correctness)
        completion_lengths_list.append(completion_lengths)
        # JAN 5: Documenting completions to csv
        for i in range(len(correctness)):
            rows.append([prompt, completion_lengths[i], completions[i], correctness[i]])
    # Stack local results into tensors (each GPU has its own subset)
    if len(correctness_list) > 0:
        correctness = torch.tensor(correctness_list)
        lengths = torch.tensor(completion_lengths_list)
        times = torch.tensor(time_list).unsqueeze(1).expand(-1, max_k)
    else:
        # Handle case where this GPU has no samples
        correctness = torch.zeros((0, max_k))
        lengths = torch.zeros((0, max_k))
        times = torch.zeros((0, max_k))
    # JAN 5: Documenting completions to csv
    from pathlib import Path
    csv_path = os.path.join(model_name_or_path, 'eval_results_text.csv')
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    # For each k value, compute mean accuracy and length using first k completions
    # This uses the FULL validation dataset (gathered from all GPUs)
    for k in k_values:
        # mean_acc = correctness[:, :k].sum() / (k * correctness.shape[0])
        pass_acc = (correctness[:, :k].sum(1) >= 1).float().sum() / correctness.shape[0]
        mean_length = lengths[:, :k].sum() / (k * lengths.shape[0])
        all_accuracies[int(math.log2(k))-1] = pass_acc.item()
        all_lengths[int(math.log2(k))-1] = mean_length.item()

    # Compute overall pass@k and length@k by averaging across all prompts
    pass_at_k = {k: all_accuracies[int(math.log2(k))-1] for k in k_values}
    length_at_k = {k: all_lengths[int(math.log2(k))-1] for k in k_values}
    time_at_k = {k: times[:, :k].sum() / (k * times.shape[0]) for k in k_values}
    return pass_at_k, length_at_k, time_at_k


def run_multi_seed_evaluation(
    model_path,
    eval_dataset,
    reward_func,
    args,
    csv_path,
    identifier,
    identifier_label,
    num_eval_runs=3,
    base_seed=42,
    k_values=[2, 4, 8, 16, 32],
    measure_time=True
):
    """
    Run evaluation with multiple seeds for a single model/checkpoint.
    
    Args:
        model_path: Path to load model and tokenizer from
        eval_dataset: Dataset to evaluate on
        reward_func: Reward function for evaluation
        args: Arguments containing vllm settings
        csv_path: Path to CSV file for saving results
        identifier: Value for the first CSV column (e.g., checkpoint number or model name)
        identifier_label: Label for print messages (e.g., "Checkpoint 100" or "Untrained Model")
        num_eval_runs: Number of evaluation runs with different seeds
        base_seed: Base seed for generating eval seeds
        k_values: List of k values for pass@k evaluation
        measure_time: Whether to measure time
    Returns:
        Tuple of (mean_pass_at_k, std_pass_at_k, mean_length_at_k, std_length_at_k) dicts
    """
    eval_seeds = [base_seed + i * 100 for i in range(num_eval_runs)]
    
    # Load model (not needed if using vLLM)
    if not args.use_vllm:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model = None  # vLLM will load the model internally
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Store results from all runs
    all_pass_at_k = {k: [] for k in k_values}
    all_length_at_k = {k: [] for k in k_values}
    all_time_at_k = {k: [] for k in k_values}

    for run_idx, eval_seed in enumerate(eval_seeds):
        print(f"\n--- Run {run_idx + 1}/{num_eval_runs} (seed={eval_seed}) ---")
        pass_at_k, length_at_k, time_at_k = evaluate_at_k(
            model, 
            tokenizer, 
            eval_dataset, 
            reward_func,
            use_vllm=args.use_vllm,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            model_name_or_path=model_path,
            seed=eval_seed,
            measure_time=measure_time
        )
        
        # Store results
        for k in k_values:
            all_pass_at_k[k].append(pass_at_k[k])
            all_length_at_k[k].append(length_at_k[k])
            all_time_at_k[k].append(time_at_k[k])
        # Print individual run results
        print(f"Run {run_idx + 1} Results:")
        print(f"Pass@k: {pass_at_k}")
        print(f"Length@k: {length_at_k}")
        print(f"Time@k: {time_at_k}")

        # Save individual run results to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [identifier, f'run_{run_idx + 1}']
            for k in k_values:
                row.append(pass_at_k[k])
            for k in k_values:
                row.append(length_at_k[k])
            for k in k_values:
                row.append(time_at_k[k])
            writer.writerow(row)
    
    # Compute mean and std for each metric
    mean_pass_at_k = {k: np.mean(all_pass_at_k[k]) for k in k_values}
    std_pass_at_k = {k: np.std(all_pass_at_k[k]) for k in k_values}
    mean_length_at_k = {k: np.mean(all_length_at_k[k]) for k in k_values}
    std_length_at_k = {k: np.std(all_length_at_k[k]) for k in k_values}
    mean_time_at_k = {k: np.mean(all_time_at_k[k]) for k in k_values}
    std_time_at_k = {k: np.std(all_time_at_k[k]) for k in k_values}
    # Print aggregated results
    print(f"\n=== {identifier_label} Aggregated Results ({num_eval_runs} runs) ===")
    print("Pass@k (mean±std):", {k: f"{mean_pass_at_k[k]:.4f}±{std_pass_at_k[k]:.4f}" for k in k_values})
    print("Length@k (mean±std):", {k: f"{mean_length_at_k[k]:.2f}±{std_length_at_k[k]:.2f}" for k in k_values})
    print("Time@k (mean±std):", {k: f"{mean_time_at_k[k]:.2f}±{std_time_at_k[k]:.2f}" for k in k_values})
    # Save mean±std row to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        summary_row = [identifier, 'mean±std']
        for k in k_values:
            summary_row.append(f"{mean_pass_at_k[k]:.4f}±{std_pass_at_k[k]:.4f}")
        for k in k_values:
            summary_row.append(f"{mean_length_at_k[k]:.2f}±{std_length_at_k[k]:.2f}")
        for k in k_values:
            summary_row.append(f"{mean_time_at_k[k]:.2f}±{std_time_at_k[k]:.2f}")
        writer.writerow(summary_row)
    
    print(f"Saved {identifier_label} results (including mean±std) to {csv_path}")
    
    return mean_pass_at_k, std_pass_at_k, mean_length_at_k, std_length_at_k, mean_time_at_k, std_time_at_k