# accelerate launch --config_file acc_configs/Qwen-2.5-0.5B-Instruct.yaml main.py --method_name GTPO --distributed --num_devices 4 --max_completion_length 2048 --model_name "/shared/public/models/Qwen/Qwen2.5-0.5B-Instruct" --data_name "GSM8K" --wandb_project "qwen2.5-0.5B_GSM8K_nov_9" --use_vllm --disable_wandb --gradient_accumulation_steps 6 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --kl_beta 0.001
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from datasets import load_dataset, Dataset, load_from_disk
import wandb
import random
import argparse
import numpy as np
from utils import setup_wandb, parse_args, set_seed, get_dataset
from compute_reward import get_reward_function
from accelerate import Accelerator
import re
from eval import evaluate_at_k, run_multi_seed_evaluation
import csv


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        # -------debug--------
        import os
        local_rank = os.environ.get("LOCAL_RANK", "not set")
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        gpu_count = torch.cuda.device_count()
        print(f"[DEBUG] LOCAL_RANK={local_rank}, CUDA_VISIBLE_DEVICES={cuda_visible}, gpu_count={gpu_count}")
        # -------end debug--------
        accelerator = Accelerator()
    # Set random seed for reproducibility
    set_seed(args.seed)
    if args.debug:
        args.wandb_project = "Debug"
    if args.mode == "eval":
        args.wandb_project = 'EVAL_' + args.wandb_project
    
    # Setup wandb environment
    os.environ["WANDB_PROJECT"] = args.wandb_project
    reward_function = get_reward_function()
    
    # Divide the batch sizes by the number of devices to get the actual per-device batch size
    # This ensures that with 8 GPUs and batch_size=8, each GPU gets 1 sample
    num_devices = args.num_devices
    per_device_train_batch = args.per_device_train_batch_size // num_devices
    per_device_eval_batch = args.per_device_eval_batch_size // num_devices
    
    if args.mode == "train" and accelerator.is_main_process:
        print(f"Total batch size specified: train={args.per_device_train_batch_size}, eval={args.per_device_eval_batch_size}")
        print(f"Number of devices: {num_devices}")
        print(f"Actual per-device batch size: train={per_device_train_batch}, eval={per_device_eval_batch}")

    # run_name = f"{args.model_name.replace('/','-')}-{args.data_name.replace('/','-')}-{args.method_name.replace('/','-')}"
    # wandb.init(project=args.wandb_project, name=run_name, mode="offline")
    output_dir = f"/shared/user/agent_rl/{args.wandb_project}/{args.model_name.replace('/','-')}-{args.data_name.replace('/','-')}-kl-{args.kl_beta}-{args.method_name.replace('/','-')}"
    if args.method_name in ['G2RPO', 'G2RPOCritics']:
        output_dir += f'-surp-{args.surprise_weight}-conf-{args.confidence_weight}-horizon-{args.surprise_horizon}-decay-{args.surprise_decay}'
    if args.method_name == 'G2RPOCritics':
        output_dir += f'-critics_llm_freezed_{args.critics_llm_freezed}'
    os.makedirs(output_dir, exist_ok=True)
    print('created output directory:', output_dir)
    
    
    
    config_class = getattr(trl, f"{args.method_name}Config")
    model_init_kwargs = {
                    # "attn_implementation": "eager", # for eager attention
                    "use_cache": False, 
                    "torch_dtype": torch.bfloat16
                    } # for bfloat16
    if args.predict_mode != 'last_token':
        model_init_kwargs.update({
            "attn_implementation": "flash_attention_2", # for flash attention 2
        })
    # Define shared RL args
    # Determine report_to based on disable_wandb flag
    if args.disable_wandb:
        report_to = "none"
    else:
        report_to = "wandb" if (args.mode == "eval" or accelerator.is_main_process) else "none"
    
    train_dataset, eval_dataset = get_dataset(args.data_name, args.model_name)
    # Split dataset into train (80%) and eval (20%) using a simple approach
    run_name = setup_wandb(accelerator if args.mode == 'train' else None,
                           output_dir,
                           args.model_name,
                           args.sft_checkpoint,
                           args.seed,
                           args.wandb_project,
                           disable_wandb=args.disable_wandb)
    
    if args.mode == "train":
        base_rl_args = dict(
            model_init_kwargs=model_init_kwargs,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            beta=args.kl_beta,
            report_to=report_to,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            do_eval=True,
            output_dir=output_dir,
            max_completion_length=args.max_completion_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # gradient_checkpointing=True if args.predict_mode != 'last_token' else False, # For memory efficiency, comment if you have enough memory
            gradient_checkpointing=True,
            per_device_train_batch_size=per_device_train_batch,  # Mini batch size per device
            per_device_eval_batch_size=per_device_eval_batch,
            learning_rate=args.learning_rate,
            seed=args.seed,
            run_name=run_name,
            save_strategy="steps",      # or "epoch", "no"
            save_steps=100,             # Save every 100 steps        # Keep only last 3 checkpoints
            save_only_model=False,      # Save optimizer/scheduler states too
            # gradient_accumulation_steps=4,
            dataloader_num_workers=0,   # Disable multiprocessing to avoid SIGBUS errors from shared memory exhaustion
            num_generations=args.num_generations,
            logging_steps=args.logging_steps,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
        )
        
        
        # Add vLLM args if enabled
        if args.use_vllm:
            base_rl_args.update({
                "use_vllm": True,
                "vllm_mode": args.vllm_mode,
                "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            })
        # Conditionally add method-specific args
        if args.method_name == "SGRPOLee":
            rl_args_dict = base_rl_args.copy()
            rl_args_dict.update({
                "alpha": args.alpha,
                "k": args.k,
                "token_prob": args.token_prob,
                "loss_type":'grpo'
            })
            rl_args = config_class(**rl_args_dict)
        elif args.method_name in ["G2RPO", "G2RPOCritics"]:
            # Finally, create rl_args
            # add surprise_weight=args.surprise_weight,
            # confidence_weight=args.confidence_weight,
            rl_args_dict = base_rl_args.copy()
            rl_args_dict.update({
                "surprise_weight": args.surprise_weight,
                "confidence_weight": args.confidence_weight,
                "surprise_horizon": args.surprise_horizon,
                "surprise_decay": args.surprise_decay,
                "epsilon_high": 0.28,
                "loss_type":'bnpo',
            })
            if args.method_name == "G2RPO":
                rl_args_dict.update({
                    "model_name": args.model_name,
                    "data_name": args.data_name,
                    "predict_mode": args.predict_mode
                })
                if args.predict_mode == 'last_token':
                    rl_args_dict.update({
                        "torch_empty_cache_steps": 1,
                    })
            elif args.method_name == "G2RPOCritics":
                rl_args_dict.update({
                    "critics_path": os.path.join(args.critics_path, args.data_name, f'critics_llm_freezed') if args.critics_llm_freezed else os.path.join(args.critics_path, args.data_name, 'critics_full_trained')
                })
            rl_args = config_class(**rl_args_dict)
        elif args.method_name == "GFPO":
            rl_args_dict = base_rl_args.copy()
            rl_args_dict.update({
                "top_num_gen": args.top_num_gen,
                "loss_type":'grpo'
            })
            rl_args = config_class(**rl_args_dict)
        elif args.method_name == "DAPO":
            rl_args_dict = base_rl_args.copy()
            rl_args_dict.update({
                "loss_type":'bnpo',
                "epsilon_high": 0.28
            })
            rl_args = config_class(**rl_args_dict)
        elif args.method_name == "GTPO":
            rl_args_dict = base_rl_args.copy()
            rl_args_dict.update({
                "loss_type":'bnpo'
            })
            rl_args = config_class(**rl_args_dict)
        else:
            rl_args = config_class(**base_rl_args)
            print(f"Find {args.method_name} in original TRL package")

        # SFT Training
        if args.sft:
            sft_trainer = trl.SFTTrainer(
                model=args.model_name,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=trl.SFTConfig(
                    max_steps=args.sft_max_steps,
                    per_device_train_batch_size=per_device_train_batch,
                    per_device_eval_batch_size=per_device_eval_batch,
                    output_dir=f"/shared/user/agent_rl/sft_model_{args.model_name.replace('/','-')}_{args.seed}",
                    report_to=report_to,  # Use the same report_to as RL training
                    eval_strategy="steps",
                    eval_steps=args.sft_eval_steps,
                    do_eval=True,
                    learning_rate=1e-6,
                    seed=args.seed,
                )
            )
            sft_trainer.train(resume_from_checkpoint=args.resume)
            # After SFT, use the trained model for RL
            args.model_name = "/shared/user/agent_rl/sft_model"
        # Get the trainer class dynamically
        
        trainer_class = getattr(trl, f"{args.method_name}Trainer")
        
        trainer = trainer_class(
            model=args.model_name,  # Start from SFT model
            train_dataset=train_dataset,
            # compute_metrics=compute_accuracy,  # Compute accuracy metric during eval
            eval_dataset=eval_dataset,  # Now using separate eval set (20% of data)
            reward_funcs=reward_function,  # This is used for both training and evaluation
            args=rl_args,
            )
        trainer.train(args.resume)
        accelerator.end_training()

    elif args.mode == "eval":
        if args.eval_untrained_model:
            # TODO: mimic when args.mode is eval, load from eval_checkpoint_dir (I already specified it to untrained models' ckpts). However, csv path should be changed. you can save to folder untrained_model_eval/{model}/{data}/eval_results.csv. you can only add after line 343 
            # Number of evaluation runs for computing mean and variance
            num_eval_runs = 3
            
            # Initialize CSV file for saving results - save to untrained_model_eval/{model}/{data}/
            model_name_clean = args.model_name.replace('/', '-')
            data_name_clean = args.data_name.replace('/', '-')
            eval_output_dir = os.path.join('/shared/user/agent_rl/untrained_model_eval', model_name_clean, data_name_clean)
            os.makedirs(eval_output_dir, exist_ok=True)
            csv_path = os.path.join(eval_output_dir, 'eval_results.csv')
            csv_exists = os.path.exists(csv_path)
            
            # Create CSV with headers if it doesn't exist
            k_values = [2, 4, 8, 16, 32]
            if not csv_exists:
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    headers = ['model', 'run'] + [f'pass@{k}' for k in k_values] + [f'length@{k}' for k in k_values]
                    writer.writerow(headers)
                print(f"Created results CSV file at: {csv_path}")
            else:
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([])
                print(f"Appending to existing CSV file at: {csv_path}")
            
            print(f"Evaluating untrained model from {args.eval_checkpoint_dir} with {num_eval_runs} runs...")
            
            run_multi_seed_evaluation(
                model_path=args.eval_checkpoint_dir,
                eval_dataset=eval_dataset,
                reward_func=reward_function[1],
                args=args,
                csv_path=csv_path,
                identifier=model_name_clean,
                identifier_label="Untrained Model",
                num_eval_runs=num_eval_runs,
                base_seed=args.seed,
                k_values=k_values
            )
        else:
            # load checkpoints checkpoint-100, checkpoint-200, checkpoint-300, ..., checkpoint-N one by one and evaluate 
            # first find all ckpt numbers
            
            # ckpt_numbers = [max([int(ckpt.split('-')[-1]) for ckpt in os.listdir(args.eval_checkpoint_dir) if ckpt.startswith('checkpoint-')])]
            with open(os.path.join(args.eval_checkpoint_dir, 'best_steps.txt'), 'r') as f:
                ckpt_numbers=[int(line.strip()) for line in f.readlines()]
            # ckpt_numbers = [10000]
            
            # Number of evaluation runs for computing mean and variance
            num_eval_runs = 3
            
            # Initialize CSV file for saving results
            csv_path = os.path.join(args.eval_checkpoint_dir, 'eval_results.csv')
            csv_exists = os.path.exists(csv_path)
            
            # Create CSV with headers if it doesn't exist
            k_values = [2, 4, 8, 16, 32]
            if not csv_exists:
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Headers include: checkpoint, run, individual metrics, and aggregated stats (mean/std)
                    headers = ['checkpoint', 'run'] + [f'pass@{k}' for k in k_values] + [f'length@{k}' for k in k_values]
                    writer.writerow(headers)
                print(f"Created results CSV file at: {csv_path}")
            else:
                # add a blank line at csv
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([])
                print(f"Appending to existing CSV file at: {csv_path}")
            
            best_token_effi=0
            for ckpt_number in ckpt_numbers:
                print(f"Evaluating checkpoint {ckpt_number} with {num_eval_runs} runs...")
                checkpoint_dir = os.path.join(args.eval_checkpoint_dir, f'checkpoint-{ckpt_number}')
                
                mean_pass_at_k, std_pass_at_k, mean_length_at_k, std_length_at_k, mean_time_at_k, std_time_at_k = run_multi_seed_evaluation(
                    model_path=checkpoint_dir,
                    eval_dataset=eval_dataset,
                    reward_func=reward_function[1],
                    args=args,
                    csv_path=csv_path,
                    identifier=ckpt_number,
                    identifier_label=f"Checkpoint {ckpt_number}",
                    num_eval_runs=num_eval_runs,
                    base_seed=args.seed,
                    k_values=k_values
                )
                
                if mean_pass_at_k[32]/mean_length_at_k[32] > best_token_effi:
                    best_token_effi = mean_pass_at_k[32]/mean_length_at_k[32]
                    best_token_effi_ckpt = ckpt_number
                    best_token_effi_row = [ckpt_number, 'best_token_effi']
                    for k in k_values:
                        best_token_effi_row.append(f"{mean_pass_at_k[k]:.4f}±{std_pass_at_k[k]:.4f}")
                    for k in k_values:
                        best_token_effi_row.append(f"{mean_length_at_k[k]:.2f}±{std_length_at_k[k]:.2f}")
                    for k in k_values:
                        best_token_effi_row.append(f"{mean_time_at_k[k]:.2f}±{std_time_at_k[k]:.2f}")
                    
                    
            print("Write the best token efficiency to the file")
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(best_token_effi_row)


    else:
        raise ValueError(f"Invalid mode: {args.mode}")
