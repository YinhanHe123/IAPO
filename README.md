# IAPO: Information-Aware Policy Optimization

Official implementation for the paper **"Information-Aware Policy Optimization"**

## 📖 Abstract

Reinforcement learning (RL)-based post-training methods such as GRPO significantly improve the reasoning accuracy of large language models (LLMs), but often incur excessive token usage. Existing token-efficient approaches rely on length- or position-based heuristics for token advantage assignment, which are content-agnostic and fail to distinguish informative tokens from verbose reasoning.  We propose **Information-Aware Policy Optimization (IAPO)**, a post-training framework that assigns token-level advantages based on each token's conditional mutual information (MI) with the final answer. This information-theoretic design explicitly promotes informative reasoning tokens while suppressing redundant generation. To enable conditional MI estimation, we introduce an early-exit-based conditional MI estimator. To accelerate training, we propose KV-cache preloading and chunk-wise forwarding techniques to reduce computational overhead. We theoretically show that IAPO reduces expected completion length while preserving reasoning accuracy, and empirically validate its effectiveness on multiple mathematical reasoning benchmarks and model scales. **IAPO consistently outperforms state-of-the-art baselines in token efficiency, achieving up to 36% token reduction without sacrificing accuracy.**

## 🏗️ Project Structure

```
agent_rl_proj/
├── src/
│   ├── main.py                    # Main training/evaluation entry point
│   ├── eval.py                    # Evaluation utilities (pass@k, length@k)
│   ├── utils.py                   # Dataset loading, argument parsing, utilities
│   ├── compute_reward.py          # Reward function definitions
│   ├── train_critics.py           # Critics model training
│   ├── requirements.txt           # Python dependencies
│   ├── acc_configs/               # Accelerate configurations per model
│   │   ├── Qwen2.5-0.5B-Instruct.yaml
│   │   ├── Qwen2.5-1.5B-Instruct.yaml
│   │   └── Qwen2.5-7B-Instruct.yaml
│   ├── package_code/              # Custom TRL trainer implementations
│   │   └── trainer/
│   │       ├── g2rpo_trainer.py   # IAPO (G2RPO) trainer
│   │       ├── g2rpo_config.py    # IAPO configuration
│   │       ├── dapo_trainer.py    # DAPO baseline
│   │       ├── gtpo_trainer.py    # GTPO baseline
│   │       ├── gfpo_trainer.py    # GFPO baseline
│   │       └── ...                # Other trainers
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 12.6+ (for GPU acceleration)
- 4-8 GPUs recommended (tested on A100/H100)

### Setup

```bash
# Install dependencies
pip install -r src/requirements.txt

# Set environment variables
export WANDB_MODE="offline"
export WANDB_DIR="/shared/user/agent_rl/"

# Install custom TRL trainers (required for IAPO and baselines)
yes | cp -rf src/package_code/trainer/* ~/.local/lib/python3.10/site-packages/trl/trainer/
yes | cp -rf src/package_code/__init__.py ~/.local/lib/python3.10/site-packages/trl/
```

### Key Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| `trl` | 0.21.0 | Transformer RL library (with vLLM support) |
| `torch` | 2.7.1+cu128 | PyTorch with CUDA |
| `vllm` | 0.10.0 | Fast LLM inference |
| `deepspeed` | latest | Distributed training |
| `flash_attn` | 2.7.4.2 | Flash Attention 2 |

## 🎯 Training

### Quick Start

Train IAPO (G2RPO) on DAPO-Math-17k with Qwen2.5-1.5B:

```bash
bash scripts/200-rl-run.sh
```

### Configuration

Modify the first few lines in `scripts/200-rl-run.sh` to customize your experiment:

```bash
DATA_NAME="DAPO-Math-17k"       # Dataset: DAPO-Math-17k | GSM8K | MATH-500
MODEL_NAME="/path/to/Qwen2.5-1.5B-Instruct"  # Model path
METHOD_NAME="G2RPO"             # Method: G2RPO | GRPO | GTPO | DAPO | GFPO | SGRPOLee
RESUME="0"                      # Resume from checkpoint: 0 | 1
G2RPO_PREDICT_MODE="last_token" # Prediction mode: last_token | next_token
SURPRISE_WEIGHT="1"             # Surprise weight (MI coefficient)
CONFIDENCE_WEIGHT="1"           # Confidence weight
```

### Supported Methods

| Method | Description | Key Arguments |
|--------|-------------|---------------|
| **G2RPO** | IAPO (Ours) | `--surprise_weight`, `--confidence_weight`, `--predict_mode` |
| GRPO | Group Relative Policy Optimization | - |
| GTPO | Group Token Policy Optimization | - |
| DAPO | Dynamic Advantage Policy Optimization | - |
| GFPO | Group Filtered Policy Optimization | `--top_num_gen` |
| SGRPOLee | Stochastic GRPO | `--alpha`, `--k`, `--token_prob` |

### Hyperparameter Search

For reproducing our results, adjust `SURPRISE_WEIGHT` and `CONFIDENCE_WEIGHT` within:

```bash
[1e-6, 1e-4, 1e-2, 1]
```

### Full Training Script

The training script handles different model sizes, datasets, and methods with appropriate default parameters. You can modify the first 8 lines to configure your experiment:

```bash
#!/bin/bash

# ============== Configuration (modify these) ==============
DATA_NAME="DAPO-Math-17k"                                    # Dataset: DAPO-Math-17k | GSM8K | MATH-500
MODEL_NAME="/shared/public/models/Qwen/Qwen2.5-1.5B-Instruct" # Model path
METHOD_NAME="G2RPO"                                          # Method: G2RPO | GRPO | GTPO | DAPO | GFPO | SGRPOLee
RESUME="0"                                                   # Resume from checkpoint: 0 | 1
G2RPO_PREDICT_MODE="last_token"                              # Prediction mode: last_token | next_token
SURPRISE_WEIGHT="1e-6"                                       # Surprise weight (MI coefficient)
CONFIDENCE_WEIGHT="1e-6"                                     # Confidence weight

# ============== Auto-configured variables ==============
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume" || RESUME_FLAG=""
MODEL_BASE=$(basename ${MODEL_NAME})

# Environment setup
export WANDB_MODE="offline"
export WANDB_DIR="/shared/user/agent_rl/"

# Increase NCCL timeout to prevent deadlock during ZeRO-3 + vLLM weight sync
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800

# Set gradient accumulation based on model size
if [ "${MODEL_BASE}" = "Qwen2.5-7B-Instruct" ]; then
    GRADIENT_ACCUMULATION_STEPS=8
else
    GRADIENT_ACCUMULATION_STEPS=6
fi

# Set wandb project name
if [ "${SURPRISE_WEIGHT}" = "1e-6" ] && [ "${CONFIDENCE_WEIGHT}" = "1e-6" ]; then
    WANDB_PROJECT="${MODEL_BASE}_${DATA_NAME}_${METHOD_NAME}_${G2RPO_PREDICT_MODE}"
else
    WANDB_PROJECT="${MODEL_BASE}_${DATA_NAME}_${METHOD_NAME}_${G2RPO_PREDICT_MODE}_surp-${SURPRISE_WEIGHT}_conf-${CONFIDENCE_WEIGHT}"
fi

# ============== Training launch ==============
# Small models (0.5B, 1.5B): 4 GPUs
if [ "${MODEL_BASE}" = "Qwen2.5-0.5B-Instruct" ] || [ "${MODEL_BASE}" = "Qwen2.5-1.5B-Instruct" ]; then
    NUM_DEVICES=4
    
    if [ "${DATA_NAME}" = "MATH-500" ]; then
        EXTRA_ARGS="--num_train_epochs 152"
    else
        EXTRA_ARGS=""
    fi
    
    if [ "${METHOD_NAME}" = "G2RPO" ]; then
        accelerate launch --config_file acc_configs/${MODEL_BASE}.yaml main.py \
            --method_name ${METHOD_NAME} --distributed --num_devices ${NUM_DEVICES} \
            --max_completion_length 2048 --model_name ${MODEL_NAME} --data_name ${DATA_NAME} \
            --wandb_project "${WANDB_PROJECT}" --use_vllm --disable_wandb \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
            --kl_beta 0.001 ${EXTRA_ARGS} ${RESUME_FLAG} \
            --predict_mode ${G2RPO_PREDICT_MODE} \
            --surprise_weight ${SURPRISE_WEIGHT} --confidence_weight ${CONFIDENCE_WEIGHT}
    else
        accelerate launch --config_file acc_configs/${MODEL_BASE}.yaml main.py \
            --method_name ${METHOD_NAME} --distributed --num_devices ${NUM_DEVICES} \
            --max_completion_length 2048 --model_name ${MODEL_NAME} --data_name ${DATA_NAME} \
            --wandb_project "${WANDB_PROJECT}" --use_vllm --disable_wandb \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
            --kl_beta 0.001 ${EXTRA_ARGS} ${RESUME_FLAG} \
            --surprise_weight ${SURPRISE_WEIGHT} --confidence_weight ${CONFIDENCE_WEIGHT}
    fi

# Large models (7B): 8 GPUs
else
    NUM_DEVICES=8
    
    if [ "${DATA_NAME}" = "MATH-500" ]; then
        EXTRA_ARGS="--num_train_epochs 152"
    else
        EXTRA_ARGS=""
    fi
    
    if [ "${METHOD_NAME}" = "G2RPO" ]; then
        accelerate launch --config_file acc_configs/${MODEL_BASE}.yaml main.py \
            --method_name ${METHOD_NAME} --distributed --num_devices ${NUM_DEVICES} \
            --max_completion_length 2048 --model_name ${MODEL_NAME} --data_name ${DATA_NAME} \
            --wandb_project "${WANDB_PROJECT}" --use_vllm --disable_wandb \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
            --kl_beta 0.001 ${EXTRA_ARGS} ${RESUME_FLAG} \
            --predict_mode ${G2RPO_PREDICT_MODE} \
            --surprise_weight ${SURPRISE_WEIGHT} --confidence_weight ${CONFIDENCE_WEIGHT}
    else
        accelerate launch --config_file acc_configs/${MODEL_BASE}.yaml main.py \
            --method_name ${METHOD_NAME} --distributed --num_devices ${NUM_DEVICES} \
            --max_completion_length 2048 --model_name ${MODEL_NAME} --data_name ${DATA_NAME} \
            --wandb_project "${WANDB_PROJECT}" --use_vllm --disable_wandb \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
            --kl_beta 0.001 ${EXTRA_ARGS} ${RESUME_FLAG} \
            --surprise_weight ${SURPRISE_WEIGHT} --confidence_weight ${CONFIDENCE_WEIGHT}
    fi
fi
```

### Configuration Notes

| Model Size | GPUs | Gradient Accumulation | Special Args |
|------------|------|----------------------|--------------|
| 0.5B-1.5B | 4 | 6 | - |
| 7B | 8 | 8 | - |
| MATH-500 | - | - | `--num_train_epochs 152` |

## 📊 Evaluation

### Checkpoint Selection

1. Identify the top 5 checkpoints with highest validation correctness reward in `trainer_state.json`
2. Save checkpoint steps to `best_steps.txt` in the checkpoint directory
3. Run evaluation on selected checkpoints

### Running Evaluation

```bash
python src/main.py \
    --mode eval \
    --model_name /path/to/model \
    --data_name DAPO-Math-17k \
    --eval_checkpoint_dir /path/to/checkpoints \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 2
```

### Evaluation Metrics

- **Pass@k**: Percentage of problems solved within k generations (k = 2, 4, 8, 16, 32)
- **Length@k**: Average token length of completions
- **Token Efficiency**: Pass@k / Length@k ratio

Results are saved to `eval_results.csv` in the checkpoint directory with format:
```
checkpoint, run, pass@2, pass@4, ..., pass@32, length@2, length@4, ..., length@32
```

### Evaluating Untrained Models

```bash
python src/main.py \
    --mode eval \
    --eval_untrained_model \
    --model_name /path/to/model \
    --data_name DAPO-Math-17k \
    --eval_checkpoint_dir /path/to/untrained/model
```

## 📈 Supported Datasets

| Dataset | Size | Split | Description |
|---------|------|-------|-------------|
| DAPO-Math-17k | 7,000 (first) | 80/20 train/eval | Mathematical reasoning problems |
| GSM8K | Full | train/test | Grade school math problems |
| MATH-500 | 500 | 80/20 train/eval | Competition math problems |

## 🔧 Key Arguments

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method_name` | G2RPO | RL method to use |
| `--model_name` | - | Path to base model |
| `--data_name` | DAPO-Math-17k | Dataset name |
| `--num_devices` | 1 | Number of GPUs |
| `--max_completion_length` | 2048 | Max generation length |
| `--learning_rate` | 1e-6 | Learning rate |
| `--kl_beta` | 0.001 | KL divergence coefficient |
| `--num_generations` | 8 | Generations per prompt |
| `--gradient_accumulation_steps` | 6 | Gradient accumulation |
| `--use_vllm` | False | Enable vLLM acceleration |

### IAPO (G2RPO) Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--surprise_weight` | 1e-6 | Weight for MI-based surprise bonus |
| `--confidence_weight` | 1e-6 | Weight for confidence bonus |
| `--surprise_horizon` | 1 | Horizon for surprise calculation |
| `--surprise_decay` | 0.5 | Decay rate for surprise bonus |
| `--predict_mode` | next_token | Prediction mode: `next_token` or `last_token` |

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{iapo2026,
  title={Information-Aware Policy Optimization},
  author={...},
  journal={...},
  year={2026}
}
```

## 📄 License

This project builds upon the [TRL library](https://github.com/huggingface/trl) (Apache 2.0 License) with custom modifications for IAPO training.

## 🙏 Acknowledgments

- [Hugging Face TRL](https://github.com/huggingface/trl) for the base RL training framework
- [vLLM](https://github.com/vllm-project/vllm) for fast inference
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) for distributed training
