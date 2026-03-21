"""
MoE (Qwen2Moe) pretraining script for PEFT framework.
Supports Pile dataset (HuggingFace online or local offline).
Supports AdamW and RL_AdamW_Wrapper optimizers.
"""

import os
import sys
import time
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2MoeConfig

import datasets
import datasets.distributed

import wandb
from tqdm import tqdm
from loguru import logger

from para_eff_pt.peft_pretraining import training_utils
from para_eff_pt.peft_pretraining.dataloader import PreprocessedIterableDataset

from para_eff_pt.pt_rl_opt.rl_optimizer import RL_AdamW_Wrapper

transformers.logging.set_verbosity_error()

# ──────────────────────────────────────────────────────────────────────────────
# Qwen2Moe model configurations
# ──────────────────────────────────────────────────────────────────────────────

# pad_token_id=0 collides with bos_token_id=0 in llama2 tokenizer, which causes
# attention_mask to be all-zeros for padded positions that happen to share the same
# token id as real tokens, leading to empty token tensors inside
# load_balancing_loss_func.  Use vocab_size-1 (31999) as pad instead — it is never
# emitted by normal text and is safely outside the bos/eos ids.
_PAD_TOKEN_ID = 31999  # shared constant; must match vocab_size - 1

MOE_CONFIGS = {
    "qwen2-1B": dict(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=15,
        num_attention_heads=12,
        num_key_value_heads=12,
        hidden_act="silu",
        max_position_embeddings=1024,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=False,
        rope_theta=10000.0,
        attention_dropout=0.0,
        moe_intermediate_size=768,
        shared_expert_intermediate_size=3072,
        norm_topk_prob=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=_PAD_TOKEN_ID,
    ),
    "qwen2-2B": dict(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=1024,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=False,
        rope_theta=10000.0,
        attention_dropout=0.0,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=4096,
        norm_topk_prob=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=_PAD_TOKEN_ID,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(args):
    parser = argparse.ArgumentParser(description="Qwen2Moe pretraining with PEFT framework")

    # ── Model ──
    parser.add_argument("--moe_config", type=str, required=True,
                        choices=list(MOE_CONFIGS.keys()),
                        help="MoE model size preset, e.g. 'qwen2-1B'")
    parser.add_argument("--num_experts", type=int, default=32,
                        help="Total number of experts in each MoE layer")
    parser.add_argument("--num_experts_per_tok", type=int, default=4,
                        help="Number of experts activated per token")
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.01,
                        help="Coefficient for load-balancing auxiliary loss (built into HF MoE)")
    parser.add_argument("--router_z_loss_coef", type=float, default=1e-3,
                        help="Coefficient for router z-loss (additional regularization)")

    # ── Tokenizer ──
    parser.add_argument("--tokenizer_path", type=str, default="./llama2tokenizer",
                        help="Path to the tokenizer directory (default: llama2tokenizer)")

    # ── Dataset ──
    parser.add_argument("--hf_dataset", default=False, action="store_true",
                        help="Use HuggingFace online Pile dataset instead of local files")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to local Pile dataset (required when --hf_dataset is not set). "
                             "Expects train/*.jsonl.zst and val.jsonl.zst under this directory.")

    # ── Training ──
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Per-GPU batch size")
    parser.add_argument("--total_batch_size", type=int, default=None,
                        help="Total batch size across all GPUs and gradient accumulation steps")
    parser.add_argument("--gradient_accumulation", type=int, default=None,
                        help="Number of gradient accumulation steps (auto-computed from "
                             "total_batch_size if not provided)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of optimizer update steps to train for")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number,
                        default=None,
                        help="Alternative to num_training_steps: total tokens to train on "
                             "(supports M/B suffixes, e.g. '1B'). Overwrites num_training_steps.")
    parser.add_argument("--dtype", type=str,
                        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_gpu", default=False, action="store_true",
                        help="Disable DDP and run on a single GPU")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of DataLoader worker processes")
    parser.add_argument("--activation_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")

    # ── Optimizer ──
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "rl_adamw"],
                        help="Optimizer: 'adamw' or 'rl_adamw'")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clipping", type=float, default=1.0,
                        help="Gradient clipping norm (0.0 to disable)")

    # ── Scheduler ──
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_restarts", "cosine_quick_recovery"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--cycle_length", type=int, default=None,
                        help="Cycle length for cosine scheduler (defaults to num_training_steps)")
    parser.add_argument("--recovery_steps", type=int, default=10)

    # ── Checkpointing ──
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Path to a checkpoint directory to resume training from")
    parser.add_argument("--keep_only_last_model", default=False, action="store_true")

    # ── Evaluation ──
    parser.add_argument("--eval_every", type=int, default=5_000)

    # ── Logging ──
    parser.add_argument("--wandb_project_name", type=str, default="qwen2moe-pretraining")
    parser.add_argument("--model_name", type=str, default="qwen2moe_run",
                        help="Run name shown in wandb")
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated tags for wandb")

    # ── RL Optimizer specific ──
    parser.add_argument("--rl_agent_lr", type=float, default=3e-4)
    parser.add_argument("--rl_history_len", type=int, default=5)
    parser.add_argument("--rl_stats_window", type=int, default=20)
    parser.add_argument("--rl_agent_save_dir", type=str, default="rl_checkpoints")
    parser.add_argument("--rl_agent_load_path", type=str, default=None)
    parser.add_argument("--rl_epoch", type=int, default=1)
    parser.add_argument("--rl_mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--rl_round", type=int, default=1)

    args = parser.parse_args(args)

    # ── Post-processing / validation ──
    if not args.hf_dataset and args.dataset_path is None:
        parser.error("Either --hf_dataset or --dataset_path must be provided.")

    if args.save_dir is None:
        from datetime import datetime
        args.save_dir = f"checkpoints/qwen2moe-{args.moe_config}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, \
        "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Computed num_training_steps = {args.num_training_steps} from max_train_tokens")

    if args.continue_from is not None:
        assert os.path.exists(args.continue_from), \
            f"--continue_from={args.continue_from} does not exist"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported. Use bfloat16 or float32.")

    if args.cycle_length is None:
        args.cycle_length = args.num_training_steps

    return args


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap_model(model_obj):
    """Return the underlying model, stripping DDP if necessary."""
    if isinstance(model_obj, torch.nn.parallel.DistributedDataParallel):
        return model_obj.module
    return model_obj


def _build_moe_config(args):
    """Construct a Qwen2MoeConfig from CLI args.

    IMPORTANT: We set router_aux_loss_coef=0.0 in the HF config to DISABLE HF's
    internal load_balancing_loss_func.  That function uses dist.get_rank() as a
    column index into tokens_per_expert, which goes out-of-bounds on GPUs whose
    rank >= num_experts_per_tok (e.g. rank 5 with top-k=4).  Instead we compute
    aux_loss manually from the returned router_logits (see _compute_aux_loss).
    """
    assert args.moe_config in MOE_CONFIGS, \
        f"Unknown moe_config '{args.moe_config}'. Choices: {list(MOE_CONFIGS.keys())}"

    base = MOE_CONFIGS[args.moe_config].copy()
    base.update(
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        output_router_logits=True,   # must be True to get router_logits
        router_aux_loss_coef=0.0,    # disable HF's broken multi-GPU aux_loss; we compute it manually
    )
    config = Qwen2MoeConfig(**base)

    # Verify critical fields were accepted
    assert getattr(config, "num_experts", None) == args.num_experts, \
        f"num_experts mismatch: config={getattr(config, 'num_experts', None)}"
    assert getattr(config, "num_experts_per_tok", None) == args.num_experts_per_tok, \
        (f"num_experts_per_tok mismatch: config={getattr(config, 'num_experts_per_tok', None)}. "
         f"Check transformers version — some use 'num_experts_per_token' instead.")
    assert getattr(config, "output_router_logits", None) is True, \
        f"output_router_logits not set to True in config"

    return config


def _compute_z_loss(router_logits):
    """
    Router z-loss across all MoE layers.
    router_logits: tuple/list of (batch_size * seq_len, num_experts) tensors.
    """
    z_loss = torch.tensor(0.0, device=router_logits[0].device)
    for layer_logits in router_logits:
        z_loss = z_loss + torch.logsumexp(layer_logits, dim=-1).pow(2).mean()
    return z_loss / len(router_logits)


def _compute_aux_loss(router_logits, num_experts, num_experts_per_tok):
    """
    Manual load-balancing auxiliary loss (replaces HF's load_balancing_loss_func).

    HF's built-in version uses dist.get_rank() as a column offset into tokens_per_expert,
    which goes out-of-bounds on GPUs whose rank >= num_experts_per_tok.  We compute the
    same quantity without any distributed rank indexing.

    Formula (identical to Switch Transformer / Mixtral aux loss):
        aux_loss = num_experts * sum_over_experts(f_i * P_i)
    where:
        f_i  = fraction of tokens dispatched to expert i
        P_i  = mean router probability for expert i

    router_logits: tuple/list of (T, num_experts) tensors, T = batch*seq tokens.
    Returns a scalar tensor.
    """
    if not router_logits:
        return torch.tensor(0.0)

    device = router_logits[0].device
    aux_loss = torch.tensor(0.0, device=device)

    for layer_logits in router_logits:
        # layer_logits: (T, num_experts)
        routing_weights = torch.softmax(layer_logits.float(), dim=-1)  # (T, E)

        # Top-k expert indices
        _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)  # (T, k)

        # One-hot dispatch mask: (T, E)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=num_experts
        ).float()                          # (T, k, E)
        expert_mask = expert_mask.sum(dim=1)   # (T, E): 1 if token goes to that expert

        # f_i: fraction of tokens routed to each expert, shape (E,)
        tokens_per_expert = expert_mask.mean(dim=0)

        # P_i: mean router probability for each expert, shape (E,)
        router_prob_per_expert = routing_weights.mean(dim=0)

        aux_loss = aux_loss + num_experts * (tokens_per_expert * router_prob_per_expert).sum()

    return aux_loss / len(router_logits)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size,
                   device, batch_size, single_gpu=False, data_file=None, hf_dataset=False):
    _time = time.time()

    if hf_dataset:
        val_data = datasets.load_dataset(
            "monology/pile-uncopyrighted", split="validation", streaming=True
        )
    else:
        val_data = datasets.load_dataset(
            "json",
            data_files={"validation": f"{data_file}/val.jsonl.zst"},
            split="validation",
            streaming=True,
        )

    val_data = val_data.shuffle(seed=42, buffer_size=10_000)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f}s")

    if not single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(
            val_data, rank=global_rank, world_size=world_size
        )

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "meta"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1

    logger.info(f"Eval set prepared in {time.time() - _time:.2f}s")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100

        # Use output_router_logits=False during eval to avoid aux_loss overhead
        loss = model(**batch, labels=labels, output_router_logits=False).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum(t.item() for t in gathered_losses) / world_size

    return total_loss, evaluated_on_tokens


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ["RANK"])
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # Pass local_rank to args so RL optimizer can use it
    args.local_rank = local_rank

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, "
                f"device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    # ── Gradient accumulation ──
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, \
                "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, \
                "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must equal total_batch_size"

    # ── Logging ──
    if global_rank != 0:
        logger.remove()

    if global_rank == 0:
        wandb.init(project=args.wandb_project_name, name=args.model_name, tags=args.tags)
        logger.info("*" * 40)
        logger.info("Starting MoE training with arguments:")
        for k, v in vars(args).items():
            logger.info(f"  {k:40s} {v}")
        logger.info("*" * 40)

    # ── Dataset ──
    if args.hf_dataset:
        logger.info("Using HuggingFace online Pile dataset (monology/pile-uncopyrighted)")
        data = datasets.load_dataset(
            "monology/pile-uncopyrighted", split="train", streaming=True
        )
    else:
        logger.info(f"Using local Pile dataset from {args.dataset_path}")
        data = datasets.load_dataset(
            "json",
            data_files=f"{args.dataset_path}/train/*.jsonl.zst",
            split="train",
            streaming=True,
        )

    seed_for_shuffle = 42
    logger.info(f"Shuffling dataset with seed {seed_for_shuffle}")
    data = data.shuffle(seed=seed_for_shuffle, buffer_size=10_000)

    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size
        )

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, model_max_length=args.max_length
    )
    tokenizer.pad_token_id = _PAD_TOKEN_ID
    pad_idx = _PAD_TOKEN_ID

    def preprocess_batched(batch):
        return tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    # ── DataLoader ──
    dataset = PreprocessedIterableDataset(
        data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=args.workers
    )

    # ── Model ──
    model_config = _build_moe_config(args)
    logger.info(f"Building Qwen2Moe model: {args.moe_config} "
                f"(experts={args.num_experts}, top-k={args.num_experts_per_tok})")
    model = AutoModelForCausalLM.from_config(model_config)

    n_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {n_total_params / 1_000_000:.2f}M")

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled")

    # ── Training state (must be set before loading checkpoint) ──
    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    # ── Move model to device ──
    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    # ── Optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
    elif args.optimizer.lower() == "rl_adamw":
        optimizer = RL_AdamW_Wrapper(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_lr=args.lr,
            min_lr=args.lr * args.min_lr_ratio,
            warmup_steps=args.warmup_steps,
            max_steps=args.num_training_steps,
            agent_load_path=args.rl_agent_load_path,
            agent_save_dir=args.rl_agent_save_dir,
            run_id=args.model_name,
            rank=dist.get_rank() if dist.is_initialized() else 0,
            mode=args.rl_mode,
            action_scale=args.action_scale,
            device=device,
            history_len=args.rl_history_len,
            stats_window=args.rl_stats_window,
            rl_agent_lr=args.rl_agent_lr,
            round=args.rl_round,
        )
    else:
        raise ValueError(f"Optimizer '{args.optimizer}' not supported. "
                         "Choose 'adamw' or 'rl_adamw'.")

    is_rl_optimizer = isinstance(optimizer, RL_AdamW_Wrapper)
    if is_rl_optimizer:
        logger.info("RL optimizer detected — external scheduler will be DISABLED.")

    # ── Scheduler (AdamW only) ──
    if not is_rl_optimizer:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            cycle_length=args.cycle_length,
            recovery_steps=args.recovery_steps,
        )

    # ── Load checkpoint ──
    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Resuming from checkpoint: {args.continue_from}")

        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            # Try safetensors
            from safetensors.torch import load_file
            st_path = os.path.join(args.continue_from, "model.safetensors")
            assert os.path.exists(st_path), \
                f"Neither pytorch_model.bin nor model.safetensors found in {args.continue_from}"
            state_dict = load_file(st_path)
            torch.save(state_dict, checkpoint_path)
            logger.info(f"Converted safetensors -> {checkpoint_path}")

        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu"), strict=True
        )
        logger.info("Model weights loaded (strict=True)")

        opt_ckpt_path = os.path.join(args.continue_from, "optimizer.pt")
        if os.path.exists(opt_ckpt_path):
            opt_ckpt = torch.load(opt_ckpt_path, map_location="cpu")
            optimizer.load_state_dict(opt_ckpt["optimizer"])
            if not is_rl_optimizer:
                scheduler.load_state_dict(opt_ckpt["scheduler"])
            logger.info("Optimizer (and scheduler) state loaded")
        else:
            logger.warning(f"No optimizer checkpoint found at {opt_ckpt_path}, starting fresh")

        training_state_path = os.path.join(args.continue_from, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path) as f:
                _old = json.load(f)
            global_step        = _old["global_step"]
            update_step        = _old["update_step"]
            tokens_seen        = _old["tokens_seen"]
            tokens_seen_before = _old["tokens_seen_before"]
            logger.info(f"Restored training state: global_step={global_step}, "
                        f"update_step={update_step}, tokens_seen={tokens_seen}")
        else:
            logger.warning("No training_state.json found — step counters start from 0")

        logger.info("*" * 40)

    # ── DDP ──
    if not args.single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,  # Required for MoE: some experts may have no grad
        )

    # ── wandb config ──
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),
        "total_params_M": n_total_params / 1_000_000,
        "model_config": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })
    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")
        pbar = tqdm(total=args.num_training_steps - update_step,
                    desc="Update steps", ncols=80, mininterval=10.0)

    logger.info(f"\n{_unwrap_model(model)}\n")
    logger.info(f"Trainable params: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"Saving to {args.save_dir} every {args.save_every} update steps")

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoint save helper
    # ──────────────────────────────────────────────────────────────────────────

    def save_checkpoint(save_path, update_step, global_step, tokens_seen, tokens_seen_before,
                        update_time):
        os.makedirs(save_path, exist_ok=True)

        # Save model weights in HuggingFace format (compatible with from_pretrained)
        _unwrap_model(model).save_pretrained(save_path, max_shard_size="100GB")

        # Save optimizer (and scheduler if AdamW)
        opt_state = {"optimizer": optimizer.state_dict(), "update_step": update_step,
                     "global_step": global_step, "config": run_config, "dtype": args.dtype}
        if not is_rl_optimizer:
            opt_state["scheduler"] = scheduler.state_dict()
        if global_rank == 0:
            opt_state["wandb"] = wandb.run.dir
        torch.save(opt_state, os.path.join(save_path, "optimizer.pt"))

        # Save RL agent if applicable
        if is_rl_optimizer and global_rank == 0:
            optimizer.save_agent(os.path.join(save_path, "rl_agent.pth"))

        # Save training state
        training_state = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(os.path.join(save_path, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=4)

        # Save wandb run ID for resuming
        if global_rank == 0:
            wandb_info = {"wandb_id": wandb.run.id}
            with open(os.path.join(args.save_dir, "wandb.json"), "w") as f:
                json.dump(wandb_info, f, indent=4)

    # ──────────────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────────────

    update_time = time.time()
    local_step = 0
    grad_accumulation = args.gradient_accumulation

    max_memory = torch.cuda.max_memory_allocated()
    if global_rank == 0:
        logger.info(f"Peak memory before training loop: {max_memory / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

    for batch_idx, batch in enumerate(dataloader):

        if update_step > args.num_training_steps:
            logger.info(f"Reached max update steps ({args.num_training_steps}). Stopping.")
            break

        global_step += 1
        local_step  += 1

        batch = {k: v.to(device) for k, v in batch.items()}

        # Build labels: mask padding tokens with -100
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        # ── Forward ──
        # router_aux_loss_coef=0.0 in config, so outputs.loss == lm_loss only.
        # We compute aux_loss and z_loss manually from router_logits.
        outputs = model(**batch, labels=labels, output_router_logits=True)

        lm_loss = outputs.loss

        aux_loss = torch.tensor(0.0, device=device)
        z_loss   = torch.tensor(0.0, device=device)
        if outputs.router_logits is not None and len(outputs.router_logits) > 0:
            aux_loss = _compute_aux_loss(
                outputs.router_logits, args.num_experts, args.num_experts_per_tok
            )
            z_loss = _compute_z_loss(outputs.router_logits)

        total_loss = (lm_loss
                      + args.router_aux_loss_coef * aux_loss
                      + args.router_z_loss_coef   * z_loss)

        scaled_loss = total_loss / grad_accumulation
        scaled_loss.backward()

        # Detach metrics for logging (before graph is freed)
        aux_loss_val  = aux_loss.detach().item()
        z_loss_val    = z_loss.detach().item()
        reported_loss = lm_loss.detach().item()

        if global_step % grad_accumulation != 0:
            continue

        # ──────────────────────────────────────────────────────────────────────
        # Update step
        # ──────────────────────────────────────────────────────────────────────

        # Grad clipping
        if args.grad_clipping != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        grad_norm = sum(
            torch.norm(p.grad.clone().detach().cpu())
            for p in model.parameters() if p.grad is not None
        )

        if global_rank == 0:
            pbar.update(1)

        if is_rl_optimizer:
            # All-reduce total_loss across ranks so the RL agent sees the same reward signal
            rl_loss = total_loss.detach().clone()
            dist.all_reduce(rl_loss, op=dist.ReduceOp.AVG)
            _, abort_training = optimizer.step(loss=rl_loss)

            if abort_training:
                if global_rank == 0:
                    logger.error(f"[RL] Loss spike detected at step {update_step}. "
                                 "Aborting training.")
                    os.makedirs(args.rl_agent_save_dir, exist_ok=True)
                    crash_path = os.path.join(
                        args.rl_agent_save_dir,
                        f"agent_CRASH_step{update_step}.pth"
                    )
                    optimizer.save_agent(crash_path)
                    logger.warning(f"Emergency RL agent saved to {crash_path}")
                time.sleep(5)
                sys.exit(1)
        else:
            optimizer.step()
            scheduler.step()

        optimizer.zero_grad()
        update_step  += 1
        update_time   = time.time() - update_time

        # ── Checkpoint ──
        if (local_step > grad_accumulation
                and update_step % args.save_every == 0
                and global_rank == 0):

            if args.keep_only_last_model:
                ckpt_dir = os.path.join(args.save_dir, "model_last")
            else:
                ckpt_dir = os.path.join(args.save_dir, f"model_{update_step}")

            logger.info(f"Saving checkpoint to {ckpt_dir} at update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)

            save_checkpoint(
                ckpt_dir, update_step, global_step,
                tokens_seen, tokens_seen_before, update_time,
            )

            # Also save periodic RL agent checkpoint separately
            if is_rl_optimizer and global_rank == 0:
                agent_path = os.path.join(
                    args.rl_agent_save_dir,
                    f"agent_epoch{args.rl_epoch}_step{update_step}.pth"
                )
                os.makedirs(args.rl_agent_save_dir, exist_ok=True)
                optimizer.save_agent(agent_path)
                optimizer.save_agent(
                    os.path.join(args.rl_agent_save_dir,
                                 f"agent_epoch{args.rl_epoch}_final.pth")
                )

        # ── Evaluation ──
        if update_step % args.eval_every == 0:
            logger.info(f"Running evaluation at update step {update_step}")
            eval_loss, eval_tokens = evaluate_model(
                model=model,
                preprocess_batched=preprocess_batched,
                pad_idx=pad_idx,
                global_rank=global_rank,
                world_size=world_size,
                device=device,
                batch_size=args.batch_size,
                single_gpu=args.single_gpu,
                data_file=args.dataset_path,
                hf_dataset=args.hf_dataset,
            )
            if global_rank == 0:
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/perplexity": np.exp(eval_loss),
                    "eval/tokens": eval_tokens,
                }, step=update_step)
            logger.info(f"Eval loss={eval_loss:.4f} "
                        f"perplexity={np.exp(eval_loss):.2f} at step {update_step}")

        # ── Logging ──
        if not is_rl_optimizer:
            lr = optimizer.param_groups[0]["lr"]
        else:
            # RL optimizer manages LR internally; report average across param groups
            lr = float(np.mean([g["lr"] for g in optimizer.param_groups]))

        tokens_in_update   = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update  = grad_accumulation * world_size
        max_memory         = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        if global_rank == 0:
            log_dict = {
                "loss": reported_loss,
                "train/aux_loss": aux_loss_val,
                "train/z_loss": z_loss_val,
                "train/total_loss": total_loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                "gradnorm": grad_norm,
                "max_memory_GB": max_memory / 1e9,
            }
            wandb.log(log_dict, step=update_step)

        update_time = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # End of training loop
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("Training finished")
    if global_rank == 0:
        pbar.close()

    # Save final checkpoint if not already saved
    final_ckpt_dir = os.path.join(args.save_dir, f"model_{update_step}")
    if global_rank == 0 and not os.path.exists(final_ckpt_dir):
        logger.info(f"Saving final checkpoint to {final_ckpt_dir}")
        os.makedirs(args.save_dir, exist_ok=True)
        save_checkpoint(
            final_ckpt_dir, update_step, global_step,
            tokens_seen, tokens_seen_before, update_time,
        )

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    if not is_rl_optimizer:
        del optimizer, scheduler
    else:
        del optimizer
    import gc; gc.collect()
    torch.cuda.empty_cache()

    final_loss, final_tokens = evaluate_model(
        model=model,
        preprocess_batched=preprocess_batched,
        pad_idx=pad_idx,
        global_rank=global_rank,
        world_size=world_size,
        device=device,
        batch_size=args.batch_size,
        single_gpu=args.single_gpu,
        data_file=args.dataset_path,
        hf_dataset=args.hf_dataset,
    )

    if global_rank == 0:
        wandb.log({
            "final_eval/loss": final_loss,
            "final_eval/perplexity": np.exp(final_loss),
            "final_eval/tokens": final_tokens,
        }, step=update_step)
        logger.info(f"Final eval loss={final_loss:.4f} perplexity={np.exp(final_loss):.2f}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting Qwen2Moe pretraining script")
    args = parse_args(None)
    main(args)
