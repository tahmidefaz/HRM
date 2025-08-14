from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.layers.base import Module

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import SparseEmbeddingSignSGD


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []


@dataclass
class TrainState:
    model: Module
    optimizers: Sequence[Any]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, **kwargs):
    """Create MLX-compatible data loader"""
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        **kwargs
    ), split=split)
    
    # MLX doesn't have DataLoader, so we'll use the dataset directly
    return dataset, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size,

        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    model: Module = model_cls(model_cfg)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

    # Optimizers and lr
    # Split parameters into puzzle embedding and other parameters
    puzzle_emb_params = []
    other_params = []
    
    def collect_params(module, path=""):
        params = []
        if hasattr(module, 'puzzle_emb'):
            # This is the sparse embedding, handle separately
            puzzle_emb_params.extend([module.puzzle_emb.weights])
        else:
            # Collect all parameters from this module
            for name, param in module.parameters().items():
                if isinstance(param, mx.array):
                    params.append(param)
        return params
    
    # Get all non-puzzle-embedding parameters
    other_params = [p for p in model.parameters().values() if isinstance(p, mx.array)]
    
    optimizers = [
        SparseEmbeddingSignSGD(
            lr=0,  # Will be set by scheduler
            weight_decay=config.puzzle_emb_weight_decay,
        ),
        optim.AdamW(
            learning_rate=0,  # Will be set by scheduler
            weight_decay=config.weight_decay,
            betas=[config.beta1, config.beta2]
        )
    ]
    
    optimizer_lrs = [
        config.puzzle_emb_lr,
        config.lr
    ]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(config.checkpoint_path, f"step_{train_state.step}.npz")
    mx.save_weights(model_path, train_state.model.parameters())


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def value_and_grad_fn(model, batch, global_batch_size, return_keys):
    """Value and gradient function for MLX training"""
    def forward_fn(params):
        # Update model with current parameters
        model.update(params)
        
        # Initialize carry if needed
        carry = model.initial_carry(batch)
        
        # Forward pass
        new_carry, loss, metrics, outputs, all_finish = model(
            carry=carry, 
            batch=batch, 
            return_keys=return_keys,
            training=True
        )
        
        # Scale loss by batch size
        scaled_loss = loss / global_batch_size
        
        return scaled_loss, (new_carry, metrics, outputs, all_finish)
    
    return mx.value_and_grad(forward_fn)


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    # Convert batch to MLX arrays
    batch = {k: mx.array(v) for k, v in batch.items()}

    # Get value and gradient function
    vg_fn = value_and_grad_fn(train_state.model, batch, global_batch_size, [])
    
    # Forward and backward
    loss_and_aux, grads = vg_fn(train_state.model.parameters())
    loss, (new_carry, metrics, _, _) = loss_and_aux
    
    # Update carry
    train_state.carry = new_carry

    # Apply optimizers
    lr_this_step = None
    
    # Update non-sparse embedding parameters with AdamW
    if len(train_state.optimizers) > 1:
        lr_this_step = compute_lr(train_state.optimizer_lrs[1], config, train_state)
        train_state.optimizers[1].learning_rate = lr_this_step
        
        # Filter gradients for non-sparse parameters
        filtered_grads = {k: v for k, v in grads.items() if 'puzzle_emb' not in k}
        filtered_params = {k: v for k, v in train_state.model.parameters().items() if 'puzzle_emb' not in k}
        
        train_state.optimizers[1].update(filtered_params, filtered_grads)
        train_state.model.update(filtered_params)

    # Update sparse embedding with custom optimizer
    if hasattr(train_state.model.model, 'puzzle_emb') and 'puzzle_emb.weights' in grads:
        lr_sparse = compute_lr(train_state.optimizer_lrs[0], config, train_state)
        train_state.optimizers[0].lr = lr_sparse
        
        # Update sparse embedding
        puzzle_emb_grad = grads['puzzle_emb.weights']
        puzzle_emb_indices = batch['puzzle_identifiers']  # Assuming this exists
        train_state.optimizers[0].update(train_state.model.model.puzzle_emb, puzzle_emb_grad, puzzle_emb_indices)

    # Process metrics
    if len(metrics):
        reduced_metrics = {}
        count = max(float(metrics["count"]), 1.0)  # Avoid NaNs
        for k, v in metrics.items():
            if k.endswith("loss"):
                reduced_metrics[f"train/{k}"] = float(v) / global_batch_size
            else:
                reduced_metrics[f"train/{k}"] = float(v) / count

        reduced_metrics["train/lr"] = lr_this_step
        return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader, eval_metadata):
    """Evaluation function for MLX"""
    with mx.eval_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}
        
        metric_keys = []
        metric_values = None
        metric_counts = [0 for _ in range(len(set_ids))]
        
        for set_name, batch, global_batch_size in eval_loader:
            # Convert to MLX arrays
            batch = {k: mx.array(v) for k, v in batch.items()}
            
            # Initialize carry
            carry = train_state.model.initial_carry(batch)

            # Forward until completion
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(
                    carry=carry, 
                    batch=batch, 
                    return_keys=config.eval_save_outputs,
                    training=False
                )
                
                if all_finish:
                    break

            # Collect predictions if needed
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v)
                        
            # Aggregate metrics
            set_id = set_ids[set_name]
            
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = mx.zeros((len(set_ids), len(metrics)))
                
            metric_array = mx.array([float(metrics[k]) for k in metric_keys])
            metric_values = metric_values.at[set_id].add(metric_array)
            metric_counts[set_id] += global_batch_size

        # Save predictions if needed
        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: mx.concatenate(v, axis=0) for k, v in all_preds.items()}
            
            os.makedirs(config.checkpoint_path, exist_ok=True)
            pred_path = os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.npz")
            mx.savez(pred_path, **all_preds)

        # Process metrics
        if metric_values is not None:
            reduced_metrics = {}
            for set_id, set_name in enumerate(set_ids):
                set_metrics = {}
                for metric_id, metric_name in enumerate(metric_keys):
                    if metric_name == "count":
                        continue
                    set_metrics[metric_name] = float(metric_values[set_id, metric_id]) / metric_counts[set_id]
                reduced_metrics[set_name] = set_metrics

            return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    # Load config
    config = PretrainConfig(**hydra_config)  # type: ignore

    # Naming
    if config.project_name is None:
        config.project_name = f"{os.path.basename(config.data_path).capitalize()} HRM-MLX"
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

    # Seed RNGs
    mx.random.seed(config.seed)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size)
    eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size)

    # Train state
    train_state = init_train_state(config, train_metadata)

    # Progress bar and logger
    progress_bar = tqdm.tqdm(total=train_state.total_steps)

    wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))
    
    # Count parameters
    num_params = sum(p.size for p in train_state.model.parameters().values() if isinstance(p, mx.array))
    wandb.log({"num_params": num_params}, step=0)
    save_code_and_config(config)

    # Training Loop
    for _iter_id in range(total_iters):
        print(f"Epoch {_iter_id * train_epochs_per_iter}")

        # Train Iter
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size)

            if metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)

        # Evaluation
        metrics = evaluate(config, train_state, eval_loader, eval_metadata)

        if metrics is not None:
            wandb.log(metrics, step=train_state.step)
            
        # Checkpointing
        if config.checkpoint_every_eval or (_iter_id == total_iters - 1):
            save_train_state(config, train_state)

    wandb.finish()


if __name__ == "__main__":
    launch()