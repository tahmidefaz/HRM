from typing import Optional, Dict, Any
import os
import json

import mlx.core as mx
import mlx.nn as nn

import tqdm
import hydra
from omegaconf import DictConfig

from puzzle_dataset_mlx import PuzzleDataset, PuzzleDatasetConfig
from utils.functions_mlx import load_model_class_mlx
from models.losses_mlx import IGNORE_LABEL_ID


def create_model_for_eval(config_dict: dict, checkpoint_path: str):
    """Create model and load from checkpoint"""
    
    # Load model config from checkpoint
    checkpoint_config_path = os.path.join(checkpoint_path, "all_config.yaml")
    if os.path.exists(checkpoint_config_path):
        import yaml
        with open(checkpoint_config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
            # Use saved architecture config
            config_dict.update(saved_config.get('arch', {}))
    
    # Instantiate model with loss head
    model_cls = load_model_class_mlx(config_dict['arch']['name'])
    loss_head_cls = load_model_class_mlx(config_dict['arch']['loss']['name'])

    # Create model
    model_cfg = dict(
        **config_dict['arch'].get('__pydantic_extra__', {}),
        batch_size=config_dict['global_batch_size'],
        vocab_size=config_dict.get('vocab_size', 1000),  # Will be updated from metadata
        seq_len=config_dict.get('seq_len', 512),  # Will be updated from metadata
        num_puzzle_identifiers=config_dict.get('num_puzzle_identifiers', 100),  # Will be updated from metadata
        causal=False
    )

    model = model_cls(model_cfg)
    model = loss_head_cls(model, **config_dict['arch']['loss'].get('__pydantic_extra__', {}))

    # Load weights from checkpoint
    weight_files = [f for f in os.listdir(checkpoint_path) if f.startswith('step_') and f.endswith('.npz')]
    if weight_files:
        # Sort by step number and take the latest
        weight_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_weights = os.path.join(checkpoint_path, weight_files[-1])
        
        print(f"Loading weights from {latest_weights}")
        weights = mx.load(latest_weights)
        model.load_weights(weights)
    else:
        print("Warning: No checkpoint weights found")

    return model


def evaluate_model(model, eval_loader, eval_metadata, eval_save_outputs):
    """Evaluate the model on the evaluation dataset"""
    
    with mx.eval_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}
        
        metric_keys = []
        metric_values = None
        metric_counts = [0 for _ in range(len(set_ids))]
        
        progress_bar = tqdm.tqdm(desc="Evaluating")
        
        for set_name, batch, global_batch_size in eval_loader:
            # Initialize carry
            carry = model.initial_carry(batch)

            # Forward until completion
            steps = 0
            max_steps = 1000  # Safety limit
            while steps < max_steps:
                carry, _, metrics, preds, all_finish = model(
                    carry=carry, 
                    batch=batch, 
                    return_keys=eval_save_outputs,
                    training=False
                )
                
                steps += 1
                if all_finish:
                    break
            
            if steps >= max_steps:
                print(f"Warning: Reached max steps ({max_steps}) for batch in set {set_name}")

            # Collect predictions if needed
            for collection in [batch, preds]:
                for k, v in collection.items():
                    if k in eval_save_outputs:
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
            
            progress_bar.update(1)

        progress_bar.close()

        # Process and return metrics
        if metric_values is not None:
            reduced_metrics = {}
            for set_id, set_name in enumerate(set_ids):
                set_metrics = {}
                for metric_id, metric_name in enumerate(metric_keys):
                    if metric_name == "count":
                        continue
                    set_metrics[metric_name] = float(metric_values[set_id, metric_id]) / max(metric_counts[set_id], 1)
                reduced_metrics[set_name] = set_metrics

            return reduced_metrics, all_preds


def save_evaluation_results(results, preds, output_path):
    """Save evaluation results and predictions"""
    os.makedirs(output_path, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_path, "eval_metrics.json"), 'w') as f:
        # Convert mx.array values to float for JSON serialization
        json_results = {}
        for set_name, metrics in results.items():
            json_results[set_name] = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
        json.dump(json_results, f, indent=2)
    
    # Save predictions if any
    if preds:
        pred_path = os.path.join(output_path, "predictions.npz")
        pred_dict = {k: mx.concatenate(v, axis=0) for k, v in preds.items()}
        mx.savez(pred_path, **pred_dict)
        
    print(f"Evaluation results saved to {output_path}")


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(cfg: DictConfig):
    """Main evaluation function"""
    
    # Get checkpoint path from config or command line
    checkpoint_path = cfg.get('checkpoint', None)
    if checkpoint_path is None:
        raise ValueError("Please specify checkpoint path using checkpoint=<path>")
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    print(f"Evaluating checkpoint: {checkpoint_path}")
    
    # Setup evaluation dataset
    eval_config = PuzzleDatasetConfig(
        seed=cfg.get('seed', 0),
        dataset_path=cfg.data_path,
        global_batch_size=cfg.global_batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    
    eval_dataset = PuzzleDataset(eval_config, split="test")
    eval_metadata = eval_dataset.metadata
    
    # Update config with metadata
    config_dict = dict(cfg)
    config_dict.update({
        'vocab_size': eval_metadata.vocab_size,
        'seq_len': eval_metadata.seq_len,
        'num_puzzle_identifiers': eval_metadata.num_puzzle_identifiers,
    })
    
    # Create and load model
    model = create_model_for_eval(config_dict, checkpoint_path)
    
    # Run evaluation
    eval_save_outputs = cfg.get('eval_save_outputs', [])
    results, preds = evaluate_model(model, eval_dataset, eval_metadata, eval_save_outputs)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for set_name, metrics in results.items():
        print(f"\n{set_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Save results
    output_path = os.path.join(checkpoint_path, "evaluation_results")
    save_evaluation_results(results, preds, output_path)


if __name__ == "__main__":
    main()