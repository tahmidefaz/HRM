from typing import Any, Tuple, Dict, Sequence, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.base import Module


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return mx.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1
    )


def log_stablemax(x, axis=-1):
    s_x = s(x)
    return mx.log(s_x / mx.sum(s_x, axis=axis, keepdims=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.astype(mx.float32), axis=-1)

    valid_mask = labels != ignore_index
    transformed_labels = mx.where(valid_mask, labels, 0)
    prediction_logprobs = mx.take_along_axis(logprobs, mx.expand_dims(transformed_labels.astype(mx.int32), axis=-1), axis=-1).squeeze(-1)

    return -mx.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32 and flatten
    flat_logits = logits.astype(mx.float32).reshape(-1, logits.shape[-1])
    flat_labels = labels.astype(mx.int32).reshape(-1)
    
    # Create mask for valid labels
    valid_mask = flat_labels != ignore_index
    
    # Compute cross entropy only for valid positions
    log_probs = mx.log_softmax(flat_logits, axis=-1)
    losses = -mx.take_along_axis(log_probs, mx.expand_dims(mx.where(valid_mask, flat_labels, 0), axis=-1), axis=-1).squeeze(-1)
    
    # Set loss to 0 for ignored positions
    losses = mx.where(valid_mask, losses, 0)
    
    return losses.reshape(labels.shape)


class ACTLossHead(Module):
    def __init__(self, model: Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        if loss_type == "stablemax_cross_entropy":
            self.loss_fn = stablemax_cross_entropy
        elif loss_type == "softmax_cross_entropy":
            self.loss_fn = softmax_cross_entropy
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def __call__(
        self,
        return_keys: Sequence[str],
        training: bool = True,
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, mx.array, Dict[str, mx.array], Optional[Dict[str, mx.array]], mx.array]:
        # Model logits
        new_carry, outputs = self.model(**model_kwargs, training=training)
        labels = new_carry.current_data["labels"]

        # Correctness
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mx.sum(mask, axis=-1)
        loss_divisor = mx.maximum(loss_counts, 1).reshape(-1, 1)  # Avoid NaNs in division

        is_correct = mask & (mx.argmax(outputs["logits"], axis=-1) == labels)
        seq_is_correct = mx.sum(is_correct, axis=-1) == loss_counts
        
        # Metrics (halted)
        valid_metrics = new_carry.halted & (loss_counts > 0)
        metrics = {
            "count": mx.sum(valid_metrics),
            
            "accuracy": mx.sum(mx.where(valid_metrics, mx.sum(is_correct.astype(mx.float32) / loss_divisor, axis=-1), 0)),
            "exact_accuracy": mx.sum(valid_metrics & seq_is_correct),

            "q_halt_accuracy": mx.sum(valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)),
            "steps": mx.sum(mx.where(valid_metrics, new_carry.steps, 0)),
        }

        # Losses
        lm_loss = mx.sum(self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor)
        
        # Binary cross entropy with logits for q_halt_loss
        q_halt_logits = outputs["q_halt_logits"]
        seq_is_correct_float = seq_is_correct.astype(q_halt_logits.dtype)
        # Manual log_sigmoid implementation since mx.log_sigmoid may not be available
        log_sigmoid_pos = -mx.logaddexp(0.0, -q_halt_logits)
        log_sigmoid_neg = -mx.logaddexp(0.0, q_halt_logits)
        q_halt_loss = mx.sum(-seq_is_correct_float * log_sigmoid_pos - (1 - seq_is_correct_float) * log_sigmoid_neg)

        metrics.update({
            "lm_loss": mx.stop_gradient(lm_loss),
            "q_halt_loss": mx.stop_gradient(q_halt_loss),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = mx.array(0.0)
        if "target_q_continue" in outputs:
            q_continue_logits = outputs["q_continue_logits"]
            target_q_continue = outputs["target_q_continue"]
            # Manual log_sigmoid implementation since mx.log_sigmoid may not be available
            log_sigmoid_pos = -mx.logaddexp(0.0, -q_continue_logits)
            log_sigmoid_neg = -mx.logaddexp(0.0, q_continue_logits)
            q_continue_loss = mx.sum(-target_q_continue * log_sigmoid_pos - (1 - target_q_continue) * log_sigmoid_neg)

            metrics["q_continue_loss"] = mx.stop_gradient(q_continue_loss)

        # Filter outputs for return
        detached_outputs = {k: mx.stop_gradient(outputs[k]) for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, mx.all(new_carry.halted)