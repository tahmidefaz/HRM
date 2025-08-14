from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.base import Module
from pydantic import BaseModel

from models.layers_mlx import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding_mlx import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: mx.array
    z_L: mx.array


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: mx.array
    halted: mx.array
    
    current_data: Dict[str, mx.array]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float


class HierarchicalReasoningModel_ACTV1Block(Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def __call__(self, hidden_states: mx.array, cos_sin: Optional[CosSin] = None) -> mx.array:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(hidden_states, cos_sin=cos_sin), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = layers

    def __call__(self, hidden_states: mx.array, input_injection: mx.array, **kwargs) -> mx.array:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers, 
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size, 
                init_std=0.0
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len, 
                self.config.hidden_size, 
                init_std=embed_init_std
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)]
        )
        
        # Initial states - using random normal initialization
        self.H_init = mx.random.normal((self.config.hidden_size,))
        self.L_init = mx.random.normal((self.config.hidden_size,))

        # Q head special init - init Q to (almost) zero for faster learning during bootstrapping
        self.q_head.weight = mx.zeros(self.q_head.weight.shape)
        self.q_head.bias = mx.full(self.q_head.bias.shape, -5.0)

    def _input_embeddings(self, input: mx.array, puzzle_identifiers: mx.array):
        # Token embedding
        embedding = self.embed_tokens(input.astype(mx.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = mx.pad(puzzle_embedding, [(0, 0), (0, pad_count)])

            puzzle_embedding = puzzle_embedding.reshape(-1, self.puzzle_emb_len, self.config.hidden_size)
            embedding = mx.concatenate([puzzle_embedding, embedding], axis=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight)

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=mx.zeros((batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)),
            z_L=mx.zeros((batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)),
        )
        
    def reset_carry(self, reset_flag: mx.array, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        reset_expanded = reset_flag.reshape(-1, 1, 1)
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=mx.where(reset_expanded, self.H_init, carry.z_H),
            z_L=mx.where(reset_expanded, self.L_init, carry.z_L),
        )

    def __call__(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, mx.array]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, mx.array, Tuple[mx.array, mx.array]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations (no gradients for efficiency)
        z_H, z_L = carry.z_H, carry.z_L

        for _H_step in range(self.config.H_cycles):
            for _L_step in range(self.config.L_cycles):
                if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

            if not (_H_step == self.config.H_cycles - 1):
                z_H = self.H_level(z_H, z_L, **seq_info)

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=mx.stop_gradient(z_H), z_L=mx.stop_gradient(z_L))
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0])
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(Module):
    """ACT wrapper for MLX."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, mx.array]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            steps=mx.zeros((batch_size,), dtype=mx.int32),
            halted=mx.ones((batch_size,), dtype=mx.bool_),  # Default to halted
            
            current_data={k: mx.zeros_like(v) for k, v in batch.items()}
        )
        
    def __call__(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, mx.array], training: bool = True) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, mx.array]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = mx.where(carry.halted, 0, carry.steps)

        # Update current data for halted sequences
        new_current_data = {}
        for k, v in carry.current_data.items():
            batch_val = batch[k]
            halted_expanded = carry.halted.reshape((-1,) + (1,) * (batch_val.ndim - 1))
            new_current_data[k] = mx.where(halted_expanded, batch_val, v)

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # Step
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps
        
        halted = is_last_step

        # if training, and ACT is enabled
        if training and (self.config.halt_max_steps > 1):
            # Halt signal
            halted = halted | (q_halt_logits > q_continue_logits)

            # Exploration
            rand_vals = mx.random.uniform(shape=q_halt_logits.shape)
            exploration_mask = rand_vals < self.config.halt_exploration_prob
            min_halt_steps = exploration_mask * mx.random.randint(2, self.config.halt_max_steps + 1, shape=new_steps.shape)

            halted = halted & (new_steps >= min_halt_steps)

            # Compute target Q
            next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
            
            target_q_continue = mx.sigmoid(mx.where(is_last_step, next_q_halt_logits, mx.maximum(next_q_halt_logits, next_q_continue_logits)))
            outputs["target_q_continue"] = target_q_continue

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs