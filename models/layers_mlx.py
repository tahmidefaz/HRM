from typing import Tuple, Optional
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.base import Module


CosSin = Tuple[mx.array, mx.array]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: mx.array):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    seq_len = q.shape[1]
    cos = cos[:seq_len, :]  # Truncate to sequence length
    sin = sin[:seq_len, :]  # Truncate to sequence length
    
    q_embed = (q * cos[None, :, None, :]) + (rotate_half(q) * sin[None, :, None, :])
    k_embed = (k * cos[None, :, None, :]) + (rotate_half(k) * sin[None, :, None, :])
    return q_embed, k_embed


class CastedLinear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        scale = math.sqrt(1.0 / in_features)
        self.weight = mx.random.normal((out_features, in_features)) * scale
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output


class CastedEmbedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float):
        super().__init__()
        self.embedding_weight = mx.random.normal((num_embeddings, embedding_dim)) * init_std
        
    def __call__(self, indices: mx.array) -> mx.array:
        return self.embedding_weight[indices]


class RotaryEmbedding(Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        
        # RoPE
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        t = mx.arange(max_position_embeddings, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)

    def __call__(self):
        return self.cos_cached, self.sin_cached


class Attention(Module):
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_key_value_heads: int, causal: bool = False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.scale = 1.0 / math.sqrt(head_dim)

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array, cos_sin: Optional[CosSin] = None) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, hidden_size]
        qkv = self.qkv_proj(hidden_states)
        
        # Split head
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Multi-head attention using MLX's scaled_dot_product_attention
        # Reshape for attention: [batch, num_heads, seq_len, head_dim]
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)  
        value = value.transpose(0, 2, 1, 3)
        
        # Expand key and value if using grouped query attention
        if self.num_key_value_heads < self.num_heads:
            key = mx.repeat(key, self.num_heads // self.num_key_value_heads, axis=1)
            value = mx.repeat(value, self.num_heads // self.num_key_value_heads, axis=1)
        
        # Attention computation
        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=None
        )
        
        # Reshape back: [batch_size, seq_len, output_size]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.output_size)
        
        return self.o_proj(attn_output)


class SwiGLU(Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up_proj(x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


def rms_norm(hidden_states: mx.array, variance_epsilon: float) -> mx.array:
    variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * mx.rsqrt(variance + variance_epsilon)
    return hidden_states