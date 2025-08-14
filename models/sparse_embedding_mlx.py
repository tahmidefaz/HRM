from typing import Union
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.base import Module
import mlx.optimizers as optim


class CastedSparseEmbedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float = 0.0):
        super().__init__()
        
        # Real Weights - initialized with truncated normal
        if init_std > 0:
            self.weights = mx.random.normal((num_embeddings, embedding_dim)) * init_std
        else:
            self.weights = mx.zeros((num_embeddings, embedding_dim))

        # Local weights and IDs for training
        self.local_weights = mx.zeros((batch_size, embedding_dim))
        self.local_ids = mx.zeros((batch_size,), dtype=mx.int32)
        self.batch_size = batch_size

    def __call__(self, inputs: mx.array) -> mx.array:
        # In MLX, we don't have explicit training mode, so we'll simulate it
        # For now, always use weights directly - gradient handling will be different
        return self.weights[inputs]

    def update_local_state(self, inputs: mx.array):
        """Update local state for training - called externally when needed"""
        self.local_weights = self.weights[inputs]
        self.local_ids = inputs


class SparseEmbeddingSignSGD:
    """MLX version of sparse embedding optimizer using SignSGD"""
    
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-2):
        self.lr = lr
        self.weight_decay = weight_decay
    
    def update(self, sparse_emb: CastedSparseEmbedding, gradients: mx.array, indices: mx.array):
        """Update sparse embedding weights using SignSGD"""
        if gradients is None or indices is None:
            return
            
        # Get unique indices and aggregate gradients
        unique_indices, inverse_indices = mx.unique(indices, return_inverse=True)
        
        # Aggregate gradients for duplicate indices
        aggregated_grads = mx.zeros((len(unique_indices), gradients.shape[-1]))
        for i, grad in enumerate(gradients):
            idx_pos = mx.where(unique_indices == indices[i])[0][0]
            aggregated_grads = aggregated_grads.at[idx_pos].add(grad)
        
        # Get current weights for these indices
        current_weights = sparse_emb.weights[unique_indices]
        
        # Apply SignSGD with decoupled weight decay
        updated_weights = current_weights * (1.0 - self.lr * self.weight_decay)
        updated_weights = updated_weights - self.lr * mx.sign(aggregated_grads)
        
        # Update the weights
        sparse_emb.weights = sparse_emb.weights.at[unique_indices].set(updated_weights)