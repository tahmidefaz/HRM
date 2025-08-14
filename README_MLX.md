# Hierarchical Reasoning Model - MLX Version

This is an MLX port of the Hierarchical Reasoning Model (HRM) for Apple Silicon Macs. The original CUDA/PyTorch implementation has been converted to run natively on Apple Silicon using Apple's MLX framework.

## What's New in the MLX Version

- **Native Apple Silicon Support**: Runs efficiently on M1, M2, M3, and M4 chips
- **MLX Framework**: Uses Apple's MLX for optimized performance on Apple Silicon
- **Unified Memory**: Takes advantage of Apple Silicon's unified memory architecture
- **Metal Performance Shaders**: Leverages Metal for GPU acceleration

## Quick Start Guide 🚀

### Prerequisites ⚙️

- Apple Silicon Mac (M1, M2, M3, or M4)
- Python 3.8+
- macOS 12.0+ (Monterey or later)

### Install Python Dependencies 🐍

```bash
pip install -r requirements_mlx.txt
```

### Quick Demo: Sudoku Solver 💻

Train a master-level Sudoku AI on your Mac:

```bash
# Download and build Sudoku dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Start training
python pretrain_mlx.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=64 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Note: Batch size is reduced for Apple Silicon memory constraints*

## Key Differences from CUDA Version

### Performance Optimizations
- **Batch Size**: Recommend smaller batch sizes (64-128) vs original (384+) due to memory architecture
- **Memory Management**: MLX automatically manages unified memory
- **Compilation**: No need for CUDA toolkit installation

### Model Architecture
- **Attention**: Uses MLX's optimized `scaled_dot_product_attention`
- **Sparse Embeddings**: Custom MLX implementation for puzzle embeddings
- **Gradient Computation**: Uses MLX's `value_and_grad` for automatic differentiation

### Training
- **Distributed Training**: Currently single-device only (MLX doesn't have distributed training)
- **Mixed Precision**: MLX handles precision automatically
- **Checkpointing**: Uses MLX's native `.npz` format

## Dataset Preparation

Same as original version:

```bash
# Initialize submodules
git submodule update --init --recursive

# ARC-1
python dataset/build_arc_dataset.py

# ARC-2
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Maze
python dataset/build_maze_dataset.py
```

## Training Examples

### ARC-1 (Single Device):
```bash
python pretrain_mlx.py data_path=data/arc-1-aug-1000 global_batch_size=64
```

### ARC-2:
```bash
python pretrain_mlx.py data_path=data/arc-2-aug-1000 global_batch_size=64
```

### Sudoku Extreme:
```bash
python pretrain_mlx.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=64
```

### Maze 30x30:
```bash
python pretrain_mlx.py data_path=data/maze-30x30-hard-1k epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=64
```

## Evaluation

```bash
python evaluate_mlx.py checkpoint=<CHECKPOINT_PATH>
```

## Performance Tips for Apple Silicon

1. **Memory Management**: MLX automatically manages unified memory, but monitor usage with Activity Monitor
2. **Batch Size**: Start with smaller batch sizes (32-64) and increase based on available memory
3. **Thermal Management**: Ensure good ventilation for sustained training
4. **Power Management**: Use power adapter for best performance

## File Structure (MLX-specific files)

```
├── models/
│   ├── layers_mlx.py              # MLX layer implementations
│   ├── losses_mlx.py              # MLX loss functions
│   ├── sparse_embedding_mlx.py    # MLX sparse embedding
│   └── hrm/
│       └── hrm_act_v1_mlx.py      # MLX HRM model
├── pretrain_mlx.py                # MLX training script
├── evaluate_mlx.py                # MLX evaluation script
├── puzzle_dataset_mlx.py          # MLX dataset loader
├── utils/
│   └── functions_mlx.py           # MLX utility functions
├── requirements_mlx.txt           # MLX dependencies
└── README_MLX.md                  # This file
```

## Known Limitations

1. **Distributed Training**: MLX doesn't support multi-device training yet
2. **FlashAttention**: Uses MLX's optimized attention instead of FlashAttention
3. **Compilation**: No explicit model compilation like PyTorch 2.0

## Troubleshooting

### Memory Issues
- Reduce `global_batch_size`
- Close other applications
- Check available memory with `vm_stat`

### Performance Issues
- Ensure you're using a power adapter
- Check thermal throttling with `powermetrics`
- Try reducing model size parameters

### Import Errors
- Make sure MLX is properly installed: `pip install mlx`
- Check Python version compatibility

## Migration from CUDA Version

To convert existing checkpoints:
1. Checkpoints are not directly compatible
2. Retrain using the MLX version
3. Model architecture is preserved, only framework changed

## Contributing

When contributing to the MLX version:
1. Maintain compatibility with the original model architecture
2. Follow MLX best practices
3. Test on different Apple Silicon variants when possible

## Citation

Same as original paper - this is just a framework port:

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```