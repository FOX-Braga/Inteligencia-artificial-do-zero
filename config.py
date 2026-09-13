import os
import torch
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model architecture parameters
    VOCAB_SIZE = 256  # For a simple character-level tokenizer
    BLOCK_SIZE = 128  # Maximum context length
    N_EMBD = 128      # Embedding dimension (leve para CPU)
    N_HEAD = 4        # Number of attention heads
    N_LAYER = 4       # Number of transformer blocks
    DROPOUT = 0.1

    # Training parameters
    BATCH_SIZE = 32        # Batch leve para CPU
    LEARNING_RATE = 1e-3   # LR maior = converge mais rápido
    MAX_ITERS = 50000      # Muitas iterações acumuladas entre runs
    EVAL_INTERVAL = 1000
    CHECKPOINT_PATH = "checkpoint.pth"  # Salva pesos + otimizador

    # PyTorch/Local configurations - auto-detecta GPU automaticamente
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
