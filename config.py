import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model architecture parameters
    VOCAB_SIZE = 256 # For a simple character-level tokenizer
    BLOCK_SIZE = 128 # Maximum context length
    N_EMBD = 128     # Embedding dimension
    N_HEAD = 4       # Number of attention heads
    N_LAYER = 2      # Number of transformer blocks
    DROPOUT = 0.2
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    MAX_ITERS = 5000
    EVAL_INTERVAL = 500
    
    # PyTorch/Local configurations
    DEVICE = os.getenv("DEVICE", "cpu") # 'cuda' or 'cpu'
