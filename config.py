import os
import torch
from dotenv import load_dotenv

load_dotenv()

def _dml_device():
    try:
        import torch_directml
        if torch_directml.device_count() > 0:
            return torch_directml.device(0)
    except ImportError:
        pass
    return None

def _detect_device():
    """
    Detecta a GPU disponível na máquina, em ordem:
      1. DirectML (AMD/Intel — qualquer placa DX12 no Windows)
      2. CUDA (NVIDIA)
      3. CPU (fallback)
    """
    dml = _dml_device()
    if dml is not None:
        return dml
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# Permite forçar o device via variável de ambiente (DEVICE=cpu|cuda|dml)
_FORCE_DEVICE = os.getenv("DEVICE", "").lower()

def _resolve_device():
    if _FORCE_DEVICE == "cpu":
        return "cpu"
    if _FORCE_DEVICE == "cuda":
        return "cuda"
    if _FORCE_DEVICE == "dml":
        return _dml_device() or "cpu"
    return _detect_device()

class Config:
    # Device de execução (auto-detectado, mas pode ser forçado por DEVICE no .env)
    DEVICE = _resolve_device()

    # ── Arquitetura do Modelo (GPT-like, ~17M parâmetros) ────────────────────
    VOCAB_SIZE = 8000      # Tokenizer BPE (atualizado dinamicamente no treino)
    BLOCK_SIZE = 256       # Tamanho máximo do contexto em tokens
    N_EMBD = 384           # Dimensão de embedding
    N_HEAD = 8             # Número de cabeças de self-attention (head = 48)
    N_LAYER = 6            # Número de blocos Transformer
    DROPOUT = 0.1

    # ── Treinamento ───────────────────────────────────────────────────────────
    BATCH_SIZE = 32        # Grandes batches diluem o overhead fixo do DirectML
    LEARNING_RATE = 3e-4
    MAX_ITERS = 60000      # Estimado: ~2.5s/passo -> ~40h; use parametros=pare quando quiser
    EVAL_INTERVAL = 2000
    CHECKPOINT_PATH = "checkpoint.pth"   # Salva pesos + otimizador