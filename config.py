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

    # ── Caminhos da LLM (D:\LLM — o C: estava sem espaço) ──────────────────
    # Mova o projeto para D e aponte aqui. Ajustável via LLM_DIR no .env.
    LLM_DIR = os.getenv("LLM_DIR", r"D:\LLM")
    os.makedirs(LLM_DIR, exist_ok=True)
    VOCAB_PATH = os.path.join(LLM_DIR, "vocab.json")
    DATA_BIN_PATH = os.path.join(LLM_DIR, "data.bin")
    MODEL_PATH = os.path.join(LLM_DIR, "model_weights.pth")
    ERROR_LOG = os.getenv("ERROR_LOG", os.path.join(LLM_DIR, "error.log"))

    # ── Arquitetura do Modelo (GPT-like, ~17M parâmetros) ────────────────────
    VOCAB_SIZE = 8000      # Tokenizer BPE (atualizado dinamicamente no treino)
    BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "128"))   # Contexto max em tokens
    # 128 em vez de 256: a RX 580 2048SP tem só 4GB de VRAM e 256 estourava a
    # memória do DirectML (dá OOM / crash no treino). Pode ser ajustado via env.
    N_EMBD = 384           # Dimensão de embedding
    N_HEAD = 8             # Número de cabeças de self-attention (head = 48)
    N_LAYER = 6            # Número de blocos Transformer
    DROPOUT = 0.1

    # ── Treinamento ───────────────────────────────────────────────────────────
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))      # micro-batch por passo (DML)
    GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "4"))
    # Gradiente acumulado em BATCH_SIZE*GRAD_ACCUM_STEPS passos = batch efetivo 32.
    # Batch grande em 1 passo sozinho estourava a VRAM (4GB) e crashava o treino.
    LEARNING_RATE = 3e-4
    MAX_ITERS = int(os.getenv("MAX_ITERS", "60000"))    # ~2.5s/passo; pare quando quiser
    EVAL_INTERVAL = 2000
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH",
                                os.path.join(LLM_DIR, "checkpoint.pth"))   # Salva pesos + otimizador