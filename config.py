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
      1. CPU (padrão — RAM é mais estável que o DirectML na RX 580)
      2. DirectML (AMD/Intel — opt-in via DEVICE=dml)
      3. CUDA (NVIDIA — opt-in via DEVICE=cuda)
    """
    return "cpu"

# Permite forçar o device via variável de ambiente (DEVICE=cpu|cuda|dml).
# Padrão: CPU. Para voltar ao DirectML, definir DEVICE=dml no .env.
_FORCE_DEVICE = os.getenv("DEVICE", "").lower()

def _resolve_device():
    if _FORCE_DEVICE == "dml":
        return _dml_device() or "cpu"
    if _FORCE_DEVICE == "cuda":
        return "cuda"
    return "cpu"

# CPU: usa todos os núcleos lógicos (Xeon 6C/12T). Ajustável via TORCH_THREADS.
torch.set_num_threads(int(os.getenv("TORCH_THREADS", os.cpu_count() or 6)))

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
    # 128 (e não 256): a RX 580 2048SP tem só 4GB de VRAM e 256 estourava no
    # DirectML. Em CPU/RAM (padrão atual) também é um bom tamanho de contexto.
    N_EMBD = 384           # Dimensão de embedding
    N_HEAD = 8             # Número de cabeças de self-attention (head = 48)
    N_LAYER = 6            # Número de blocos Transformer
    DROPOUT = 0.1

    # ── Treinamento ───────────────────────────────────────────────────────────
    # Em CPU/RAM (padrão) dá para usar micro-batch maior que na VRAM da RX 580.
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))     # micro-batch por passo
    GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "2"))
    # Batch efetivo = BATCH_SIZE * GRAD_ACCUM_STEPS = 32 (mesmo de antes).
    LEARNING_RATE = 3e-4
    MAX_ITERS = int(os.getenv("MAX_ITERS", "60000"))    # pare quando quiser
    EVAL_INTERVAL = 2000
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH",
                                os.path.join(LLM_DIR, "checkpoint.pth"))   # Salva pesos + otimizador