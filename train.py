import os
import hashlib
import time
import torch
import threading
import numpy as np
from config import Config
from src.llm.tokenizer import BPETokenizer
from src.llm.transformer import BigramLanguageModel
from src.llm.optimizer import AdamW_DML
from src.visualization.nn_gui import NeuralNetVisualizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


def get_batch(data, block_size, batch_size, device):
    """Gera um pequeno batch de dados para input x e target y"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, data, eval_iters=100):
    """Estima a loss num dataset para avaliar sem atualizar os pesos"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, Config.BLOCK_SIZE, Config.BATCH_SIZE, Config.DEVICE)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

# ── Estado compartilhado entre a thread de treino e a GUI ────────────────────
shared_state = {
    "loss": 0.0,
    "step": 0,
    "total_steps": Config.MAX_ITERS,
    "acc": 0.0,
    "error_rate": 1.0,
    "perplexity": 0.0,
    "vocab_size": 0,
    "input_text": "",
    "output_text": "",
    "inputs": [],      # valores 0-1 dos nós de entrada (15 últimos tokens)
    "hidden1": [],     # ativações reais da 1ª camada oculta amostrada
    "hidden2": [],     # ativações reais da 2ª camada oculta amostrada
    "outputs": [],     # lista de (texto_do_token, prob) dos 10 mais prováveis
    "running": True,
    "status": "Preparando ambiente...",  # mensagem de status p/ a GUI
}

def _setup_activation_hooks(model):
    """
    Registra hooks para capturar as ativações intermediárias reais do modelo.
    Retorna um dict que o loop atualiza: {"h1": tensor(384,), "h2": tensor(384,)}
    """
    activations = {}

    def _make_hook(name):
        def fn(_module, _input, output):
            activations[name] = output.detach().mean(dim=(0, 1))
        return fn

    third = max(1, Config.N_LAYER // 3)
    two_thirds = max(1, 2 * Config.N_LAYER // 3)
    model.blocks[third].register_forward_hook(_make_hook("h1"))
    model.blocks[two_thirds].register_forward_hook(_make_hook("h2"))
    return activations


def _normalize(vals):
    """Normaliza uma lista de números para 0-1 (min-max). Retorna lista float."""
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return [1.0 if v >= hi else 0.0 for v in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _log_error(context: str):
    """Grava o traceback completo em error.log para diagnóstico futuro."""
    import traceback
    try:
        with open(Config.ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {context}\n")
            traceback.print_exc(file=f)
    except Exception:
        pass


def _sample_activations(vector, n=14):
    """Amostra n valores uniformemente de um vetor (ex: 384 ativações)."""
    v = vector.float().cpu()
    idx = torch.linspace(0, v.numel() - 1, n).long()
    return _normalize(v[idx].tolist())


def _to_cpu(obj):
    """Move tensores recursivamente para a CPU (torch_directml não serializa
    tensores DML — torch.save só funciona com tensores em CPU)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_cpu(v) for v in obj]
    return obj


def _save_checkpoint(model, optimizer, step, verbose=True):
    """
    Salva o checkpoint com todos os tensores na CPU. Se o disco estiver
    cheio, avisa no console e NÃO derruba o treinamento.
    """
    try:
        ckpt = {
            "model": _to_cpu(model.state_dict()),
            "optimizer": _to_cpu(optimizer.state_dict()),
            "step": step,
            "vocab_size": Config.VOCAB_SIZE,
        }
        torch.save(ckpt, Config.CHECKPOINT_PATH)
        if verbose:
            print(f"  → Checkpoint salvo em '{Config.CHECKPOINT_PATH}'")
    except Exception as e:
        print(f"[AVISO] Não foi possível salvar checkpoint em "
              f"'{Config.CHECKPOINT_PATH}': {e}")


def training_loop(model, optimizer, train_data, val_data, tokenizer, start_iter=0):
    """Executa o treinamento em uma thread separada."""
    try:
        print(f"Dispositivo de treinamento: {Config.DEVICE}")
        if str(Config.DEVICE) == 'privateuseone:0':
            print("GPU detectada via DirectML (AMD/Intel). Aceleração ativa.")
        print(f"Iniciando treinamento a partir do passo {start_iter}...")

        shared_state["vocab_size"] = Config.VOCAB_SIZE
        shared_state["total_steps"] = Config.MAX_ITERS
        activations = _setup_activation_hooks(model)
        n_inputs = min(15, Config.BLOCK_SIZE)
        acc_steps = max(1, int(getattr(Config, "GRAD_ACCUM_STEPS", 1)))
        optimizer.zero_grad(set_to_none=True)

        for iter in range(start_iter, Config.MAX_ITERS):
            xb, yb = get_batch(train_data, Config.BLOCK_SIZE, Config.BATCH_SIZE, Config.DEVICE)
            logits, loss = model(xb, yb)
            B, T = xb.shape

            # ── Atualiza a GUI IMEDIATAMENTE após cada forward pass ───────────
            logits_3d    = logits.view(B, T, -1)
            probs        = torch.nn.functional.softmax(logits_3d, dim=-1)
            pred_indices = torch.argmax(probs, dim=-1)

            # Acurácia top-1 e % de erro deste batch
            acc = (pred_indices == yb).float().mean().item()
            ppl = torch.exp(loss).item()

            # Entrada real (últimos tokens do contexto) normalizados para 0-1
            raw_in = xb[0, -n_inputs:].float().cpu().div(max(1, Config.VOCAB_SIZE)).tolist()
            # Saída real: top-10 tokens mais prováveis do último token previsto
            last_probs = torch.nn.functional.softmax(logits_3d[0, -1, :], dim=-1)
            topk = last_probs.topk(10)
            out_labels = [tokenizer.decode([i]) for i in topk.indices.tolist()]
            out_probs = [float(p) for p in topk.values.tolist()]

            shared_state["loss"]        = loss.item()
            shared_state["step"]        = iter
            shared_state["acc"]         = acc
            shared_state["error_rate"]  = 1.0 - acc
            shared_state["perplexity"]  = ppl
            shared_state["input_text"]  = tokenizer.decode(xb[0].tolist()).replace('\n', ' ')
            shared_state["output_text"] = tokenizer.decode(pred_indices.flatten().tolist()).replace('\n', ' ')
            shared_state["inputs"]      = _normalize(raw_in)
            shared_state["hidden1"]     = _sample_activations(activations["h1"])
            shared_state["hidden2"]     = _sample_activations(activations["h2"])
            shared_state["outputs"]     = list(zip(out_labels, out_probs))

            # ── Avaliação e salvamento de checkpoint ──────────────────────────
            if iter > start_iter and iter % Config.EVAL_INTERVAL == 0:
                val_loss = estimate_loss(model, val_data)
                print(f"Passo {iter}: Loss de Validação {val_loss:.4f}")

                _save_checkpoint(model, optimizer, iter)

            # Backpropagation — com gradiente ACUMULADO: só aplica o step a cada
            # GRAD_ACCUM_STEPS micro-batches (mantém o batch efetivo alto sem
            # estourar a VRAM da RX 580 no DirectML).
            (loss / acc_steps).backward()
            if (iter + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Aplica o último gradiente acumulado, se sobrou algum
        if Config.MAX_ITERS % acc_steps != 0:
            optimizer.step()

        print("Treinamento finalizado. Salvando checkpoint final...")
        _save_checkpoint(model, optimizer, Config.MAX_ITERS - 1)
        shared_state["running"] = False

    except Exception as e:
        print(f"\n[ERRO NA THREAD DE TREINAMENTO]: {e}")
        import traceback; traceback.print_exc()
        _log_error("ERRO NA THREAD DE TREINAMENTO")
        shared_state["status"] = f"ERRO NO TREINO: {e}"
        shared_state["running"] = False


def load_oasst1_text(max_chars=2_000_000):
    """
    Baixa e extrai texto do dataset OpenAssistant/oasst1.
    Prioriza mensagens em português (pt) e limita o volume a `max_chars`
    (a RX 580 não consegue processar o dataset inteiro em tempo útil).
    """
    from datasets import load_dataset

    print("[DATASET] Baixando OpenAssistant/oasst1 do HuggingFace...")
    print("[DATASET] (Isso pode demorar na primeira vez — será cacheado depois)")

    ds = load_dataset("OpenAssistant/oasst1", split="train")

    pt_texts = [row["text"] for row in ds if row.get("lang") == "pt" and not row.get("deleted")]

    if len(pt_texts) > 200:
        print(f"[DATASET] Usando {len(pt_texts)} mensagens em Português (PT)")
        texts = pt_texts
    else:
        texts = [row["text"] for row in tqdm(ds, desc="[DATASET] Extraindo texto") if not row.get("deleted")]
        print(f"[DATASET] Usando {len(texts)} mensagens (todos os idiomas)")

    full_text = "\n\n".join(texts)
    if len(full_text) > max_chars:
        print(f"[DATASET] Limitando de {len(full_text):,} para {max_chars:,} caracteres "
              "(limite de hardware / tempo de treino).")
        full_text = full_text[:max_chars]
    print(f"[DATASET] Total de caracteres: {len(full_text):,}")
    return full_text


DATA_BIN = Config.DATA_BIN_PATH


def prepare_data(text, tokenizer):
    """
    Tokeniza o dataset UMA vez e guarda em data.bin (com cache).
    Nas próximas execuções o arquivo é recarregado instantaneamente.
    """
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    if os.path.exists(DATA_BIN):
        info = torch.load(DATA_BIN, map_location="cpu", weights_only=False)
        if info.get("text_hash") == text_hash and info.get("vocab_size") == tokenizer.vocab_size:
            print(f"[CACHE] Carregando {DATA_BIN} (dataset já tokenizado)…")
            return info["train"], info["val"]

    print("[BPE] Codificando o dataset inteiro (só na 1ª vez — leva ~1h em Python puro). "
          "Depois fica cacheado em data.bin…")
    ids = tuple(tokenizer.encode(text))
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    torch.save({
        "text_hash": text_hash,
        "vocab_size": tokenizer.vocab_size,
        "train": train_data,
        "val": val_data,
    }, DATA_BIN)
    print(f"[CACHE] Dataset salvo em '{DATA_BIN}' para reutilização.")
    return train_data, val_data


def _try_load_cached():
    """
    Se o vocab.json (BPE) e o data.bin (dataset tokenizado) já existirem e
    forem compatíveis com o VOCAB_SIZE atual, carrega tudo de uma vez — sem
    baixar dataset, sem treinar BPE, sem codificar nada. Retorna
    (tokenizer, train_data, val_data) ou None se algum cache estiver inválido.
    """
    try:
        if not os.path.exists(Config.VOCAB_PATH) or not os.path.exists(DATA_BIN):
            return None
        tokenizer = BPETokenizer()
        tokenizer.load(Config.VOCAB_PATH)
        if tokenizer.vocab_size != Config.VOCAB_SIZE:
            return None
        info = torch.load(DATA_BIN, map_location="cpu", weights_only=False)
        if not isinstance(info, dict) or info.get("vocab_size") != Config.VOCAB_SIZE:
            return None
        if "train" not in info or "val" not in info:
            return None
        print("[CACHE] Vocab + dataset já prontos — pulando download/BPE/encode.")
        return tokenizer, info["train"], info["val"]
    except Exception as e:
        print(f"[CACHE] Cache inválido — recriando do zero. ({e})")
        return None


def _start_training(tokenizer, train_data, val_data):
    """
    Cria o modelo de uma vez, retoma do checkpoint se houver e inicia o
    treinamento em thread separada.
    """
    Config.VOCAB_SIZE = tokenizer.vocab_size
    shared_state["status"] = "Criando modelo e otimizador..."
    model = BigramLanguageModel(vocab_size=Config.VOCAB_SIZE).to(Config.DEVICE)
    # AdamW customizado: só operações suportadas pelo DirectML (sem fallback p/ CPU)
    optimizer = AdamW_DML(model.parameters(), lr=Config.LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODELO] Parâmetros totais: {total_params:,}")

    # ── RETOMAR DO CHECKPOINT SE EXISTIR ────────────────────────────────────
    start_iter = 0
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"[CHECKPOINT] Retomando de '{Config.CHECKPOINT_PATH}'...")
        try:
            # Carrega na CPU e move os estados depois: o map_location="privateuseone:0"
            # quebra no torch_directml (TypeError no load).
            ckpt = torch.load(Config.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            if ckpt.get("vocab_size") == Config.VOCAB_SIZE:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                # Move o estado do otimizador p/ o mesmo device dos parâmetros
                for p in model.parameters():
                    st = optimizer.state.get(p)
                    if st:
                        for k, v in st.items():
                            if isinstance(v, torch.Tensor) and v.device != p.device:
                                st[k] = v.to(p.device)
                start_iter = ckpt.get("step", 0) + 1
                print(f"[CHECKPOINT] Continuando do passo {start_iter} / {Config.MAX_ITERS}")
            else:
                print(f"[CHECKPOINT] Vocab mudou ({ckpt.get('vocab_size')} → {Config.VOCAB_SIZE}). "
                      "Iniciando do zero.")
        except Exception as e:
            print(f"[CHECKPOINT] Checkpoint incompatível ou corrompido ({e}). "
                  "Iniciando do zero (o arquivo antigo será ignorado).")
            os.replace(Config.CHECKPOINT_PATH, Config.CHECKPOINT_PATH + ".incompat")
            start_iter = 0
    else:
        print("[CHECKPOINT] Nenhum checkpoint encontrado — iniciando do zero.")

    # Treinar em THREAD SEPARADA
    shared_state["status"] = "TREINANDO..."
    train_thread = threading.Thread(
        target=training_loop,
        args=(model, optimizer, train_data, val_data, tokenizer, start_iter),
        daemon=True
    )
    train_thread.start()
    print(f"[DATASET] Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")


def prepare_and_train(shared_state):
    """
    Roda em thread separada. Se o cache (vocab.json + data.bin) já estiver
    pronto, NÃO volta do zero: usa o cache e começa a treinar direto. Só
    baixa o dataset/treina o BPE/codifica na 1ª vez.
    """
    try:
        # Caminho rápido: tudo já está cacheado -> não baixa/treina nada
        cached = _try_load_cached()
        if cached is not None:
            tokenizer, train_data, val_data = cached
            print(f"[CACHE] Usando {len(train_data):,} tokens de treino "
                  f"/ {len(val_data):,} de validação já tokenizados.")
            _start_training(tokenizer, train_data, val_data)
            return

        # 1ª vez (ou cache apagado): baixa o dataset, treina o BPE e codifica
        shared_state["status"] = "Baixando dataset OpenAssistant (1a vez demora)..."
        text = load_oasst1_text(max_chars=2_000_000)

        # 2. Tokenizer BPE: se existir vocab.json compatível, reutiliza.
        tokenizer = BPETokenizer()
        if os.path.exists(Config.VOCAB_PATH):
            tokenizer.load(Config.VOCAB_PATH)
        if tokenizer.vocab_size == Config.VOCAB_SIZE:
            print(f"[BPE] Reutilizando '{Config.VOCAB_PATH}' "
                  f"({tokenizer.vocab_size} tokens) — pulando treino.")
        else:
            shared_state["status"] = "Treinando tokenizer BPE (1a vez, demora ~15min)..."
            tokenizer = BPETokenizer()
            tokenizer.train(text, vocab_size=Config.VOCAB_SIZE)
            tokenizer.save(Config.VOCAB_PATH)
        Config.VOCAB_SIZE = tokenizer.vocab_size
        print(f"Tamanho do vocabulário: {Config.VOCAB_SIZE} tokens")

        # 3. Tokenizar (com cache binário) e dividir em treino/validação
        shared_state["status"] = "Tokenizando dataset..."
        train_data, val_data = prepare_data(text, tokenizer)
        _start_training(tokenizer, train_data, val_data)

    except Exception as e:
        print(f"\n[ERRO NA PREPARACAO]: {e}")
        import traceback; traceback.print_exc()
        _log_error("ERRO NA PREPARACAO")
        shared_state["status"] = f"ERRO: {e}"
        shared_state["running"] = False


def main():
    # Preparação + treino em thread separada; GUI abre imediatamente na
    # thread principal (obrigatório no Windows) e mostra o status em tempo real.
    prep_thread = threading.Thread(
        target=prepare_and_train,
        args=(shared_state,),
        daemon=True
    )
    prep_thread.start()

    try:
        gui = NeuralNetVisualizer()
        gui.run(shared_state)
    except Exception as e:
        print(f"\n[ERRO NA GUI]: {e}")
        import traceback; traceback.print_exc()
        _log_error("ERRO NA GUI")
        print("O treinamento continuou em segundo plano. O erro completo foi "
              "salvo em error.log — envie esse arquivo para diagnosticar.")
        input("Pressione Enter para encerrar...")


if __name__ == "__main__":
    main()