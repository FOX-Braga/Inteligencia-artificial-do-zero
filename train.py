import os
import torch
import threading
from config import Config
from src.llm.tokenizer import CharTokenizer
from src.llm.transformer import BigramLanguageModel
from src.visualization.nn_gui import NeuralNetVisualizer

def get_batch(data, block_size, batch_size, device):
    """Gera um pequeno batch de dados para input x e target y"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
    "input_text": "",
    "output_text": "",
    "running": True,
}

def training_loop(model, optimizer, train_data, val_data, tokenizer, start_iter=0):
    """Executa o treinamento em uma thread separada."""
    try:
        print(f"Dispositivo de treinamento: {Config.DEVICE}")
        print(f"Iniciando treinamento a partir do passo {start_iter}...")

        for iter in range(start_iter, Config.MAX_ITERS):
            xb, yb = get_batch(train_data, Config.BLOCK_SIZE, Config.BATCH_SIZE, Config.DEVICE)
            logits, loss = model(xb, yb)

            # ── Atualiza a GUI IMEDIATAMENTE após cada forward pass ───────────
            logits_b0    = logits[:Config.BLOCK_SIZE]
            probs        = torch.nn.functional.softmax(logits_b0, dim=-1)
            pred_indices = torch.argmax(probs, dim=-1)

            shared_state["loss"]        = loss.item()
            shared_state["step"]        = iter
            shared_state["input_text"]  = tokenizer.decode(xb[0].tolist()).replace('\n', ' ')
            shared_state["output_text"] = tokenizer.decode(pred_indices.tolist()).replace('\n', ' ')

            # ── Avaliação e salvamento de checkpoint ──────────────────────────
            if iter > start_iter and iter % Config.EVAL_INTERVAL == 0:
                val_loss = estimate_loss(model, val_data)
                print(f"Passo {iter}: Loss de Validação {val_loss:.4f}")

                # Salva checkpoint com modelo + otimizador + passo atual
                torch.save({
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "step":       iter,
                    "vocab_size": Config.VOCAB_SIZE,
                }, Config.CHECKPOINT_PATH)
                print(f"  → Checkpoint salvo em '{Config.CHECKPOINT_PATH}'")

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Salvar checkpoint final
        print("Treinamento finalizado. Salvando checkpoint final...")
        torch.save({
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "step":       Config.MAX_ITERS - 1,
            "vocab_size": Config.VOCAB_SIZE,
        }, Config.CHECKPOINT_PATH)
        print(f"Checkpoint salvo em '{Config.CHECKPOINT_PATH}'")
        shared_state["running"] = False

    except Exception as e:
        print(f"\n[ERRO NA THREAD DE TREINAMENTO]: {e}")
        import traceback; traceback.print_exc()
        shared_state["running"] = False


def load_oasst1_text():
    """
    Baixa e extrai texto do dataset OpenAssistant/oasst1.
    Prioriza mensagens em português (pt), depois usa todas as línguas.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    print("[DATASET] Baixando OpenAssistant/oasst1 do HuggingFace...")
    print("[DATASET] (Isso pode demorar na primeira vez — será cacheado depois)")

    ds = load_dataset("OpenAssistant/oasst1", split="train")

    # Tenta pegar mensagens em português primeiro
    pt_texts = [row["text"] for row in ds if row.get("lang") == "pt" and not row.get("deleted")]
    
    if len(pt_texts) > 200:
        print(f"[DATASET] Usando {len(pt_texts)} mensagens em Português (PT)")
        texts = pt_texts
    else:
        # Fallback: usa todas as línguas
        texts = [row["text"] for row in tqdm(ds, desc="[DATASET] Extraindo texto") if not row.get("deleted")]
        print(f"[DATASET] Usando {len(texts)} mensagens (todos os idiomas)")

    # Formata como diálogos: "Usuário: ... Assistente: ..."
    full_text = "\n\n".join(texts)
    print(f"[DATASET] Total de caracteres carregados: {len(full_text):,}")
    return full_text


def main():
    # 1. Carregar Dataset oasst1
    text = load_oasst1_text()

    # 2. Tokenizar e Salvar Vocab
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    tokenizer.save("vocab.json")
    Config.VOCAB_SIZE = tokenizer.vocab_size
    print(f"Tamanho do vocabulário: {Config.VOCAB_SIZE} tokens únicos")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    print(f"[DATASET] Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    # 3. Inicializar Modelo e Otimizador
    model     = BigramLanguageModel(vocab_size=Config.VOCAB_SIZE).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODELO] Parâmetros totais: {total_params:,}")

    # ── RETOMAR DO CHECKPOINT SE EXISTIR ─────────────────────────────────────
    start_iter = 0
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"[CHECKPOINT] Retomando de '{Config.CHECKPOINT_PATH}'...")
        ckpt = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE, weights_only=False)
        # Só carrega se o vocab for compatível
        if ckpt.get("vocab_size") == Config.VOCAB_SIZE:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_iter = ckpt.get("step", 0) + 1
            print(f"[CHECKPOINT] Continuando do passo {start_iter} / {Config.MAX_ITERS}")
        else:
            print(f"[CHECKPOINT] Vocab mudou ({ckpt.get('vocab_size')} → {Config.VOCAB_SIZE}). Iniciando do zero.")
    else:
        print("[CHECKPOINT] Nenhum checkpoint encontrado — iniciando do zero.")

    # 4. Iniciar treinamento em THREAD SEPARADA
    train_thread = threading.Thread(
        target=training_loop,
        args=(model, optimizer, train_data, val_data, tokenizer, start_iter),
        daemon=True
    )
    train_thread.start()

    # 5. GUI roda na THREAD PRINCIPAL (obrigatório no Windows)
    gui = NeuralNetVisualizer()
    gui.run(shared_state)


if __name__ == "__main__":
    main()
