import torch
from config import Config
from src.llm.tokenizer import CharTokenizer
from src.llm.transformer import BigramLanguageModel
from src.visualization.nn_gui import NeuralNetVisualizer

def get_batch(data, block_size, batch_size, device):
    """Gera um pequeno batch de dados para input x e target y"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, eval_iters=200):
    """Estima a loss num dataset para avaliar sem atualizar os pesos"""
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, Config.BLOCK_SIZE, Config.BATCH_SIZE, Config.DEVICE)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

def main():
    print(f"Dispositivo de treinamento: {Config.DEVICE}")
    
    # 1. Carregar Dataset Toy
    # Para demonstração, vamos usar um texto de placeholder.
    # No mundo real, você deve carregar os dados de um arquivo texto enorme.
    text = "O projeto de Inteligência Artificial do zero visa construir um modelo de linguagem." * 100
    
    # 2. Tokenizar e Salvar Vocab
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    tokenizer.save("vocab.json")
    Config.VOCAB_SIZE = tokenizer.vocab_size
    print(f"Tamanho do vocabulário: {Config.VOCAB_SIZE}")
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # 3. Inicializar Modelo, Otimizador e GUI
    model = BigramLanguageModel(vocab_size=Config.VOCAB_SIZE).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Inicializa a Janela Gráfica
    print("Iniciando interface gráfica...")
    gui = NeuralNetVisualizer()
    
    # 4. Loop de Treinamento
    print("Iniciando treinamento...")
    
    # Variáveis para a GUI
    current_input_text = ""
    current_output_text = ""
    
    for iter in range(Config.MAX_ITERS):
        
        # Pega o batch
        xb, yb = get_batch(train_data, Config.BLOCK_SIZE, Config.BATCH_SIZE, Config.DEVICE)
        
        # Calcula a loss
        logits, loss = model(xb, yb)
        
        # Avaliar de tempos em tempos (Apenas imprime no terminal, sem spam)
        if iter % Config.EVAL_INTERVAL == 0 or iter == Config.MAX_ITERS - 1:
            val_loss = estimate_loss(model, val_data)
            print(f"Passo {iter}: Loss de Validação {val_loss:.4f}")
            
        # Atualiza a janela gráfica com a loss de treinamento atual e textos constantes
        if iter % 5 == 0:  # Atualiza a tela a cada 5 passos
            # Extrair os logits do primeiro exemplo do batch (os primeiros BLOCK_SIZE elementos)
            # Como logits foram achatados para (B*T, C), pegamos os primeiros T elementos
            logits_b0 = logits[:Config.BLOCK_SIZE]
            probs = torch.nn.functional.softmax(logits_b0, dim=-1) 
            pred_indices = torch.argmax(probs, dim=-1)
            
            current_input_text = tokenizer.decode(xb[0].tolist()).replace('\n', ' ')
            current_output_text = tokenizer.decode(pred_indices.tolist()).replace('\n', ' ')
            
            gui.update(loss.item(), iter, current_input_text, current_output_text)
            
        # Backpropagation e otimização
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 5. Salvar Modelo Treinado
    print("Treinamento finalizado. Salvando modelo...")
    torch.save(model.state_dict(), "model_weights.pth")
    print("Pesos salvos em 'model_weights.pth'")

if __name__ == "__main__":
    main()
