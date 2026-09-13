import torch
from src.llm.tokenizer import BPETokenizer
from src.llm.transformer import BigramLanguageModel
from config import Config
import os


class LocalGenerator:
    def __init__(self, model_path: str = "model_weights.pth", vocab_path: str = "vocab.json"):
        self.device = Config.DEVICE
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(vocab_path)

        # Garantir que o config usa o tamanho real do vocabulário carregado
        if self.tokenizer.vocab_size > 0:
            Config.VOCAB_SIZE = self.tokenizer.vocab_size

        self.model = BigramLanguageModel(vocab_size=Config.VOCAB_SIZE)

        if os.path.exists(model_path):
            print(f"Carregando pesos de {model_path}...")
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=False)
                # Aceita tanto um state_dict puro quanto um dict com a chave "model"
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                self.model.load_state_dict(state)
                print(f"Pesos carregados com sucesso. Vocab: {self.tokenizer.vocab_size} tokens.")
            except Exception as e:
                print(f"Não foi possível carregar os pesos existentes ({e}).")
                print("Os pesos foram salvos com uma arquitetura anterior — o modelo"
                      " será iniciado com pesos aleatórios. Rode 'python train.py' para retreinar.")
        else:
            print("Nenhum peso encontrado. Inicializando modelo com pesos aleatórios.")

        self.model.to(self.device)
        self.model.eval()

    def generate_text(self, prompt: str, max_new_tokens: int = 100) -> str:
        # Codifica o texto para tensores
        context = self.tokenizer.encode(prompt)
        # Transforma numa batch de 1 (1, T) e joga pro device
        x = torch.tensor([context], dtype=torch.long, device=self.device)

        # Gera novos tokens
        y = self.model.generate(x, max_new_tokens=max_new_tokens)

        # Pega a primeira sequencia do batch e decodifica de volta pra string
        output_list = y[0].tolist()
        # Cortar a parte do prompt, retornando só o que foi gerado
        generated_tokens = output_list[len(context):]
        return self.tokenizer.decode(generated_tokens)