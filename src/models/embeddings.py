import torch
from transformers import AutoModel, AutoTokenizer
from config import Config

class LocalEmbedder:
    """
    Exemplo de uso de PyTorch para geração local de embeddings.
    """
    def __init__(self, model_name: str = None):
        self.device = Config.DEVICE
        self.model_name = model_name or Config.EMBEDDING_MODEL
        print(f"Loading local embedding model: {self.model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
    def get_embedding(self, text: str) -> list[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Usa o estado oculto do token [CLS] (índice 0)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            
        return embedding.cpu().tolist()
