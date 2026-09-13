import os
import json

class CharTokenizer:
    """
    Um Tokenizer simples a nível de caractere.
    Ele mapeia cada caractere único encontrado nos dados para um número inteiro.
    """
    def __init__(self):
        self.stoi = {} # string to integer
        self.itos = {} # integer to string
        self.vocab_size = 0
        
    def train(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        
    def encode(self, s: str) -> list[int]:
        # Para simplificar, se encontrar um char desconhecido, ignora ou usa o id 0
        return [self.stoi.get(c, 0) for c in s]
        
    def decode(self, l: list[int]) -> str:
        return ''.join([self.itos.get(i, '') for i in l])
        
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'stoi': self.stoi, 'itos': self.itos, 'vocab_size': self.vocab_size}, f)
            
    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.stoi = data['stoi']
                # json salva as keys do dict como strings, precisamos converter de volta para int no itos
                self.itos = {int(k): v for k, v in data['itos'].items()}
                self.vocab_size = data['vocab_size']
        else:
            # Fallback vocab básico
            self.train("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\n-:")
