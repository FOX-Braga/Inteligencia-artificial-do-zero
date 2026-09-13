import os
import json

class CharTokenizer:
    """
    Um Tokenizer simples a nível de caractere.
    Ele mapeia cada caractere único encontrado nos dados para um número inteiro.
    Mantido apenas como referência educacional — para uso real prefira BPETokenizer.
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
                self.itos = {int(k): v for k, v in data['itos'].items()}
                self.vocab_size = data['vocab_size']
        else:
            self.train("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\n-:")


class BPETokenizer:
    """
    Tokenizer BPE (Byte Pair Encoding) implementado do zero.

    Funciona no nível de bytes UTF-8: o texto é convertido em bytes (valores
    0-255) e os pares de bytes mais frequentes são fundidos repetidamente em
    novos tokens. Isso funciona para qualquer idioma (inclusive português com
    acentuação) e é o mesmo princípio usado por LLMs como GPT-2.

    Vocabulário final:
      - IDs 0..255   : os 256 bytes UTF-8 individuais
      - IDs 256..    : merges de BPE (subpalavras/palavras comuns)
    """
    def __init__(self):
        self.vocab_size = 0
        self.merges = []            # lista de pares (id_a, id_b)
        self._id_to_token = {}      # id -> tupla de bytes (para decode)

    def train(self, text: str, vocab_size: int = 8000, max_train_chars: int = 300_000):
        """
        Treina o BPE no texto dado.
        O treino é limitado a `max_train_chars` chars para velocidade —
        suficiente para capturar os merges mais comuns (o vocabulário resultante
        é então aplicado ao corpus inteiro via encode).
        """
        text_bytes = text.encode('utf-8')
        train_bytes = text_bytes[:max_train_chars]
        tokens = list(train_bytes)

        num_merges = vocab_size - 256
        self.merges = []

        print(f"[BPE] Treinando {num_merges} merges em {len(train_bytes):,} bytes...")

        for i in range(num_merges):
            if len(tokens) < 2:
                break

            # Conta a frequência de cada par de tokens adjacentes
            stats = {}
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                stats[pair] = stats.get(pair, 0) + 1

            if not stats:
                break

            # Pega o par mais frequente
            pair = max(stats, key=stats.get)
            new_id = 256 + i

            # Funde todas as ocorrências do par no corpus
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == pair[0] and tokens[j + 1] == pair[1]:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1

            tokens = new_tokens
            self.merges.append(pair)

            if (i + 1) % 1000 == 0 or i == 0:
                print(f"  [BPE] merge {i + 1}/{num_merges}: par {pair} "
                      f"({stats[pair]:,} ocorrências) -> vocab {256 + len(self.merges)}")

        self.vocab_size = 256 + len(self.merges)
        print(f"[BPE] Treino completo. Vocab final: {self.vocab_size} tokens")
        self._build_vocab()

    def _build_vocab(self):
        self._id_to_token = {}
        for b in range(256):
            self._id_to_token[b] = (b,)
        for idx, (a, b) in enumerate(self.merges):
            self._id_to_token[256 + idx] = self._id_to_token[a] + self._id_to_token[b]

    def encode(self, s: str) -> list[int]:
        """Codifica texto em uma lista de IDs aplicando os merges em ordem."""
        tokens = list(s.encode('utf-8'))

        for merge_id, (a, b) in enumerate(self.merges):
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == a and tokens[j + 1] == b:
                    new_tokens.append(256 + merge_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

        return tokens

    def decode(self, l: list[int]) -> str:
        """Decodifica uma lista de IDs de volta ao texto."""
        all_bytes = b''
        for token_id in l:
            all_bytes += bytes(self._id_to_token[token_id])
        return all_bytes.decode('utf-8', errors='replace')

    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            merges = data.get("merges")
            if merges is not None:
                self.vocab_size = data["vocab_size"]
                self.merges = [tuple(m) for m in merges]
                self._build_vocab()
            else:
                # Vocabulário antigo (char-level) — incompatível, inicia fallback.
                print(f"[BPE] '{path}' está no formato antigo (char-level). "
                      "Treine novamente com 'python train.py'.")
                self._init_fallback()
        else:
            self._init_fallback()

    def _init_fallback(self):
        self.vocab_size = 256
        self.merges = []
        self._id_to_token = {b: (b,) for b in range(256)}