import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomRewardModel(nn.Module):
    """
    Exemplo de uma rede neural customizada usando PyTorch.
    Pode ser usada para dar "score" ou avaliar saídas locais do agente.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(CustomRewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)  # Score entre 0 e 1
