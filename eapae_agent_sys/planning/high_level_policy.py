from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer
class HighLevelPolicy(nn.Module):
    """
    Stage 2.1: High-Level Policy Network π_θ.
    Selects a collaboration pattern for a given task, considering the budget.
    Balanced architecture with aligned feature dimensions.
    """
    def __init__(self, embedding_dim: int, num_patterns: int, max_budget: float, hidden_dim: int = 128, device: str = 'cpu'):
        super(HighLevelPolicy, self).__init__()
        self.device = device
        self.max_budget = max_budget
        self.embedding_model_name = 'all-MiniLM-L6-v2'
        feature_dim = hidden_dim // 2
        self.embedding_layer = nn.Linear(embedding_dim, feature_dim)
        self.budget_layer = nn.Linear(1, feature_dim)
        self.combined_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_patterns)
        ).to(self.device)
        self.embedding_layer.to(device)
        self.budget_layer.to(device)
        self.model = SentenceTransformer(self.embedding_model_name, device=self.device)
    def forward(self, task_descriptions: List[str], budgets: torch.Tensor, forced_action_idx: torch.Tensor = None, is_deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with aligned feature dimensions.
        """
        with torch.no_grad():
            embedding = self.model.encode(task_descriptions, convert_to_tensor=True, device=self.device)
        normalized_budget = (budgets.unsqueeze(1) / self.max_budget).to(torch.float32)
        embedding_processed = self.embedding_layer(embedding)
        budget_processed = self.budget_layer(normalized_budget)
        combined_representation = F.relu(embedding_processed + budget_processed)
        action_logits = self.combined_network(combined_representation)
        dist = Categorical(logits=action_logits)
        if forced_action_idx is not None:
            action = forced_action_idx
        elif is_deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), action_logits 