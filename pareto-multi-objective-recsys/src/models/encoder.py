from typing import Dict, List
import torch
import torch.nn as nn


class EmbeddingBlock(nn.Module):
    def __init__(self, num_embeddings: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim)

    def forward(self, x):
        return self.emb(x)


class NumericBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.norm(x)
        return self.act(self.proj(x))


class SharedEncoder(nn.Module):
    """Build embeddings for categorical/id groups and normalize numeric features.

    The encoder projects all groups into a shared latent vector.
    """

    def __init__(self, emb_configs: Dict[str, Dict], numeric_dim: int, projection_dim: int):
        super().__init__()
        # emb_configs: {col_name: {"num_embeddings": int, "emb_dim": int}}
        self.emb_confs = emb_configs
        self.embs = nn.ModuleDict({k: EmbeddingBlock(v["num_embeddings"], v["emb_dim"]) for k, v in emb_configs.items()})
        # numeric projection
        self.numeric_block = NumericBlock(numeric_dim, projection_dim) if numeric_dim > 0 else None
        # final projection
        total_emb_dim = sum(v["emb_dim"] for v in emb_configs.values())
        in_dim = total_emb_dim + (projection_dim if numeric_dim > 0 else 0)
        self.final_proj = nn.Sequential(nn.Linear(in_dim, projection_dim), nn.ReLU())

    def forward(self, cat_inputs: Dict[str, torch.LongTensor], numeric_input: torch.Tensor = None):
        emb_outs = []
        for k, emb in self.embs.items():
            if k in cat_inputs:
                emb_outs.append(emb(cat_inputs[k]))
            else:
                # missing input: zero
                emb_outs.append(torch.zeros((numeric_input.shape[0], emb.emb.embedding_dim), device=next(self.parameters()).device))

        if emb_outs:
            cat_feat = torch.cat(emb_outs, dim=-1)
        else:
            cat_feat = None

        num_feat = None
        if self.numeric_block is not None and numeric_input is not None:
            num_feat = self.numeric_block(numeric_input)

        if cat_feat is not None and num_feat is not None:
            x = torch.cat([cat_feat, num_feat], dim=-1)
        elif cat_feat is not None:
            x = cat_feat
        elif num_feat is not None:
            x = num_feat
        else:
            raise ValueError("No input features provided to encoder")

        return self.final_proj(x)
