from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        return self.net(x)


class TaskGate(nn.Module):
    def __init__(self, input_dim: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, n_experts)

    def forward(self, x):
        # softmax over experts
        logits = self.gate(x)
        return F.softmax(logits, dim=-1)


class CustomMMoE(nn.Module):
    """Custom MMoE-inspired module.

    Features:
    - multiple shared experts (small MLPs)
    - task-specific gates producing convex mixtures of expert outputs
    - optional residual routing: each task can add a task-specific residual projection
    """

    def __init__(self, input_dim: int, n_experts: int = 4, expert_hidden: int = 128, tower_hidden: int = 64, use_task_residual: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([Expert(input_dim, expert_hidden) for _ in range(n_experts)])
        self.use_task_residual = use_task_residual
        # gates for three tasks (like, longview, creator)
        self.gate_like = TaskGate(input_dim, n_experts)
        self.gate_long = TaskGate(input_dim, n_experts)
        self.gate_creator = TaskGate(input_dim, n_experts)

        # task towers
        self.tower_like = nn.Sequential(nn.Linear(expert_hidden, tower_hidden), nn.ReLU(), nn.Linear(tower_hidden, 1))
        self.tower_long = nn.Sequential(nn.Linear(expert_hidden, tower_hidden), nn.ReLU(), nn.Linear(tower_hidden, 1))
        self.tower_creator = nn.Sequential(nn.Linear(expert_hidden, tower_hidden), nn.ReLU(), nn.Linear(tower_hidden, 1))

        if use_task_residual:
            self.res_like = nn.Linear(input_dim, expert_hidden)
            self.res_long = nn.Linear(input_dim, expert_hidden)
            self.res_creator = nn.Linear(input_dim, expert_hidden)

    def forward(self, shared_repr: torch.Tensor, tasks: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        # shared_repr: (B, input_dim)
        expert_outs = [e(shared_repr) for e in self.experts]  # list of (B, expert_hidden)
        expert_stack = torch.stack(expert_outs, dim=2)  # (B, expert_hidden, n_experts)

        # compute gates
        g_like = self.gate_like(shared_repr)  # (B, n_experts)
        g_long = self.gate_long(shared_repr)
        g_creator = self.gate_creator(shared_repr)

        # mixture: weighted sum over experts
        def mix(gate):
            # gate: (B, n_experts) -> (B, expert_hidden)
            gate = gate.unsqueeze(1)  # (B,1,n_experts)
            mixed = torch.matmul(expert_stack, gate.transpose(1,2)).squeeze(-1)
            return mixed

        out_like = mix(g_like)
        out_long = mix(g_long)
        out_creator = mix(g_creator)

        # optional residuals
        if self.use_task_residual:
            out_like = out_like + self.res_like(shared_repr)
            out_long = out_long + self.res_long(shared_repr)
            out_creator = out_creator + self.res_creator(shared_repr)

        # pass through task towers
        like_score = self.tower_like(out_like).squeeze(-1)
        long_score = self.tower_long(out_long).squeeze(-1)
        creator_score = self.tower_creator(out_creator).squeeze(-1)

        return {"like": like_score, "longview": long_score, "creator": creator_score}
