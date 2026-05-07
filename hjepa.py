"""Hierarchical JEPA on top of a frozen LeWM (LL JEPA).

Time scale: 1 HL step = K low-level (sampled) frames.
HL state: HLE(LL_emb).  Macro action: MAE(K LL actions).
HL prediction: HLP(s_hl, macro_a) -> next HL state.
Loss = ||HLP(...) - HLE(LL_emb_at_+K)||^2  +  lambda * SIGReg(HL embeddings).
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class HierarchicalJEPA(nn.Module):
    def __init__(self, ll_jepa, hle, hlp, mae, k: int = 5):
        super().__init__()
        self.ll = ll_jepa
        for p in self.ll.parameters():
            p.requires_grad_(False)
        self.ll.eval()
        self.hle = hle
        self.hlp = hlp
        self.mae = mae
        self.k = k

    def train(self, mode: bool = True):
        super().train(mode)
        self.ll.eval()  # keep LL frozen even in train mode
        return self

    @torch.no_grad()
    def encode_ll(self, pixels):
        """pixels: (B, T, C, H, W) -> (B, T, ll_dim) (post-projector CLS)."""
        return self.ll.encode({"pixels": pixels})["emb"]

    def encode_hl(self, ll_emb):
        return self.hle(ll_emb)

    def encode_macro(self, actions):
        """actions: (B, K, A) -> (B, macro_action_dim)."""
        return self.mae(actions)

    def predict_hl(self, s_hl, macro_a):
        return self.hlp(s_hl, macro_a)

    def rollout_hl(self, s_hl_init, macro_actions):
        """s_hl_init: (B, hl_dim), macro_actions: (B, T_HL, macro_a_dim)."""
        s = s_hl_init
        traj = [s]
        for t in range(macro_actions.size(1)):
            s = self.predict_hl(s, macro_actions[:, t])
            traj.append(s)
        return torch.stack(traj, dim=1)  # (B, T_HL+1, hl_dim)


class HLPlanModel(nn.Module):
    """Wrapper exposing HL CEM cost (matches CEMSolver Costable protocol)."""

    def __init__(self, hjepa: HierarchicalJEPA):
        super().__init__()
        self.h = hjepa

    def get_cost(self, info_dict, macro_candidates):
        """info_dict: {'state_hl_init': (B, hl_dim), 'goal_hl': (B, hl_dim)}.
        macro_candidates: (B, S, T_HL, macro_a_dim). Returns (B, S)."""
        s_init = info_dict["state_hl_init"]
        goal = info_dict["goal_hl"]
        B, S, T, _ = macro_candidates.shape
        s = s_init.unsqueeze(1).expand(-1, S, -1).reshape(B * S, -1)
        macro = rearrange(macro_candidates, "b s t a -> (b s) t a")
        for t in range(T):
            s = self.h.predict_hl(s, macro[:, t])
        s = s.view(B, S, -1)
        cost = ((s - goal.unsqueeze(1)) ** 2).sum(dim=-1)
        return cost


class SubgoalLLModel(nn.Module):
    """Adapter: LL CEM finds actions to reach an HL subgoal.
    Cost = ||HLE(LL_predicted_emb_at_T) - subgoal||^2."""

    def __init__(self, hjepa: HierarchicalJEPA):
        super().__init__()
        self.h = hjepa

    def get_cost(self, info_dict, action_candidates):
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)
        info_dict = self.h.ll.rollout(info_dict, action_candidates)
        pred_emb = info_dict["predicted_emb"]      # (B, S, T+1, ll_dim)
        final_ll = pred_emb[..., -1, :]            # (B, S, ll_dim)
        final_hl = self.h.encode_hl(final_ll)      # (B, S, hl_dim)
        subgoal = info_dict["subgoal_hl"]          # (B, S, hl_dim) (CEM expanded)
        return ((final_hl - subgoal) ** 2).sum(dim=-1)
