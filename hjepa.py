"""Hierarchical JEPA on top of a frozen LeWM (LL JEPA).

Time scale: 1 HL step = K low-level (sampled) frames.
HL state: HLE(LL_emb).  Macro action: MAE(K LL actions).
HL prediction: HLP(s_hl, macro_a) -> next HL state.
Loss = ||HLP(...) - HLE(LL_emb_at_+K)||^2  +  lambda * SIGReg(HL embeddings).
"""
import torch
from einops import rearrange
from torch import nn


class HierarchicalJEPA(nn.Module):
    def __init__(self, ll_jepa, hle, hl_projector, hl_pred_proj, hlp, mae, k: int = 5):
        super().__init__()
        self.ll = ll_jepa
        for p in self.ll.parameters():
            p.requires_grad_(False)
        self.ll.eval()
        self.hle = hle
        self.hl_projector = hl_projector
        self.hl_pred_proj = hl_pred_proj
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
        """LL emb -> HL state (post-HLE, pre-projector)."""
        return self.hle(ll_emb)

    def encode_hl_proj(self, ll_emb):
        """LL emb -> projected HL state (post-projector). Loss space."""
        return self.hl_projector(self.hle(ll_emb))

    def encode_macro(self, actions):
        """actions: (B, K, action_dim) raw action tokens -> (B, macro_action_dim).
        MAE consumes the raw tokens directly (Conv1D + transformer + CLS-MLP)."""
        return self.mae(actions)

    def predict_hl(self, states_proj, macros):
        """states_proj: (B, H, hl_dim) projected HL state history.
        macros:        (B, H, hl_dim) macro tokens.
        Returns (B, hl_dim): predicted next state in PROJECTED space (post pred-proj)."""
        return self.hl_pred_proj(self.hlp(states_proj, macros))

    def rollout_hl(self, states_proj_init, macros_init, new_macros):
        """states_proj_init: (B, H, hl_dim) projected state-history buffer.
        macros_init:        (B, H, hl_dim) macro-history buffer.
        new_macros:         (B, T, hl_dim) macros to apply forward.
        Returns (B, T, hl_dim): predicted future states (projected)."""
        states = states_proj_init
        macros = macros_init
        out = []
        for t in range(new_macros.size(1)):
            macros = torch.cat([macros[:, 1:], new_macros[:, t : t + 1]], dim=1)
            s_next = self.predict_hl(states, macros)
            states = torch.cat([states[:, 1:], s_next.unsqueeze(1)], dim=1)
            out.append(s_next)
        return torch.stack(out, dim=1)

    def ll_rollout_from_emb(self, init_emb, action_sequence, history_size: int = 3,
                            return_raw: bool = False):
        """Autoregressive LL rollout from a precomputed LL embedding (no pixels).
        init_emb: (B, 1, ll_dim).  action_sequence: (B, S, T, action_dim).
        Returns (B, S, 1 + T, ll_dim) of post-pred_proj embeddings (the planning
        space). If `return_raw=True`, also returns (B, S, T, ll_dim) of pre-pred_proj
        outputs ("predictor_raw"), useful for decoding via a decoder trained on
        pre-projector embeddings (works iff pred_proj ≈ projector after training)."""
        B, S, T = action_sequence.shape[:3]
        pred_dtype = next(self.ll.predictor.parameters()).dtype
        emb = rearrange(init_emb.unsqueeze(1).expand(B, S, -1, -1),
                        "b s h d -> (b s) h d").clone().to(pred_dtype)
        actions = rearrange(action_sequence, "b s t a -> (b s) t a")
        act_emb_full = self.ll.action_encoder(actions).to(pred_dtype)
        HS = history_size
        raw_steps = [] if return_raw else None
        for t in range(T):
            window_start = max(0, t + 1 - HS)
            out = self.ll.predictor(emb[:, -HS:], act_emb_full[:, window_start : t + 1])
            out = rearrange(out, "b t d -> (b t) d").float()
            if return_raw:
                raw_last = rearrange(out, "(b t) d -> b t d", b=emb.size(0))[:, -1:]
                raw_steps.append(raw_last)
            pred = self.ll.pred_proj(out)
            pred = rearrange(pred, "(b t) d -> b t d", b=emb.size(0))[:, -1:].to(pred_dtype)
            emb = torch.cat([emb, pred], dim=1)
        if return_raw:
            raw = torch.cat(raw_steps, dim=1)  # (B*S, T, ll_dim) in pre-pred_proj space
            raw = rearrange(raw, "(b s) ... -> b s ...", b=B, s=S)
            return rearrange(emb, "(b s) ... -> b s ...", b=B, s=S), raw
        return rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)


