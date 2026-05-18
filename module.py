import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation.

    NOTE: AdaLN-zero conditioning (this fn + ConditionalBlock + the c-handling
    in Transformer below) is **only used by the frozen LL predictor**
    (ARPredictor's transformer was instantiated with block_class=ConditionalBlock
    during LL training, so the LL checkpoint pickle references these classes
    by path). The HL stack uses interleaved-token conditioning instead — do
    NOT delete these classes or unpickling the LL ckpt will break.
    """
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # Only allocate cond_proj when blocks actually consume the `c` argument
        # (i.e. ConditionalBlock with AdaLN). For plain Blocks, an unused Linear
        # here makes DDP complain about parameters with no gradient.
        if block_class is not Block and input_dim != hidden_dim:
            self.cond_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.cond_proj = nn.Identity()

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class HLEncoder(nn.Module):
    """LL CLS embedding (ll_dim) -> HL state (hl_dim). MLP with `depth` Linear
    layers, **LayerNorm** + GELU between hidden layers (matches LL's ViT
    encoder norm choice). depth must be >= 2."""

    def __init__(self, ll_dim=192, hl_dim=96, hidden_dim=256, depth=4):
        super().__init__()
        assert depth >= 2
        layers = [nn.Linear(ll_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.LayerNorm(hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, hl_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPHead(nn.Module):
    """2-layer MLP with expansion (hl_dim -> hidden -> hl_dim), with
    BatchNorm1d on the hidden — matches LL projector / pred_proj exactly."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        # BN1d needs (N, C); reshape if input is (B, T, C)
        if x.dim() == 3:
            B, T, D = x.shape
            return self.net(x.reshape(B * T, D)).view(B, T, D)
        return self.net(x)


class MacroActionEncoder(nn.Module):
    """Conv1D + causal transformer over RAW action tokens (action_dim-d, e.g.
    10-d for PushT packed env-actions). The transformer operates at
    `hidden_dim` internally, but `output_proj` brings the CLS back to
    `action_dim` so the output sequence preserves the raw-token semantics.
    Final MLP head maps the `action_dim`-d CLS down to `macro_action_dim`.

    Architecture:
        actions  (B, K, action_dim)
            ↓  Conv1d(action_dim → action_dim, k=1)   [per-token feature mixing]
        x        (B, K, action_dim)
            ↓  concat CLS + add learned pos-embed
        x        (B, K+1, action_dim)
            ↓  Transformer(in=action_dim, hidden=hidden_dim, out=action_dim,
                           depth, heads, dim_head, mlp_dim, causal)
        x        (B, K+1, action_dim)
            ↓  CLS (last token) → MLP(action_dim → mlp_head_dim → macro_action_dim)
        macro    (B, macro_action_dim)
    """

    def __init__(self, action_dim=10, macro_action_dim=3, max_k=10,
                 depth=4, hidden_dim=64, heads=4, dim_head=16, mlp_dim=128,
                 mlp_head_dim=32, bounded=False):
        super().__init__()
        self.patch_embed = nn.Conv1d(action_dim, action_dim, kernel_size=1)
        self.cls = nn.Parameter(torch.randn(1, 1, action_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, max_k + 1, action_dim) * 0.02)
        self.transformer = Transformer(
            input_dim=action_dim, hidden_dim=hidden_dim, output_dim=action_dim,
            depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
            block_class=Block,
        )
        head_layers = [nn.Linear(action_dim, mlp_head_dim), nn.GELU(),
                       nn.Linear(mlp_head_dim, macro_action_dim)]
        if bounded:
            head_layers.append(nn.Tanh())
        self.head = nn.Sequential(*head_layers)

    def forward(self, actions):
        """actions: (B, K, action_dim) raw action tokens -> (B, macro_action_dim).
        K may vary up to max_k."""
        B, K = actions.shape[:2]
        x = self.patch_embed(actions.permute(0, 2, 1)).permute(0, 2, 1)  # (B, K, action_dim)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([x, cls], dim=1)                                    # (B, K+1, action_dim)
        x = x + self.pos_embed[:, : K + 1]
        x = self.transformer(x)                                           # (B, K+1, action_dim)
        return self.head(x[:, -1])                                        # (B, macro_action_dim)


class HLPredictor(nn.Module):
    """Causal transformer over interleaved (state, macro) tokens (DINO-WM /
    HWM style). The macro is projected to `hl_dim`, then interleaved with
    state tokens so it enters the predictor purely via softmax-attended K/V.

    Sequence layout for history H: [s_0, m_0, s_1, m_1, ..., s_{H-1}, m_{H-1}]
    (length 2H). Causal mask. Next-state prediction is read at position 2H-1
    (the output corresponding to the last macro token), matching the HWM
    PushT convention of predicting ẑ_{t+1} from the last (z, l)-pair output.

    Type embeddings distinguish state vs. macro positions; learned positional
    embeddings cover all 2H slots."""

    def __init__(self, hl_dim=96, macro_action_dim=96, history=3, depth=6,
                 hidden_dim=None, heads=4, dim_head=24, mlp_dim=192):
        super().__init__()
        self.history = history
        D = hidden_dim if hidden_dim is not None else hl_dim
        # project both state and macro into the transformer's hidden dim
        self.state_proj = nn.Linear(hl_dim, D) if hl_dim != D else nn.Identity()
        self.macro_proj = nn.Linear(macro_action_dim, D)
        self.pos_embed = nn.Parameter(torch.randn(1, 2 * history, D) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(2, D) * 0.02)        # 0=state, 1=macro
        self.transformer = Transformer(
            input_dim=D, hidden_dim=D, output_dim=D,
            depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
            block_class=Block,
        )
        self.head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(),
            nn.Linear(D, hl_dim),
        )

    def forward(self, states, macros):
        """states: (B, H, hl_dim).  macros: (B, H, macro_action_dim).
        Returns (B, hl_dim) — read at the last macro position."""
        B, H = states.shape[:2]
        s = self.state_proj(states) + self.type_embed[0]              # (B, H, D)
        m = self.macro_proj(macros) + self.type_embed[1]              # (B, H, D)
        D = s.size(-1)
        x = torch.stack([s, m], dim=2).reshape(B, 2 * H, D)            # (B, 2H, D)
        x = x + self.pos_embed[:, : 2 * H]
        x = self.transformer(x)
        return self.head(x[:, -1])     # last macro position
