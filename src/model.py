"""HybridSASRec: a next-to-watch transformer whose item vectors fuse a learned ID embedding
with a projected text embedding of the title's metadata (from a frozen HF sentence encoder).

This file depends only on torch and huggingface_hub, so it ships unchanged as
``modeling.py`` in the Hugging Face model repository.

Item vector:   v_i = W * content_i  +  E_id[i]            (either part can be switched off)
User state:    causal transformer over the left-padded watch history; the last position
               summarises the user, and score(user, i) = <h_user, v_i>.
ID-dropout:    during training the ID part is zeroed with probability ``id_dropout``, so the
               content path alone can rank titles that were never seen in training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class Block(nn.Module):
    """Pre-LN transformer block using scaled_dot_product_attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.attn_dropout = dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        q, k, v = (
            self.qkv(self.ln1(x)).view(b, length, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        )
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_dropout if self.training else 0.0
        )
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, length, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class HybridSASRec(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="pytorch",
    repo_url="https://github.com/zackyalgiffari/ott-recommender-ml32m",
    license="cc-by-nc-4.0",
    tags=["recommender-system", "sequential-recommendation", "sasrec", "ott", "cold-start"],
):
    def __init__(
        self,
        n_items: int,
        content_dim: int = 768,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 200,
        n_domains: int = 2,
        dropout: float = 0.2,
        id_dropout: float = 0.2,
        use_id: bool = True,
        use_content: bool = True,
        domains: list[str] | None = None,
        domain_ranges: list[list[int]] | None = None,
        encoder: str = "BAAI/bge-base-en-v1.5",
    ):
        super().__init__()
        if not (use_id or use_content):
            raise ValueError("at least one of use_id / use_content must be enabled")
        self.n_items, self.d_model, self.max_len = n_items, d_model, max_len
        self.id_dropout, self.use_id, self.use_content = id_dropout, use_id, use_content
        self.domains = domains or ["movielens", "amazon"]
        self.domain_ranges = domain_ranges  # [[lo, hi), ...] item_idx range of each domain
        self.encoder = encoder

        self.id_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0) if use_id else None
        self.content_proj = nn.Linear(content_dim, d_model) if use_content else None
        # Frozen text embeddings: not part of the state dict (shipped as item_content.safetensors).
        self.register_buffer(
            "content",
            torch.zeros(n_items + 1 if use_content else 1, content_dim, dtype=torch.float16),
            persistent=False,
        )
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dom_emb = nn.Embedding(n_domains, d_model)
        self.in_ln = nn.LayerNorm(d_model)
        self.in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding) and m.padding_idx is not None:
            with torch.no_grad():
                m.weight[m.padding_idx].zero_()

    # ------------------------------------------------------------------ item side
    def set_content(self, content: torch.Tensor) -> None:
        """Load the (n_items + 1, content_dim) text-embedding table; row 0 is padding."""
        if self.use_content:
            self.content = content.to(device=self.content.device, dtype=torch.float16).contiguous()

    @torch.no_grad()
    def reset_cold_ids(self, cold_idx: torch.Tensor) -> int:
        """Zero the ID vectors of titles that never occurred in training.

        Such titles are only ever seen as sampled negatives, so their ID vectors learn to push
        them down. With the ID part zeroed they are ranked by content alone -- the path that
        ID-dropout trains for.
        """
        if not self.use_id or not self.use_content:
            return 0
        cold_idx = cold_idx[cold_idx > 0]
        self.id_emb.weight[cold_idx.to(self.id_emb.weight.device)] = 0.0
        return int(cold_idx.numel())

    def content_vectors(self, emb: torch.Tensor) -> torch.Tensor:
        """Project raw encoder embeddings (e.g. of brand-new titles) into item space."""
        return self.content_proj(emb.to(self.content_proj.weight.dtype))

    def item_vectors(self, idx: torch.Tensor, id_scale: float = 1.0, drop_id: bool = False) -> torch.Tensor:
        v = None
        if self.use_content:
            v = self.content_vectors(self.content[idx])
        if self.use_id and id_scale != 0.0:
            e = self.id_emb(idx)
            if drop_id and self.training and self.id_dropout > 0:
                keep = (torch.rand(e.shape[:-1], device=e.device) >= self.id_dropout).unsqueeze(-1)
                e = e * keep.to(e.dtype)
            e = e * id_scale if id_scale != 1.0 else e
            v = e if v is None else v + e
        if v is None:
            raise ValueError("id_scale=0 needs a model with content vectors")
        return v

    def domain_items(self, domain: int, id_scale: float = 1.0) -> torch.Tensor:
        """Vectors of every catalog item of one domain, in item_idx order."""
        lo, hi = self.domain_ranges[domain]
        idx = torch.arange(lo, hi, device=self.pos_emb.weight.device)
        return self.item_vectors(idx, id_scale=id_scale)

    # ------------------------------------------------------------------ user side
    def forward(self, seq: torch.Tensor, domain: torch.Tensor, id_scale: float = 1.0) -> torch.Tensor:
        """seq: (B, L) left-padded item_idx (0 = padding); domain: (B,). Returns (B, L, d)."""
        b, length = seq.shape
        valid = seq != 0
        # Position 0 = first real item of the (window of the) history.
        n_pad = length - valid.sum(1, keepdim=True)
        pos = (torch.arange(length, device=seq.device)[None, :] - n_pad).clamp(0, self.max_len - 1)
        x = self.item_vectors(seq, id_scale=id_scale, drop_id=True)
        x = x + self.pos_emb(pos) + self.dom_emb(domain)[:, None, :]
        x = self.in_drop(self.in_ln(x))
        # Causal attention that ignores padded keys; every query may attend to itself, so a
        # padded row never has all keys masked (which would produce NaN).
        causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).tril()
        eye = torch.eye(length, dtype=torch.bool, device=seq.device)
        mask = (causal[None] & (valid[:, None, :] | eye[None]))[:, None]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.out_ln(x)

    def user_state(self, seq: torch.Tensor, domain: torch.Tensor, id_scale: float = 1.0) -> torch.Tensor:
        """(B, d) representation after the most recent item (sequences are left-padded)."""
        return self(seq, domain, id_scale=id_scale)[:, -1]
