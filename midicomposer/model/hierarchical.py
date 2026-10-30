"""Hierarchical conditional track generator.

Two levels:
  * Bar level - a bidirectional Transformer encoder over per-bar summaries of the *context*
    tracks (mean of their token embeddings in each bar). This gives every bar a representation
    of what the other instruments do there.
  * Token level - a causal (GPT-style) decoder over the full event sequence. Each token is
    embedded with token + absolute position + bar index + segment (prefix/context/target)
    embeddings; target tokens additionally receive the bar-level context vector of their bar.

The bar summaries only use context-segment tokens, which always precede the target in the
sequence, so no target information leaks into the conditioning. The decoder has a KV cache
for fast autoregressive sampling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .tokenizer import SEG_CONTEXT, SEG_TARGET


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    bar_layers: int = 2
    max_len: int = 1536
    max_bars: int = 64
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.p_drop = cfg.dropout

    def forward(self, x, pad_mask=None, cache=None):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=-1)
        q = q.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        if cache is not None and "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        if cache is not None:
            cache["k"], cache["v"] = k, v
        drop = self.p_drop if self.training else 0.0
        if cache is not None and k.shape[2] > T:
            # Incremental decoding: new queries attend to everything cached so far.
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        elif pad_mask is not None:
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            mask = causal[None, None] & ~pad_mask[:, None, None, :]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class HierarchicalComposer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.max_len, d)
        self.bar = nn.Embedding(cfg.max_bars + 1, d)          # last index = "not inside a bar"
        self.seg = nn.Embedding(3, d)
        enc_layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout, batch_first=True, norm_first=True)
        self.bar_encoder = nn.TransformerEncoder(enc_layer, cfg.bar_layers, enable_nested_tensor=False)
        self.bar_pos = nn.Embedding(cfg.max_bars + 1, d)
        self.bar_proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def bar_context(self, ids, bar_ids, seg_ids):
        """Bar-level encoding of the context tracks: [B, max_bars + 1, d]."""
        cfg = self.cfg
        B = ids.shape[0]
        emb = self.tok(ids)
        is_ctx = (seg_ids == SEG_CONTEXT) & (bar_ids < cfg.max_bars)
        idx = torch.where(is_ctx, bar_ids, torch.full_like(bar_ids, cfg.max_bars))
        sums = torch.zeros(B, cfg.max_bars + 1, emb.shape[-1], device=ids.device, dtype=emb.dtype)
        sums.scatter_add_(1, idx[..., None].expand_as(emb), emb * is_ctx[..., None])
        counts = torch.zeros(B, cfg.max_bars + 1, device=ids.device, dtype=emb.dtype)
        counts.scatter_add_(1, idx, is_ctx.to(emb.dtype))
        summ = sums / counts.clamp(min=1)[..., None]
        summ = summ + self.bar_pos.weight[None]
        empty = counts == 0
        empty[:, -1] = True
        all_empty = empty.all(dim=1)
        # Fully empty rows (no context at all) would make attention NaN; un-mask one slot for them.
        empty = empty.clone()
        empty[all_empty, 0] = False
        out = self.bar_encoder(summ, src_key_padding_mask=empty)
        out = out.masked_fill(empty[..., None], 0.0)
        return out

    def embed(self, ids, bar_ids, seg_ids, positions, bar_ctx):
        x = self.tok(ids) + self.pos(positions) + self.bar(bar_ids) + self.seg(seg_ids)
        tgt = (seg_ids == SEG_TARGET)[..., None]
        ctx_vec = torch.gather(bar_ctx, 1, bar_ids[..., None].expand(-1, -1, bar_ctx.shape[-1]))
        x = x + tgt * self.bar_proj(ctx_vec)
        return self.drop(x)

    def forward(self, ids, bar_ids, seg_ids, pad_mask=None):
        T = ids.shape[1]
        positions = torch.arange(T, device=ids.device)[None].expand_as(ids)
        bar_ctx = self.bar_context(ids, bar_ids, seg_ids)
        x = self.embed(ids, bar_ids, seg_ids, positions, bar_ctx)
        for blk in self.blocks:
            x = blk(x, pad_mask=pad_mask)
        return self.head(self.ln_f(x))

    # ---------------------------------------------------------------- incremental decoding
    @torch.no_grad()
    def prefill(self, ids, bar_ids, seg_ids):
        caches = [dict() for _ in self.blocks]
        T = ids.shape[1]
        positions = torch.arange(T, device=ids.device)[None]
        bar_ctx = self.bar_context(ids, bar_ids, seg_ids)
        x = self.embed(ids, bar_ids, seg_ids, positions, bar_ctx)
        for blk, c in zip(self.blocks, caches):
            x = blk(x, cache=c)
        return self.head(self.ln_f(x[:, -1])), caches, bar_ctx

    @torch.no_grad()
    def step(self, token, bar_id, position, caches, bar_ctx):
        dev = bar_ctx.device
        ids = torch.tensor([[token]], device=dev)
        bars = torch.tensor([[bar_id]], device=dev)
        segs = torch.tensor([[SEG_TARGET]], device=dev)
        pos = torch.tensor([[position]], device=dev)
        x = self.embed(ids, bars, segs, pos, bar_ctx)
        for blk, c in zip(self.blocks, caches):
            x = blk(x, cache=c)
        return self.head(self.ln_f(x[:, -1]))
