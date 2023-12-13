from typing import Dict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_pos=512):
        super().__init__()

        pos = torch.arange(max_pos)

        freq = torch.arange(dim//2) / dim
        freq = (freq * torch.tensor(10000).log()).exp()

        x = rearrange(pos, 'L -> L 1') / freq
        x = rearrange(x, 'L d -> L d 1')

        pe = torch.cat((x.sin(), x.cos()), dim=-1)
        self.pe = rearrange(pe, 'L d sc -> L (d sc)')

    def forward(self, n, *, device=torch.device('cpu')):
        enc = self.pe[:n]
        return enc.to(device)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim, p_drop):
        super().__init__()

        self.scale = dim ** 0.5
        self.fill_val = torch.tensor(-float('inf'))

        self.dropout = nn.Dropout(p_drop)

    def forward(self, q, k, v, mask=None):
        qk = torch.einsum('...id,...jd->...ij', q, k)
        scaled_qk = qk / self.scale

        if mask is not None:
            scaled_qk.masked_fill_(mask, self.fill_val)

        attn = F.softmax(scaled_qk, dim=-1)
        attn = self.dropout(attn)

        out = einsum('...ij,...jd->...id', attn, v)
        return out


class LangGuidedAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads

        kv_dim = dim // n_heads
        proj_dim = kv_dim * n_heads

        self.q_proj = nn.Linear(dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(dim, proj_dim, bias=False)
        self.v_proj = nn.Linear(dim, proj_dim, bias=False)

        self.attention = ScaledDotProductAttention(kv_dim, p_drop)

        self.out_proj = nn.Linear(proj_dim, dim, bias=False)

    def forward(self, q, kv, mask=None):
        qkv = torch.stack([self.q_proj(q), self.k_proj(kv), self.v_proj(kv)])
        q, k, v = rearrange(qkv, 'qkv b l (h d) -> qkv b h l d', h=self.n_heads)

        attn = self.attention(q, k, v, mask=mask)
        attn = rearrange(attn, 'b h l d -> b l (h d)')

        out = self.out_proj(attn)
        return out


class CLIP_IT(nn.Module):
    def __init__(self,
                 clip_model_name: str,
                 num_sentences: int = 7,
                 lgattn_heads: int = 4,
                 transformer_kwags: Dict = {}
                 ):
        super(CLIP_IT, self).__init__()

        self.image_preprocess, self.clip_model = clip.load(clip_model_name)
        self.dim = self.clip_model.visual.output_dim

        self.fusion_mlp = nn.Linear(self.dim * num_sentences, self.dim)
        self.num_sentences = num_sentences

        self.lgattn = LangGuidedAttention(self.dim, n_heads=lgattn_heads)

        self.pe = PositionalEncoding(self.dim, max_pos=4096)
        self.frame_scoring_transformer = nn.Transformer(self.dim, **transformer_kwags)

    def forward(self, videos, texts):
        assert len(texts) == self.num_sentences
        b, c, f, h, w = videos.shape

        frame_feats, text_feats = self.encode_frames(videos), self.encode_texts(texts)
        frame_features = frame_features / frame_features.norm(dim=1, keepdim=True)
        texts_features = texts_features / texts_features.norm(dim=1, keepdim=True)

        text_feat = self.fusion_mlp(text_feats)

        attended_feats = self.lgattn(frame_feats, text_feat)

        attended_feats += self.pe(f, device=videos.device)
        score = self.frame_scoring_transformer(attended_feats, attended_feats)
        return score
    
    @torch.no_grad
    def encode_frames(self, videos):
        frames = rearrange(videos, 'b c f h w -> (b f) c h w').to(torch.contiguous_format)
        features = self.clip_model.encode_image(frames)
        features = rearrange(features, '(b f) c -> b f c')
        return features
    
    @torch.no_grad
    def encode_texts(self, texts):
        return self.clip_model.encode_text(texts)
