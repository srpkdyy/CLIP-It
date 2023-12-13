from typing import Dict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torchvision.transforms import Lambda

from einops import rearrange
from einops.layers.torch import Rearrange


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
    def __init__(self, dim):
        super().__init__()

        self.scale = dim ** 0.5
        self.fill_val = torch.tensor(-float('inf'))

    def forward(self, q, k, v, mask=None):
        qk = torch.einsum('...id,...jd->...ij', q, k)
        scaled_qk = qk / self.scale

        if mask is not None:
            scaled_qk.masked_fill_(mask, self.fill_val)

        attn = F.softmax(scaled_qk, dim=-1)

        out = einsum('...ij,...jd->...id', attn, v)
        return out


class LangGuidedAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads

        kv_dim = dim // n_heads
        proj_dim = kv_dim * n_heads

        self.to_heads = Rearrange('b l (h d) -> b h l d', h=n_heads)

        self.q_proj = nn.Linear(dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(dim, proj_dim, bias=False)
        self.v_proj = nn.Linear(dim, proj_dim, bias=False)

        self.attention = ScaledDotProductAttention(kv_dim)

        self.out_proj = nn.Linear(proj_dim, dim, bias=False)

    def forward(self, q, kv, mask=None):
        q = self.to_heads(self.q_proj(q))
        k = self.to_heads(self.k_proj(kv))
        v = self.to_heads(self.v_proj(kv))

        attn = self.attention(q, k, v, mask=mask)
        attn = rearrange(attn, 'b h l d -> b l (h d)')

        out = self.out_proj(attn)
        return out


class CLIP_IT(nn.Module):
    def __init__(self,
                 clip_model_name: str,
                 num_sentences: int = 7,
                 lgattn_heads: int = 4,
                 transformer_kwargs: Dict = {}
                 ):
        super(CLIP_IT, self).__init__()

        self.clip_model, self.image_preprocess = clip.load(clip_model_name)
        self.image_preprocess.transforms[2] = Lambda(lambda x: x / 255.)
        self.image_preprocess.transforms.pop(3)
        self.dim = self.clip_model.visual.output_dim

        self.fusion_mlp = nn.Linear(self.dim * num_sentences, self.dim)
        self.num_sentences = num_sentences

        self.lgattn = LangGuidedAttention(self.dim, n_heads=lgattn_heads)

        self.pe = PositionalEncoding(self.dim, max_pos=4096)
        self.frame_scoring_transformer = nn.Transformer(self.dim, **transformer_kwargs)
        self.fc = nn.Linear(self.dim, 1)

    def forward(self, videos, texts):
        assert len(texts) == self.num_sentences
        b, c, f, h, w = videos.shape

        frame_feats, text_feats = self.encode_frames(videos), self.encode_texts(texts)
        frame_feats = frame_feats / frame_feats.norm(dim=1, keepdim=True)
        texts_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

        text_feats = rearrange(text_feats, 'b s c -> b 1 (s c)')
        text_feat = self.fusion_mlp(text_feats)

        attended_feats = self.lgattn(frame_feats, text_feat)

        attended_feats += self.pe(f, device=videos.device)
        score = self.frame_scoring_transformer(attended_feats, attended_feats)
        score = self.fc(score).squeeze(-1)
        return score
    
    @torch.no_grad
    def encode_frames(self, videos):
        b, *_ = videos.shape
        frames = rearrange(videos, 'b c f h w -> (b f) c h w')
        frames = self.image_preprocess(frames).to(memory_format=torch.contiguous_format)
        features = self.clip_model.encode_image(frames)
        features = rearrange(features, '(b f) c -> b f c', b=b)
        return features
    
    @torch.no_grad
    def encode_texts(self, texts):
        tokens = clip.tokenize(texts).to(self.device)
        return self.clip_model.encode_text(tokens).unsqueeze(0)
    
    @property
    def device(self):
        return self.clip_model.visual.conv1.weight.device
