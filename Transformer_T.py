import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, repeat
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 qkv_bias=False,
                 qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # todo NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # embed_dim = 768
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(256)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TempTransformer(nn.Module):
    def __init__(self, num_frame, num_joints=95, embed_dim_ratio=256, depth=2,
                 num_heads=2, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):

        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints

        self.num_frames = num_frame
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Temporal_norm = norm_layer(embed_dim_ratio)
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)
        self.to_alpha = nn.Linear(embed_dim_ratio*2, 1)

    def forward_features(self, x):
        x = rearrange(x, 'b f p c -> (b p) f c')
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)   # (batch, 323, 256)
        x_b = self.weighted_mean(x)     # B (batch, 1, 256)

        alpha_i = []
        for i in range(self.num_frames):
            a_i = x[:, i, :]
            a_i = a_i.unsqueeze(1)   # (batch, 1, 256)
            a_i = torch.cat([a_i, x_b], dim=2)
            a_i = self.to_alpha(a_i)     # (batch, 1, 1)
            alpha_i.append(a_i)
        alpha_i = torch.stack(alpha_i, dim=1)  # (batch, 323, 1, 1)
        x = x.unsqueeze(2)     # x:(batch, 323, 1, 256)
        x = x.mul(alpha_i)   # (batch, 323, 1, 256)
        x = x.sum(1)    # (batch, 1, 256)
        alpha_i = alpha_i.sum(1)    # (batch, 1, 1)
        x = x.div(alpha_i)    # (batch, 1, 256)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = rearrange(x, 'b c cls -> b (c cls)')

        return x


# if __name__ == '__main__':
#     x = torch.rand(2, 323, 1, 256)
#     a = TempTransformer(323)
#     b = a.forward(x)
#     print(b.shape)
