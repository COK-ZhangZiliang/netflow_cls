import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_

class PatchEmbed1D(nn.Module):
    """ 1D Patch Embedding"""
    def __init__(self, input_length, in_chans, embed_dim, patch_size):
        super().__init__()
        self.input_length = input_length
        self.patch_size = patch_size
        self.num_patches = input_length // patch_size
        self.proj = nn.Conv1d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (N, C, L)
        N, C, L = x.shape
        assert L == self.input_length, f"Input length {L} is not equal to the expected input length {self.input_length}"
        x = self.proj(x)  # (N, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (N, num_pathes, embed_dim)
        return x
    
class VisionTransformer1D(nn.Module):
    """ Vision Transformer with support for 1D data"""
    def __init__(self, seq_len=1000, patch_size=10, in_chans=1, num_classes=2, embed_dim=256, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed1D(seq_len, in_chans, embed_dim, patch_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
