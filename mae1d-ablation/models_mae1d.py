from functools import partial

import torch
from torch import nn

from models_vit1d import PatchEmbed1D
from timm.models.vision_transformer import Block

from util.pos_embed1d import get_1d_sincos_pos_embed

class MaskedAutoencoderViT1D(nn.Module):
    """
    1D Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, input_length=1000, patch_size=10, in_chans=1,
                 embed_dim=256, depth=12, num_heads=8,
                 decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_loss=False):
        super().__init__()

        # MAE1D encoder
        self.input_length = input_length
        self.patch_embed = PatchEmbed1D(input_length, in_chans, embed_dim, patch_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False) # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # MAE1D decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True)  # decoder to patch

        self.norm_loss = norm_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv1d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, seq):
        """
        packet length sequence: (N, 1, L)
        x: (N, N_P, patch_size)
        """
        p = self.patch_embed.patch_size
        assert seq.shape[2] % p == 0, f"Input length {seq.shape[2]} is not divisible by the patch size {p}"

        n_p = seq.shape[2] // p
        x = seq.reshape((seq.shape[0], n_p, p))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, N_P, patch_size)
        packet length sequence: (N, 1, L)
        """
        p = self.patch_embed.patch_size[0]
        l = x.shape[1] * p
        assert l == self.input_length, f"Output length {l} is not equal to the expected output length {self.input_length}"     
        
        seq = x.reshape((x.shape[0], 1, l))
        return seq
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, N_P, D], sequence
        """
        N, N_P, D = x.shape  # batch, length, dim
        len_keep = int(N_P * (1 - mask_ratio))
        
        noise = torch.rand(N, N_P, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, N_P], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, seqs, pred, mask):
        """
        seq: [N, 1, L]
        pred: [N, N_P, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(seqs)
        if self.norm_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, seqs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(seqs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, N_P, p]
        loss = self.forward_loss(seqs, pred, mask)
        return loss, pred, mask