import numpy as np

import torch

def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    length: int, length of the 1D position sequence
    return:
    pos_embed: [length, embed_dim] or [1+length, embed_dim] (w/ or w/o cls_token)
    """
    pos = np.arange(length, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_array(embed_dim, pos)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_array(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed_1d(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        # Calculate original and new length for the position embedding
        orig_length = pos_embed_checkpoint.shape[-2] - num_extra_tokens
        new_length = num_patches

        if orig_length != new_length:
            print(f"Position interpolate from {orig_length} to {new_length}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            
            # Reshape and interpolate position tokens
            pos_tokens = pos_tokens.reshape(-1, orig_length, embedding_size).permute(0, 2, 1)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_length,), mode='linear', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 1).flatten(1, 2)
            
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed