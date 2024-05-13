# pretrain

### base
```
python submitit_pretrain.py \
    --norm_loss \
    --device cuda:${id}
```

### freezing embeddings
```
python submitit_pretrain.py \
    --ablation \
    --norm_loss \
    --frozen_embed \
    --device cuda:${id}
```

### change patch size
```
python submitit_pretrain.py \
    --ablation \
    --norm_loss \ # if path size = 1, remove this line
    --patch_size ${patch_szie} \
    --seq_len ${seq_len} \ # seq_len = 100 * patch_size
    --embed_dim ${embed_dim} \ # {patch_size}-{embed_dim}: 1-16, 2-32, 4-64, 8-128, 16-256
    --device cuda:${id}
```

# finetune

### base
```
python submitit_finetune.py \
    --finetune ./checkpoint/pretrain/21204/checkpoint-40.pth \
    --dataset ${dataset} \
    --sample_rate ${sample_rate} \
    --device cuda:${id}
```

### change patch size
```
python submitit_finetune.py \
    --ablation \
    --finetune ./checkpoint/ablation/path_size/1/pretrain/20032/checkpoint-40.pth \
    --dataset ${dataset} \
    --sample_rate ${sample_rate} \
    --patch_size ${patch_szie} \
    --seq_len ${seq_len} \ # seq_len = 100 * patch_size
    --embed_dim ${embed_dim} \ # {patch_size}-{embed_dim}: 1-16, 2-32, 4-64, 8-128, 16-256
    --device cuda:${id}
```