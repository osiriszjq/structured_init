# Structured Initialization for Attention in Vision Transformers

This repository contains an implementation of ["Structured Initialization for Attention in Vision Transformers"](https://arxiv.org/abs/2404.01139).

🔎 Check out [this repository](https://github.com/osiriszjq/impulse_init) for previous version without using `timm` framework.

## Code overview

We added our structured initialization in `vision_transformer.py`. We trained Transformers using the `timm` framework, which we copied from [here](http://github.com/rwightman/pytorch-image-models).


Inside `pytorch-image-models`, we have made the following modifications, which can also be found in [this commit](https://github.com/osiriszjq/structured_init/commit/9c2d0e0d10ce491e720533e24a56aa2063e40211)

- Added our structured initialization to `timm/models/vision_transformer.py`
- Added other initializaitons to `timm/models/convmixer.py`
- Modified other supporting files
  - added `SVHN` dataset in `timm/data/dataset_factory.py`
  - added `SyntaxError` in `timm/utils/misc.py`


## Training
The ViT models are just trained in a normal way. However, you may need to specify which initialization strategy you want to use. Our structured initialization can **only** work with *global pooling* instead of *cls token*:

### Small Dataset

```
python train.py [/path/to/your/dataset]
    --dataset torch/[your/dataset/name]
    --dataset-download 
    --num-classes 10 
    --input-size 3 32 32 
    --mean [dataset/mean]
    --std [dataset/std]
    --model vit_tiny_patch2_32 
    --model-kwargs embed_dim=192 depth=12 num_heads=3 weight_init=[initialization/method] class_token=False no_embed_class=Ture sin_pe=True
    --gp avg 
    -b 512 
    -j 8 
    --opt adamw 
    --epochs 200 
    --sched cosine 
    --lr 0.001 
    --min-lr 0.000001 
    --warmup-epochs 10 
    --opt-eps 1e-3 
    --clip-grad 3.0 
    --weight-decay 0.01 
    --amp 
    --scale 1.0 1.0 
    --ratio 1.0 1.0 
    --crop-pct 1.0 
    --reprob 0.0 
    --aa rand-m9-mstd0.5-inc1
```

### ImageNet-1k

```
sh distributed_train.sh 8 [/path/to/your/ImageNet-1k]
    --train-split train 
    --val-split val
    --model vit_tiny_patch16_224
    --model-kwargs weight_init=[initialization/method] class_token=False no_embed_class=Ture sin_pe=True
    --gp avg 
    -b 512
    -j 10 
    --opt adamw 
    --epochs 300 
    --sched cosine
    --amp 
    --input-size 3 224 224
    --lr-base 0.004
    --lr-base-size 4096
    --weight-decay 0.05
    --aa rand-m9-mstd0.5-inc1
    --cutmix 0.8
    --mixup 1.0 
    --reprob 0.25 
    --remode pixel 
    --num-classes 1000 
    --warmup-epochs 50
```

You can replace `[initialization/method]` with ` `, `skip`, `mimetic0.7_0.7`, `impulse3` or `impulse5`.

## Citation
```
@article{zheng2024structured,
  title={Structured Initialization for Attention in Vision Transformers},
  author={Zheng, Jianqiao and Li, Xueqian and Lucey, Simon},
  journal={arXiv preprint arXiv:2404.01139},
  year={2024}
}
```