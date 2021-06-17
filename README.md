## Uformer - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2106.03106">Uformer</a>, Attention-based Unet, in Pytorch. This repository will be geared towards use in a project <a href="https://github.com/lucidrains/ddpm-proteins">for learning protein structures</a>. Specifically, it will include the ability to condition on time steps (needed for DDPM), as well as 2d relative positional encoding using rotary embeddings.

## Citations

```bibtex
@misc{wang2021uformer,
    title   = {Uformer: A General U-Shaped Transformer for Image Restoration}, 
    author  = {Zhendong Wang and Xiaodong Cun and Jianmin Bao and Jianzhuang Liu},
    year    = {2021},
    eprint  = {2106.03106},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
