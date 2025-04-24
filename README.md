# SatMAE++ Implementation

## Overview

The SatMAE++ framework was introduced in ["Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery"](https://arxiv.org/abs/2403.05419) (Noman et al., 2024). This submodule contains the code, installation and setup guide necessary for running the finetuning part of the framework on the Solafune deforestation drivers competition. I use the pretrained model with weights provided by the authors on GitHub at [techmn/satmae_pp.](https://github.com/techmn/satmae_pp).

<img width="1096" alt="image" src="images/overall_architecture.png">

## Method




## Installation and Setup

1. **Add the forked SatMAE++ repository as a submodule**

   ```bash
   git submodule add https://github.com/auroraingebrigtsen/satmae_pp.git satmae_pp
   cd satmae_pp
    ```
    Ensure dependencies is installed by the global requirements.txt

2. **Download the ViT-Large [pretrained weights](https://huggingface.co/mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel) from hugging face**
    ```bash
    wget -O checkpoint_ViT-L_pretrain_fmow_sentinel.pth \
    https://huggingface.co/mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel/resolve/main/pytorch_model.bin
    ```

3. **Move the weights into the repo**


## Usage
To reproduce the finetuning run the following command

```bash LINUX
python main_finetune_seg.py \
  --device cpu \
  --input_size 1024 \
  --patch_size 16 \
  --nb_classes 5 \
  --path/to/checkpoint_ViT-L_pretrain_fmow_sentinel.pth \
  --epochs 10 \
  --batch_size 8 \
  --mixup 0.0 \
  --cutmix 0.0

  WINDOWS
```


## Citation

```
@inproceedings{satmaepp2024rethinking,
      title={Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery}, 
      author={Mubashir Noman and Muzammal Naseer and Hisham Cholakkal and Rao Muhammad Anwar and Salman Khan and Fahad Shahbaz Khan},
      year={2024},
      booktitle={CVPR}
}
```

> techmn. _satmae_pp_. GitHub. 2025. https://github.com/techmn/satmae_pp
