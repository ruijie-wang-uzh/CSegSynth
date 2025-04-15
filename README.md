# CSegSynth: Deep Generative Model-Based Generation of Synthetic Individual-Specific Brain MRI Segmentations

## In this repository, we provide the code for the following paper:

**Deep Generative Model-Based Generation of Synthetic Individual-Specific Brain MRI Segmentations**  
_Ruijie Wang, Luca Rossetto, Susan Mérillat, Christina Röcke, Mike Martin, Abraham Bernstein_

## The repository contains the following files:
- `data_prep.py`: Data preparation script.
- `vae_models.py`: VAE model definition.
- `models.py`: Definition of other models.
- `utils.py`: Utility functions.
- `pretrain_vae.py`: Pre-training the VAE model.
- `finetune_cvae.py`: Fine-tuning the CVAE model.
- `pretrain_gan.py`: Pre-training the GAN model.
- `finetune_cgan.py`: Fine-tuning the CGAN model.
- `pretrain_alpha_wgan.py`: Pre-training the Alpha-WGAN model.
- `finetune_csegsynth.py`: Fine-tuning the CSegSynth model.

Please refer to [https://creatis-myriad.github.io/2024/01/30/brain-imaging-generation.html](https://creatis-myriad.github.io/2024/01/30/brain-imaging-generation.html) for the training of the LDM and CLDM.

## Requirements
[Python 3](https://www.python.org/), [PyTorch](https://pytorch.org/), [MONAI](https://monai.io/index.html)
, [Timm](https://huggingface.co/docs/timm/en/index), [Numpy](https://numpy.org/), [Nibabel](https://nipy.org/nibabel/)
, [Tqdm](https://github.com/tqdm/tqdm)


## Datasets
- AOMIC dataset: https://nilab-uva.github.io/AOMIC.github.io/
- CamCAN dataset: https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/

