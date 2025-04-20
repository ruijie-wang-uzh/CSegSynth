# CSegSynth: Deep Generative Model-Based Generation of Synthetic Individual-Specific Brain MRI Segmentations

### In this repository, we provide the code for the following paper:

**[Deep Generative Model-Based Generation of Synthetic Individual-Specific Brain MRI Segmentations](https://arxiv.org/abs/2504.12352)**  
_Ruijie Wang, Luca Rossetto, Susan Mérillat, Christina Röcke, Mike Martin, Abraham Bernstein_

### The repository contains the following files:
- `data_prep.py`: Data preparation script.
- `vae_models.py`: Variational Autoencoder (VAE) model definition.
- `models.py`: Definition of other models.
- `utils.py`: Utility functions.
- `pretrain_vae.py`: Pre-training the VAE model.
- `finetune_cvae.py`: Fine-tuning the Conditional Variational Autoencoder (C-VAE) model.
- `pretrain_gan.py`: Pre-training the Generative Adversarial Network (GAN) model.
- `finetune_cgan.py`: Fine-tuning the Conditional Generative Adversarial Network (C-GAN) model.
- `pretrain_alpha_wgan.py`: Pre-training the Alpha-Generative Adversarial Network (Alpha-GAN) model.
- `finetune_csegsynth.py`: Fine-tuning the Conditional Segmentation Synthesis (CSegSynth) model.

Please refer to [https://creatis-myriad.github.io/2024/01/30/brain-imaging-generation.html](https://creatis-myriad.github.io/2024/01/30/brain-imaging-generation.html) for the training of the Latent Diffusion Model (LDM) and the Conditional Latent Diffusion Model (C-LDM).

### Configurations

Please refer to [configurations.pdf](https://github.com/ruijie-wang-uzh/CSegSynth/blob/main/configurations.pdf) for more detailed information about the used machines, training configurations, and running time of each training process.

### Requirements

Please set up a [Python 3](https://www.python.org/) environment with the following packages installed:
[PyTorch](https://pytorch.org/), [MONAI](https://monai.io/index.html)
, [Timm](https://huggingface.co/docs/timm/en/index), [Numpy](https://numpy.org/), [Nibabel](https://nipy.org/nibabel/)
, [Tqdm](https://github.com/tqdm/tqdm).


### Datasets
- AOMIC dataset: https://nilab-uva.github.io/AOMIC.github.io/
- CamCAN dataset: https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/


### Citation

```bibtex
@misc{wang2025deepgenerativemodelbasedgeneration,
      title={Deep Generative Model-Based Generation of Synthetic Individual-Specific Brain MRI Segmentations}, 
      author={Ruijie Wang and Luca Rossetto and Susan Mérillat and Christina Röcke and Mike Martin and Abraham Bernstein},
      year={2025},
      eprint={2504.12352},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC},
      url={https://arxiv.org/abs/2504.12352}, 
}
```


