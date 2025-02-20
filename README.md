# How to Backdoor Consistency Models?

Pytorch implementation for [*How to Backdoor Consistency Models?*](https://arxiv.org/abs/2410.19785). The paper has been accepted to PAKDD 2025 Special Session.

The implementation is based on the code from [ECT](https://github.com/locuslab/ect) and [VillanDiffusion](https://github.com/IBM/VillanDiffusion).

## Environment

Create a new Conda environment and install PyTorch:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install other libraries:

```bash
pip install click imageio-ffmpeg pyspng tqdm requests psutil scipy datasets matplotlib joblib
```

## Datasets

Prepare the dataset as described [here](https://github.com/NVlabs/edm?tab=readme-ov-file#preparing-datasets). Put the dataset under the folder `datasets`

## Training

For CIFAR10 dataset:

```bash
python ct_train.py --outdir=ct-runs --data=datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp --metrics=fid50k_full --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl --duration=25.6 --tick=12.8 --double=250 --batch=128 --lr=0.0001 --optim=RAdam --dropout=0.2 --augment=0.0 --seed 15 --backdoor --trigger NOISE --target HAT --poison_rate 0.1
```

`--backdoor`: backdoor training, clean consistency models training otherwise.

For FFHQ dataset:

```bash
torchrun --standalone --nproc_per_node=4 ct_train.py --outdir=ct-runs --data=datasets/ffhq-64x64.zip --cond=0 --arch=ddpmpp --metrics=fid50k_full --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl --duration=25.6 --tick=12.8 --double=250 --batch=256 --cres=1,2,2,2 --lr=2e-4 --optim=Adam --dropout=0.05 --augment=0 --seed 15 -q 4 --mean -1.0 --std 1.4 --lr_ref_ticks 500 --ecm_loss_weight --backdoor --trigger NOISE --target HAT --poison_rate 0.1
```

The results are saved in a folder under `ct-runs/`.

## Citation

```bibtex
@article{wang2024backdoor,
  title={How to Backdoor Consistency Models?},
  author={Wang, Chengen and Kantarcioglu, Murat},
  journal={arXiv preprint arXiv:2410.19785},
  year={2024}
}
```
