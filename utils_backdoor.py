import os
from training.dataset_backdoor import DatasetLoader
import torch


def repeat_img(target, img_shape):
    repeat_times = (img_shape[0], *([1] * len(img_shape[1:])))
    # target_grid = target.repeat(repeat_times).to(clean.device)
    target_grid = target.repeat(repeat_times)

    return target_grid

def calc_mse(clean, target):
    loss = torch.nn.MSELoss(reduction='none')

    target_grid = repeat_img(target, clean.shape)
    assert clean.shape == target_grid.shape
    # mse = loss(clean, target_grid).mean(dim=[i for i in range(1, len((clean.shape)))])
    mse = loss(clean, target_grid).mean().item()

    return mse

def get_data_loader(config):
    dataset = config.data.split('/')[1].split('-')[0].upper()
    # if dataset != 'CIFAR10':
    #     print(f'*** dataset name is {dataset}, changed to CELEBA ***')
    #     dataset = 'CELEBA'
    config.dataset = dataset
    ds_root = os.path.join(config.dataset_path)

    vmin: float = -1.0
    vmax: float = 1.0
    if config.sde_type == 'sde_vp' or config.sde_type == 'sde_ldm':
        vmin, vmax = -1.0, 1.0
    elif config.sde_type == 'sde_ve':
        vmin, vmax = 0.0, 1.0
    else:
        raise NotImplementedError(f"sde_type: {config.sde_type} isn't implemented")

    if hasattr(config, 'R_trigger_only'):
        dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch, vmin=vmin,
                            vmax=vmax).set_poison(trigger_type=config.trigger, target_type=config.target,
                                                  clean_rate=config.clean_rate, poison_rate=config.poison_rate,
                                                  ext_poison_rate=config.ext_poison_rate).prepare_dataset(
            mode=config.dataset_load_mode, R_trigger_only=config.R_trigger_only)
    else:
        dsl = DatasetLoader(root=ds_root, name=config.dataset, batch_size=config.batch, vmin=vmin,
                            vmax=vmax).set_poison(trigger_type=config.trigger, target_type=config.target,
                                                  clean_rate=config.clean_rate, poison_rate=config.poison_rate,
                                                  ext_poison_rate=config.ext_poison_rate).prepare_dataset(
            mode=config.dataset_load_mode)
    print(f"datasetloader len: {len(dsl)}")
    return dsl

