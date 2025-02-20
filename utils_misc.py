import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from torchvision.utils import make_grid
from torchvision.io import read_image
import torch
from torchvision import transforms
import json
import re

dpi = 96
if not os.path.exists('figures'):
    os.mkdir('figures')

def save_noise(noise=None, size=32):
    if noise is None:
        noise = torch.randn(size=(3, size, size))
    noise = (noise - noise.min()) / (noise.max() - noise.min())  # Normalize to [0, 1]
    noise = (noise * 255).numpy().astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    noise_image = np.transpose(noise, (1, 2, 0))  # Shape becomes (32, 32, 3)
    image = Image.fromarray(noise_image)
    image.save('gaussian_noise.png')
    image.save('gaussian_noise.pdf')

def save_img(img):
    # img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    img = (img * 255).numpy().astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    img = np.transpose(img, (1, 2, 0))  # Shape becomes (128, 128, 3)
    image = Image.fromarray(img)
    image.save('torch_image.png')
    image.save('torch_image.pdf')

def plot_mse(file, args=None):
    my_data = np.genfromtxt(file, delimiter='\t')
    plt.plot(my_data[:, 0], my_data[:, 1])
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('tick')
    plt.ylabel('MSE')
    plt.title(f'MSE vs ticks: trigger {args.trigger}, target {args.target}, poison rate {args.poison_rate}')
    plt.show()

def plot_img_sequence(folder, grid_h=2, grid_w=3, backdoor=False, step=1):
    idx = range(0, grid_h * grid_w * step, step)
    images = []
    s_bd = '_backdoor' if backdoor else ''
    for i in idx:
        fp = f'{folder}/{i:06d}{s_bd}.png'
        img = Image.open(fp) # PngImageFile
        images.append(img)
    images = np.stack(images) # ndarray
    _, H, W, C = images.shape
    images = images.reshape(grid_h, grid_w, H, W, C)
    images = images.transpose((0, 2, 1, 3, 4))
    images = images.reshape(grid_h * H, grid_w * W, C)
    Image.fromarray(images, 'RGB').save(f'{folder}/img_seq{s_bd}.png')

def infer_dataset(folder):
    dataset = re.search(r'-([a-zA-Z0-9]+)-', folder).group(1)
    trigger, target = None, None
    m = re.findall(r'_([A-Za-z0-9]+)', folder)
    if m:
        trigger = '_'.join(m[:-1])
        target = m[-1]
    return dataset, trigger, target

def show_grid(folder, grid_h=2, grid_w=3, backdoor=False, step=1, img_num=1, img_size=32):
    dataset, trigger, target = infer_dataset(folder)

    idx = range(0, grid_h * grid_w * step, step)
    images = []
    s_bd = '_backdoor' if backdoor else ''
    for i in idx:
        fp = f'{folder}/{i:06d}{s_bd}.png'
        img = read_image(fp)[:, :img_size * img_num, :img_size * img_num]
        images.append(img)
    images = torch.stack(images)
    images = make_grid(images, nrow=grid_w, padding=2, pad_value=0)
    images = transforms.ToPILImage()(images)
    # images.show()
    s_backdoor = 'backdoor' if backdoor else 'clean'
    plt.figure(figsize=(6,2))
    plt.title(f'{s_backdoor} image sampling')
    plt.axis('off')
    # plt.yticks([])
    plt.imshow(images)
    plt.savefig(f'figures/grid_{s_backdoor}-{dataset}-{trigger}-{target}.pdf', bbox_inches='tight') # tight -> less blank space
    plt.show()

def show_grids(folder, grid_h=2, grid_w=3, step=1, img_num=2, img_size=32, finetune='DM', pad_value=255):
    dataset, trigger, target = infer_dataset(folder)

    idx = range(0, grid_h * grid_w * step, step)
    images, images_backdoor = [], []
    for i in idx:
        fp = f'{folder}/{i:06d}.png'
        img = read_image(fp)[:, :img_size * img_num, :img_size * img_num]
        images.append(img)

        fp = f'{folder}/{i:06d}_backdoor.png'
        img_backdoor = read_image(fp)[:, :img_size * img_num, :img_size * img_num]
        images_backdoor.append(img_backdoor)

    images = torch.stack(images)
    images = make_grid(images, nrow=grid_w, padding=2, pad_value=pad_value)
    images = transforms.ToPILImage()(images)

    images_backdoor = torch.stack(images_backdoor)
    images_backdoor = make_grid(images_backdoor, nrow=grid_w, padding=2, pad_value=pad_value)
    images_backdoor = transforms.ToPILImage()(images_backdoor)

    # fig, axs = plt.subplots(2, 1) # use default figsize
    scale = img_size // 32
    fig, axs = plt.subplots(2, 1, figsize=(12 * scale, 4.2 * scale)) # adjust figsize based on dpi and image resolution
    # fig, axs = plt.subplots(2, 1, figsize=(1058/dpi, 134/dpi), dpi=dpi)
    axs[0].imshow(images)
    axs[0].set_title('clean image sampling', fontsize=16)
    axs[0].axis('off')
    axs[0].yaxis.set_visible(False)

    axs[1].imshow(images_backdoor)
    axs[1].set_title('backdoor image sampling', fontsize=16)
    axs[1].axis('off')
    axs[1].yaxis.set_visible(False)

    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.4)
    plt.savefig(f'figures/grids-{dataset}-{trigger}-{target}-{finetune}-{step}.pdf', bbox_inches='tight', pad_inches=0.02, dpi=dpi) # dpi, figsize closely related, dpi=240 for default figsize
    plt.savefig(f'figures/grids-{dataset}-{trigger}-{target}-{finetune}-{step}.png', bbox_inches='tight', pad_inches=0.02, dpi=dpi)
    plt.show()

def get_fid(folder, metric='fid50k_full', tick=240, two_step=False):
    s_2_step = 'two_step_' if two_step else ''
    with open(f'{folder}/metric-{s_2_step}{metric}.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    data = np.array(data)
    for cur in data:
        matches = re.findall(r'\d+', cur['snapshot_pkl'])
        if matches: ## else ...latest.pkl
            cur_tick = int(matches[0])
            if cur_tick == tick:
                return cur['results']['fid50k_full']
    raise Exception

def infer_folder(num):
    folders = os.listdir('ct-runs')
    for folder in folders:
        if re.match(fr'^0*{num}-', folder):
            break
    return f'ct-runs/{folder}'

def get_mse(folder, tick=240):
    data = np.genfromtxt(f'{folder}/mse.csv', delimiter='\t')
    row = np.where(data[:,0] == tick)[0]
    if len(row) > 0:
        return data[row, 1][0]
    raise Exception

def get_fids(folders, tick=240, two_step=True, fid_only=False, finetune='DM'):
    fids, MSEs = [], []
    dataset, trigger, target = None, None, None
    for num in folders:
        folder = infer_folder(num)
        pr = float(re.findall(fr'-pr([\d.]+)', folder)[-1])
        if trigger is None and pr > 0:
            dataset, trigger, target = infer_dataset(folder)
        fid = get_fid(folder, tick=tick, two_step=two_step)
        fids.append((pr, fid))
        mse = get_mse(folder, tick=tick)
        MSEs.append((pr, mse))

    # print(sorted(fids))
    fids = sorted(fids, key=lambda x: x[0])
    fids = np.array(fids).astype('float32')
    MSEs = sorted(MSEs, key=lambda x: x[0])
    MSEs = np.array(MSEs).astype('float32')

    x, y = fids[:, 0], fids[:, 1]
    if fid_only:
        plt.plot(x, y, marker='o', linestyle='-', color='b', label='Data points')
        # plt.bar(x, y, width=0.04)
        plt.xlabel('poison rate')
        plt.ylabel('FID')
        plt.ylim(ymin=0)
        plt.savefig('figures/FID_poison-rate.pdf',  dpi=300, bbox_inches='tight')
        plt.show()
    else:
        fig, ax1 = plt.subplots()
        x2, y2 = MSEs[:, 0], MSEs[:, 1]
        assert (x == x2).all()
        ax1.plot(x, y, 'b-', label='FID')
        ax1.set_ylabel('FID', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        # ax1.grid(color='lightgray', linestyle=':', linewidth=0.3)
        ax1.grid()
        # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=1)  # Add legend for the first plot
        ax2 = ax1.twinx()
        ax2.plot(x, y2, 'r-', label='MSE')
        ax2.set_ylabel('MSE', color='r')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='r')
        # ax2.grid(color='lightgray', linestyle=':', linewidth=0.3)
        ax2.grid()
        # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=1)  # Add legend for the second plot
        ax1.set_xlabel('poison rate')
        plt.title(f'Dataset {dataset.upper()}, trigger {trigger}, target {target}, fine-tuned on {finetune}')
        # plt.legend(loc='best')
        # plt.text(0.1, 2.2, 'CIFAR10')
        plt.savefig(f'figures/FID_MSE-{dataset}-{trigger}-{target}-{finetune}.pdf', dpi=300, bbox_inches='tight')
        plt.show()

def get_MSEs(folders, tick=240):
    MSE = []
    for num in folders:
        folder = infer_folder(num)
        pr = re.findall(fr'-pr([\d.]+)', folder)[-1]
        cur_mse = get_mse(folder, tick=tick)
        MSE.append((pr, cur_mse))

    MSE = sorted(MSE, key=lambda x: x[0])
    MSE = np.array(MSE).astype('float32')
    x, y = MSE[:, 0], MSE[:, 1]
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Data points')
    plt.xlabel('poison rate')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.ylim(ymin=0)
    # plt.grid(color='lightgray', linestyle=':', linewidth=0.3)
    plt.grid(True)
    plt.savefig('figures/MSE_poison-rate.pdf',  dpi=300, bbox_inches='tight')
    plt.show()

def draw_square(bg_size=512, square_size=64, margin=20):
    image = np.zeros((bg_size, bg_size, 3), dtype=np.uint8)
    bottom_right_start = (bg_size - square_size - margin, bg_size - square_size - margin)
    image[bottom_right_start[0]:bottom_right_start[0] + square_size,
    bottom_right_start[1]:bottom_right_start[1] + square_size] = 255
    return image

def img_sched(img_size=512, step=4):
    trigger = draw_square(img_size) / 255
    plt.imshow(trigger)
    plt.show()
    image_path = 'static/penguins.PNG'
    image = Image.open(image_path)
    image = np.array(image) / 255
    plt.imshow(image)
    plt.show()
    target_path = 'static/fedora-hat.png'
    target = mpimg.imread(target_path)[...,:3]
    # target = Image.open(target_path)
    # target = target.convert('RGB')
    # target = np.array(target) / 255
    plt.imshow(target)
    plt.show()
    noise = np.random.rand(img_size, img_size, 3)
    plt.imshow(noise)
    plt.show()
    clean_images, target_images = [image], [target]
    steps = [1, 2, 4, 8, 16, 32, 64, 80]
    print(steps)
    for i in steps:
        noise = np.random.rand(img_size, img_size, 3)
        clean_image = (image + noise * i) / i
        clean_image = clean_image / clean_image.max()
        clean_images.append(clean_image.clip(0, 1))
        target_image = (target + (noise + trigger) * i) / i
        target_image = target_image / target_image.max()
        target_images.append(target_image.clip(0, 1))

    clean_images = torch.tensor(np.stack(clean_images)).permute(0, 3, 1, 2)
    target_images = torch.tensor(np.stack(target_images)).permute(0, 3, 1, 2)
    clean_grid = make_grid(clean_images, nrow=len(clean_images), normalize=True).numpy().transpose((1, 2, 0))
    target_grid = make_grid(target_images, nrow=len(target_images), normalize=True).numpy().transpose((1, 2, 0))
    plt.figure(figsize=(10, 10))
    plt.imshow(clean_grid); plt.axis('off');
    plt.savefig('figures/clean_sched.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.figure(figsize=(10, 10))
    plt.imshow(target_grid); plt.axis('off');
    plt.savefig('figures/target_sched.png', bbox_inches='tight', pad_inches=0)
    plt.show()

from transformers import PreTrainedModel, PretrainedConfig

def torch2hf():
    import pickle

    path = '/home/wangc/lab/ect/finetune'
    torch_model = f'{path}/cm-cifar10-pr01-box14-cat.pkl'

    with open(torch_model, 'rb') as f:
        data = pickle.load(f)
    model = data['ema']

    class MyConfig(PretrainedConfig):
        model_type = 'mymodel'

        def __init__(self, important_param=42, **kwargs):
            super().__init__(**kwargs)
            self.important_param = important_param

    class MyModel(PreTrainedModel):
        config_class = MyConfig

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.model = model

        def forward(self, input):
            return self.model(input)

    config = MyConfig(4)
    model = MyModel(config)

if __name__ == '__main__':

    # folder = '00087-cifar10-32x32-uncond-ddpmpp-ect-RAdam-0.000100-gpus1-batch32-fp32'
    # args = dnnlib.EasyDict(trigger='BOX_14', target='HAT', poison_rate=0.1)
    # folder = f'ct-runs/{folder}'
    # plot_mse(f'{folder}/mse.csv', args=args)

    # plot_img_sequence(folder, backdoor=False, grid_h=10, grid_w=12)
    # plot_img_sequence(folder, backdoor=True)
    # plot_img_sequence(folder, backdoor=True, grid_h=10, grid_w=12)

    folder = '00097-cifar10-32x32-uncond-ddpmpp-ect-RAdam-0.000100-gpus1-batch32-fp32-pr0.1' # DM 3381; slow trained on DM

    # folder = '00096-cifar10-32x32-uncond-ddpmpp-ect-RAdam-0.000100-gpus1-batch32-fp32-pr0.1_NOISE_HAT' # fast trained on DM
    folder = '00070-ffhq-64x64-uncond-ddpmpp-ect-Adam-0.000200-gpus1-batch64-fp32' # DM, 3381
    folder = f'ct-runs/{folder}'
    # show_grids(folder, grid_h=2, grid_w=16, img_num=2, img_size=32, step=1, finetune='DM') # CIFAR10
    show_grids(folder, grid_h=2, grid_w=8, img_num=2, img_size=64, step=10, finetune='DM') # FFHQ

    folder = '00146-cifar10-32x32-uncond-ddpmpp-ect-RAdam-0.000100-gpus1-batch128-fp32-pr0.1_NOISE_HAT' # CM, 3381
    # folder = '00020-ffhq-64x64-uncond-ddpmpp-ect-Adam-0.000200-gpus4-batch128-fp32-pr0.1_NOISE_HAT'## slow trained on CM
    # folder = '00186-ffhq-64x64-uncond-ddpmpp-ect-Adam-0.000200-gpus1-batch64-fp32-pr0.1_NOISE_HAT'
    # folder = '00053-ffhq-64x64-uncond-ddpmpp-ect-Adam-0.000200-gpus3-batch96-fp32-pr0.1_NOISE_HAT'
    folder = '00186-ffhq-64x64-uncond-ddpmpp-ect-Adam-0.000200-gpus1-batch64-fp32-pr0.1_NOISE_HAT' # CM, 3381
    folder = f'ct-runs/{folder}'

    # show_grids(folder, grid_h=2, grid_w=16, img_size=32, img_num=2, step=1, finetune='CM') # CIFAR10
    show_grids(folder, grid_h=2, grid_w=8, img_size=64, img_num=2, step=10, finetune='CM') # FFHQ
    # show_grids(folder, grid_h=2, grid_w=16, img_num=2, step=1, finetune='CM')

    # show_grid(folder, grid_h=2, grid_w=16, img_num=2, step=10)
    # show_grid(folder, grid_h=2, grid_w=16, img_num=2, backdoor=True)
    # print(get_fid(infer_folder(87), metric='two_step_fid50k_full'))

    # folders = [87, 104, 105, 106, 107]
    # folders = [113, 87, 108, 109, 106, 107, 127]  # NOISE->HAT including pr=0.01
    # folders = [113, 87, 108, 109, 106, 107] # NOISE->HAT, DM
    # folders = [113, 126, 129, 131, 132, 133] # BOX_14 -> CAT, DM
    # folders = [113, 134, 135, 136, 137, 138] # GLASSES -> CAT, DM
    # get_fids(folders, tick=240, fid_only=False)

    folders = [164, 147, 148, 149, 150, 151] # NOISE -> HAT, CM
    folders = [164, 152, 153, 154, 156, 165]  # NOISE -> HAT, CM ## 157 abnormal -> 165
    # folders = [164, 158, 159, 160, 161, 162]  # NOISE -> HAT, CM
    # get_fids(folders, tick=250, finetune='CM')

    # get_MSEs(folders, tick=240)

    # save_noise()

    # img_sched()