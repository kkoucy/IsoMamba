import os
import torch
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import torch.nn as nn
from natsort import natsorted
import torch.nn.functional as F
from pytorch_msssim import ssim
from basicsr.models.archs.clusterM_arch import ClusterMamba
from skimage import img_as_ubyte
import time
from PIL import Image
parser = argparse.ArgumentParser(description='Underwater Image Enhancement using WaterMamba')
parser.add_argument('--yaml', default="./options/UIE.yml", type=str, help='Path to yaml')

parser.add_argument('--input_dir', default='', type=str,
                    help='Directory of validation input images') #! input
parser.add_argument('--target_dir', default='', type=str,
                    help='Directory of validation target images') #! test

parser.add_argument('--result_dir', default="", type=str, 
                    help='Directory for results') #! result
parser.add_argument('--weights',default=''
                    , type=str, help='Path to weights')
args = parser.parse_args()
####### Load yaml #######
yaml_file = args.yaml
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')

##########################
model_restoration = ClusterMamba(**x['network_g'])   #! net

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint["params"])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

factor = 8
datasets = ['EUVP']

import cv2
import numpy as np

def load_img(path):
    # 以 BGR 读取，再转 RGB
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.float32(img) / 255.0


def save_img(path: str, img) -> str:
    """
    Save an image array or PIL Image to disk.

    Args:
        path (str): Destination file path, e.g. ".../xxx.png".
        img (np.ndarray | PIL.Image.Image): Image data.
            - np.ndarray: shapes (H,W), (H,W,3), (H,W,4) or (C,H,W)
            - dtype float in [0,1] or [0,255], or uint8
            - PIL.Image.Image also accepted

    Returns:
        str: The path that was saved to.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    if isinstance(img, Image.Image):
        img.save(path)
        return path

    arr = np.asarray(img)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC

    if np.issubdtype(arr.dtype, np.floating):
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = arr.round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    Image.fromarray(arr).save(path)
    return path



for dataset in datasets:
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    inp_dir = args.input_dir
    target_dir = args.target_dir
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    SSIM = []
    PSNR = []
    inference_times = []

    with torch.no_grad():
        for file_ in tqdm(files):
            img = load_img(file_)
            target = np.float32(load_img(os.path.join(target_dir, file_.split('/')[-1])))

            img = torch.from_numpy(img).permute(2, 0, 1)
            target = torch.from_numpy(target).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()
            target_ = target.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            start_time = time.time()

            restored = model_restoration(input_)

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            inference_times.append(elapsed_time)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            output = restored.clamp_(0, 1)

            
            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target_)).item()
            ssim_val = ssim(output, target_, data_range=1.1, size_average=False).item()
            # print(file_)
            # print(psnr_val)


            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')),
                           img_as_ubyte(restored))

            PSNR.append(psnr_val)
            SSIM.append(ssim_val)


        print('final PSNR:{:.4f}'.format(np.mean(PSNR)), 'final SSIM:{:.4f}'.format(np.mean(SSIM)))
        print('Average inference time per image: {:.4f} seconds'.format(np.mean(inference_times)))
