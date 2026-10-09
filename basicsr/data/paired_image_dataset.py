from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2

class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

import os
from pathlib import Path
from torch.utils import data


from pathlib import Path

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in _IMG_EXTS

def _build_gt_map(gt_folder: str):
    gt_map = {}
    for p in Path(gt_folder).iterdir():
        if _is_image_file(p):
            gt_map[p.stem] = str(p)
    return gt_map

def _collect_images_under(folder: Path, recursive: bool = True):
    it = folder.rglob("*") if recursive else folder.glob("*")
    out = [str(p) for p in it if _is_image_file(p)]
    out.sort()
    return out

def _collect_lq_paths(input_root: str, parts=None, recursive: bool = True):
    """
    parts is None or empty: scan input_root directly
    parts is list/tuple: scan input_root/part for each part
    """
    input_root = Path(input_root)

    if not parts:
        return _collect_images_under(input_root, recursive=recursive)

    lq_paths = []
    for part in parts:
        part_dir = input_root / str(part)
        if not part_dir.exists():
            continue
        lq_paths.extend(_collect_images_under(part_dir, recursive=recursive))

    lq_paths.sort()
    return lq_paths

def paired_paths_from_sots_folders(input_root: str, gt_root: str, parts=None, recursive: bool = True):
    """
    input: 0025_0.8_0.1.jpg
    gt:    0025.jpg (or other ext)
    match rule: take prefix before first underscore as key
    """
    gt_map = _build_gt_map(gt_root)
    lq_paths = _collect_lq_paths(input_root, parts=parts, recursive=recursive)

    paths = []
    missing = []
    for lq_path in lq_paths:
        stem = Path(lq_path).stem
        key = stem.split("_", 1)[0]
        gt_path = gt_map.get(key)
        if gt_path is None:
            missing.append((lq_path, key))
            continue
        paths.append({"lq_path": lq_path, "gt_path": gt_path})

    if missing:
        show_n = min(20, len(missing))
        msg = "\n".join([f"[{i}] key={k}  lq={p}" for i, (p, k) in enumerate(missing[:show_n])])
        raise FileNotFoundError(
            f"Found {len(missing)} input images without GT in {gt_root}.\n"
            f"First {show_n} missing pairs:\n{msg}\n"
            f"GT must contain files named like 0025.jpg (stem matches prefix before underscore)."
        )

    return paths


class Dataset_SotsImage(data.Dataset):
    """
    input_root:
      part1/0025_0.8_0.1.jpg
      part2/0025_0.8_0.2.jpg
      part3/...
      part4/...
    gt_root:
      0025.jpg  (or png, jpeg, ...)
    """

    def __init__(self, opt):
        super(Dataset_SotsImage, self).__init__()
        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None

        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt["dataroot_lq"]

        if self.io_backend_opt["type"] == "lmdb":
            raise NotImplementedError(
                "Dataset_SotsImage currently supports folder/disk style data. "
                "Your setting is lmdb."
            )

        parts = opt.get("parts", None)  # 不给就是 None
        recursive = opt.get("recursive", True)

        self.paths = paired_paths_from_sots_folders(self.lq_folder, self.gt_folder, parts=parts, recursive=recursive
                )

        if self.opt["phase"] == "train":
            self.geometric_augs = opt.get("geometric_augs", True)

    def __getitem__(self, index):
        if self.file_client is None:
            # keep same style as your original dataset
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        scale = self.opt["scale"]
        index = index % len(self.paths)

        gt_path = self.paths[index]["gt_path"]
        img_bytes = self.file_client.get(gt_path, "gt")
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception as e:
            raise Exception(f"gt path {gt_path} not working, error={e}")

        lq_path = self.paths[index]["lq_path"]
        img_bytes = self.file_client.get(lq_path, "lq")
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except Exception as e:
            raise Exception(f"lq path {lq_path} not working, error={e}")

        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]

            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.paths)





import os
from pathlib import Path
from torch.utils import data

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in _IMG_EXTS


def paired_paths_from_gopro_train(train_root: str, blur_dirname="blur", sharp_dirname="sharp", recursive=True):
    """
    train_root/
      sceneA/blur/xxxx.png
      sceneA/sharp/xxxx.png
      sceneB/blur/xxxx.png
      sceneB/sharp/xxxx.png

    Pair rule:
      for each file under */blur/, match */sharp/<same relative path>.
      if exact match does not exist, try same stem with any extension in sharp folder.
    """
    train_root = Path(train_root)
    if not train_root.exists():
        raise FileNotFoundError(f"train_root not found: {train_root}")

    blur_roots = []
    for sub in train_root.iterdir():
        if not sub.is_dir():
            continue
        b = sub / blur_dirname
        s = sub / sharp_dirname
        if b.exists() and s.exists():
            blur_roots.append((b, s))

    if len(blur_roots) == 0:
        raise FileNotFoundError(
            f"No valid subfolders found under {train_root}. Expect */{blur_dirname} and */{sharp_dirname}."
        )

    paths = []
    missing = []

    for blur_root, sharp_root in blur_roots:
        it = blur_root.rglob("*") if recursive else blur_root.glob("*")
        for lq_p in it:
            if not _is_image(lq_p):
                continue

            rel = lq_p.relative_to(blur_root)  # keep sub-structure if any
            gt_p = sharp_root / rel

            if not gt_p.exists():
                # try same stem with different extension inside the same parent folder
                cand_dir = (sharp_root / rel.parent)
                stem = rel.stem
                found = None
                if cand_dir.exists():
                    for ext in _IMG_EXTS:
                        c = cand_dir / f"{stem}{ext}"
                        if c.exists():
                            found = c
                            break
                if found is None:
                    missing.append((str(lq_p), str(gt_p)))
                    continue
                gt_p = found

            paths.append({"lq_path": str(lq_p), "gt_path": str(gt_p)})

    if missing:
        show_n = min(20, len(missing))
        msg = "\n".join([f"[{i}] lq={a} | expected_gt={b}" for i, (a, b) in enumerate(missing[:show_n])])
        raise FileNotFoundError(
            f"Found {len(missing)} blur images without sharp pairs.\nFirst {show_n}:\n{msg}"
        )

    paths.sort(key=lambda d: d["lq_path"])
    return paths


class Dataset_GoproImage(data.Dataset):
    """
    GoPro train dataset: each subfolder contains blur (input) and sharp (gt).
    """

    def __init__(self, opt):
        super(Dataset_GoproImage, self).__init__()
        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None

        # 你只给一个 train 路径时，用 dataroot 这个键更直观
        # 兼容老写法：如果没有 dataroot，就用 dataroot_lq 或 dataroot_gt 当 root
        self.train_root = opt.get("dataroot", None)
        if self.train_root is None:
            self.train_root = opt.get("dataroot_lq", None)
        if self.train_root is None:
            self.train_root = opt.get("dataroot_gt", None)
        if self.train_root is None:
            raise KeyError("Please provide opt['dataroot'] (GoPro train root).")

        blur_dirname = opt.get("blur_dirname", "blur")
        sharp_dirname = opt.get("sharp_dirname", "sharp")
        recursive = opt.get("recursive", True)

        if self.io_backend_opt["type"] == "lmdb":
            raise NotImplementedError("Dataset_GoproImage currently supports disk/folder style data, not lmdb.")

        self.paths = paired_paths_from_gopro_train(
            self.train_root,
            blur_dirname=blur_dirname,
            sharp_dirname=sharp_dirname,
            recursive=recursive
        )

        if self.opt["phase"] == "train":
            self.geometric_augs = opt.get("geometric_augs", True)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        scale = self.opt["scale"]
        index = index % len(self.paths)

        gt_path = self.paths[index]["gt_path"]
        lq_path = self.paths[index]["lq_path"]

        img_bytes = self.file_client.get(gt_path, "gt")
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception as e:
            raise Exception(f"gt path {gt_path} not working, error={e}")

        img_bytes = self.file_client.get(lq_path, "lq")
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except Exception as e:
            raise Exception(f"lq path {lq_path} not working, error={e}")

        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]

            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.paths)



import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

# 这些函数/类你在原工程里应该已经有同名 import 方式
# from basicsr.utils import FileClient, imfrombytes, img2tensor
# from basicsr.data.transforms import padding, paired_random_crop, random_augmentation, normalize


def _scan_images(folder: str, recursive: bool = True):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    p = Path(folder)
    if not p.exists():
        return []
    it = p.rglob("*") if recursive else p.glob("*")
    paths = [str(x) for x in it if x.is_file() and x.suffix.lower() in exts]
    paths.sort()
    return paths


class Dataset_BSDWED_GaussianDenoise(data.Dataset):
    """
    Clean->Noisy denoising dataset (BSD400 + WED).

    opt 里常用字段:
        dataroot_gt (str): root，里面包含 BSD400 和 WED 两个子目录
        io_backend (dict): 例如 {"type": "disk"}
        phase (str): "train" / "val"
        gt_size (int): train 时 crop patch 大小
        geometric_augs (bool): 是否做翻转旋转增强
        sigma_list (list[int]): 默认 [15, 25, 50]
        mean, std: 可选，用于 normalize
        scale: 为了兼容旧接口，默认按 1 处理
    """

    def __init__(self, opt):
        super(Dataset_BSDWED_GaussianDenoise, self).__init__()
        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None

        self.root = opt["dataroot_gt"]
        self.subfolders = opt.get("subfolders", ["BSD400", "WED"])
        self.sigma_list = opt.get("sigma_list", [15, 25, 50])

        self.paths = []
        for name in self.subfolders:
            folder = os.path.join(self.root, name)
            for p in _scan_images(folder, recursive=True):
                self.paths.append({"gt_path": p})

        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found under: {self.root}. Expected subfolders: {self.subfolders}"
            )

        if self.opt.get("phase", "train") == "train":
            self.geometric_augs = opt.get("geometric_augs", True)

    @staticmethod
    def _add_gaussian_noise(img_gt: np.ndarray, sigma: int) -> np.ndarray:
        # img_gt: HWC, float32, range [0,1]
        std = float(sigma) / 255.0
        noise = np.random.normal(0.0, std, img_gt.shape).astype(np.float32)
        img_lq = img_gt + noise
        img_lq = np.clip(img_lq, 0.0, 1.0)
        return img_lq

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        index = index % len(self.paths)
        gt_path = self.paths[index]["gt_path"]

        img_bytes = self.file_client.get(gt_path, "gt")
        try:
            img_gt = imfrombytes(img_bytes, float32=True)  # HWC, BGR, [0,1]
        except Exception:
            raise Exception(f"gt path {gt_path} not working")

        # train augmentation: 在干净图上做 crop/aug，然后再加噪生成 lq
        if self.opt.get("phase", "train") == "train":
            gt_size = self.opt["gt_size"]

            # padding / crop 复用 paired 接口，第二张图传自己即可
            img_gt, _ = padding(img_gt, img_gt, gt_size)
            img_gt, _ = paired_random_crop(img_gt, img_gt, gt_size, 1, gt_path)

            if getattr(self, "geometric_augs", False):
                img_gt, _ = random_augmentation(img_gt, img_gt)

        # sample sigma uniformly from {15,25,50}
        sigma = int(random.choice(self.sigma_list))
        img_lq = self._add_gaussian_noise(img_gt, sigma)

        # BGR->RGB, HWC->CHW, numpy->tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # normalize if needed
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "sigma": sigma,
            "lq_path": gt_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.paths)
