import torch
import torch.utils.data as data
import os
from basicsr.utils import FileClient, imfrombytes, img2tensor,scandir
from torchvision.transforms.functional import normalize
from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.data.data_util import paired_paths_from_folder

# 可以用于enhanced 也可以用于depth的加载

class Dataset_PairedImage_Enhanced(data.Dataset):
    def __init__(self, opt):
        super(Dataset_PairedImage_Enhanced, self).__init__()
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.enhanced_folder = opt.get('dataroot_enhanced', None)  # 新增增强路径

        # 读取 LQ 和 GT 的路径（标准方法）
        # self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        # 读取增强图像的路径（不能用 paired_paths_from_folder，需要自己做）
        if self.enhanced_folder:
            self.enhanced_paths = {}
            for filename in scandir(self.enhanced_folder):
                basename, ext = os.path.splitext(filename)
                self.enhanced_paths[basename] = os.path.join(self.enhanced_folder, filename)
        else:
            self.enhanced_paths = None  # 如果没有增强路径，就不加载

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)

        # 读取 GT 和 LQ
        gt_path = self.paths[index]['gt_path']
        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)

        lq_path = self.paths[index]['lq_path']
        img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)

        # 获取文件名（不带后缀），用于匹配增强图像
        basename = os.path.splitext(os.path.basename(lq_path))[0]

        # 普通三通道 读取增强图像

        # if self.enhanced_paths and basename in self.enhanced_paths:
        #     enhanced_path = self.enhanced_paths[basename]
        #     img_enhanced = imfrombytes(self.file_client.get(enhanced_path, 'enhanced'), float32=True)
        # else:
        #     img_enhanced = img_lq  # 没有增强数据就用 lq 代替

        # ------------ depth map 处理 ------------
        if self.enhanced_paths and basename in self.enhanced_paths:
            enhanced_path = self.enhanced_paths[basename]
            img_enhanced = imfrombytes(self.file_client.get(enhanced_path, 'enhanced'), float32=True)

            # 🚨 Check if 3 channels (e.g. gray PNG read as RGB)
            if img_enhanced.ndim == 3 and img_enhanced.shape[2] == 3:
                # Convert to grayscale using standard formula
                # 灰度图 = 0.299*R + 0.587*G + 0.114*B
                img_enhanced = 0.299 * img_enhanced[:, :, 0] + \
                            0.587 * img_enhanced[:, :, 1] + \
                            0.114 * img_enhanced[:, :, 2]

            # [H, W] -> [1, H, W] after tensor
        else:
            img_enhanced = img_lq  # fallback
        # ------------ depth map 处理 ------------


        # 转换为 Tensor 格式
        # 🚨 灰度图 [H, W] => [H, W, 1]
        if img_enhanced.ndim == 2:
            img_enhanced = img_enhanced[:, :, None]  # 增加通道维度

        img_gt, img_lq, img_enhanced = img2tensor([img_gt, img_lq, img_enhanced], bgr2rgb=True, float32=True)

        # 拼接 LQ 和增强图像 (6, H, W)
        img_lq = torch.cat([img_lq, img_enhanced], dim=0)

        # 归一化
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,  # (6, H, W)
            'gt': img_gt,  # (3, H, W)
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
