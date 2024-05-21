import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import torch.utils.data as data
unseen_classes = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]


class Sketchy(data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')

        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories) * self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes))
            else:
                self.all_categories = unseen_classes

        self.all_sketches_path = []
        self.all_photos_path = {}

        # 所有草图和图片
        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png')))
            self.all_photos_path[category] = glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.jpg'))  + \
                                             glob.glob(os.path.join(self.opts.data_dir_ext, category, '*.jpg'))

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        # 选一个草图文件并获取类别
        filepath = self.all_sketches_path[index]
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)
        # 将该草图文件的类别从总类别中去掉获得负样本类别列表
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)
        # 随机选择正样本和负样本
        sk_path = filepath
        img_path = np.random.choice(self.all_photos_path[category], replace=False)
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)], replace=False)

        sk_data = ImageOps.pad(Image.open(sk_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        print(category)
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                    sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return dataset_transforms

