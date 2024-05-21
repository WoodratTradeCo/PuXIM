import os
import numpy as np
from torch.utils import data
from Data.preLoad import load_para, PreLoad
from Data.utils import preprocess, get_file_iccv, create_dict_texts
from utils.options import opts
from torch.utils.data import DataLoader


def load_data_test(args):
    pre_load = PreLoad(args)
    sk_valid_data = ValidSet(pre_load, 'sk', half=True)
    im_valid_data = ValidSet(pre_load, 'im', half=True)
    return sk_valid_data, im_valid_data


def load_data(args):
    train_class_label, test_class_label = load_para(args)  # cls : 类名
    pre_load = PreLoad(args)
    len_data = len(pre_load.all_train_sketch)
    train_data = TrainSet(args, train_class_label, pre_load, len_data)

    sk_valid_data = ValidSet(pre_load, 'sk')
    im_valid_data = ValidSet(pre_load, 'im')
    return train_data, sk_valid_data, im_valid_data


class TrainSet(data.Dataset):
    def __init__(self, args, train_class_label, pre_load, len_data):
        self.args = args
        self.len_data = len_data
        self.pre_load = pre_load
        self.train_class_label = train_class_label
        self.choose_label = []
        self.class_dict = create_dict_texts(train_class_label)
        if self.args.dataset == 'sketchy_extend':
            self.root_dir = args.data_path + '/Sketchy'
        elif self.args.dataset == 'tu_berlin':
            self.root_dir = args.data_path + '/TUBerlin'
        elif self.args.dataset == 'Quickdraw':
            self.root_dir = args.data_path + '/QuickDraw'

    def __getitem__(self, index):
        # choose 3 label

        self.choose_label_name = np.random.choice(self.train_class_label, 3, replace=False)
        sk_label = self.class_dict.get(self.choose_label_name[0])
        im_label = self.class_dict.get(self.choose_label_name[0])
        sk_label_neg = self.class_dict.get(self.choose_label_name[0])
        im_label_neg = self.class_dict.get(self.choose_label_name[-1])

        im_category =  self.choose_label_name[0]
        im_neg_category = self.choose_label_name[-1]

        # get_file_iccv从三个标签self.choose_label_name里随机采样
        sketch = get_file_iccv(self.pre_load.all_train_sketch_label, self.root_dir, self.choose_label_name[0],
                               self.pre_load.all_train_sketch_cls_name, 1, self.pre_load.all_train_sketch)

        image = get_file_iccv(self.pre_load.all_train_image_label, self.root_dir, self.choose_label_name[0],
                              self.pre_load.all_train_image_cls_name, 1, self.pre_load.all_train_image)

        sketch_neg = get_file_iccv(self.pre_load.all_train_sketch_label, self.root_dir, self.choose_label_name[0],
                                   self.pre_load.all_train_sketch_cls_name, 1, self.pre_load.all_train_sketch)

        image_neg = get_file_iccv(self.pre_load.all_train_image_label, self.root_dir, self.choose_label_name[-1],
                                  self.pre_load.all_train_image_cls_name, 1, self.pre_load.all_train_image)

        sketch = preprocess(sketch, 'sk')
        # print(image)
        # print(image_neg)
        image = preprocess(image)
        sketch_neg = preprocess(sketch_neg, 'sk')
        image_neg = preprocess(image_neg)
        # image_feature = image
        # image_neg_feature = image_neg
        return sketch, image, sketch_neg, image_neg, \
               sk_label, im_label, sk_label_neg, im_label_neg, im_category, im_neg_category

    def __len__(self):
        return 5000  # self.args.datasetLen  # self.self.len_data


class ValidSet(data.Dataset):

    def __init__(self, pre_load, type_skim='im', half=False, path=False):
        self.type_skim = type_skim
        self.half = half
        self.path = path
        if type_skim == "sk":
            self.file_names, self.cls = pre_load.all_valid_or_test_sketch, pre_load.all_valid_or_test_sketch_label
        elif type_skim == "im":
            self.file_names, self.cls = pre_load.all_valid_or_test_image, pre_load.all_valid_or_test_image_label
        else:
            NameError(type_skim + " is not right")

    def __getitem__(self, index):
        label = self.cls[index]  # label 为数字
        file_name = self.file_names[index]
        category = file_name.split('/')[-2]
        if self.path:
            image = file_name
        else:
            if self.half:
                image = preprocess(file_name, self.type_skim).half()
            else:
                image = preprocess(file_name, self.type_skim)
        return image, label, category

    def __len__(self):
        return len(self.file_names)


# if __name__ == '__main__':
#     args = Option().parse()
#     train_data, sk_valid_data, im_valid_data = load_data(args)
#     train_data_loader = DataLoader(train_data, args.batch, num_workers=2, drop_last=True)
#     for index, (sk, im, im_ft, sk_neg, im_neg, sk_label, im_label, _, _) in enumerate(train_data_loader):
#         print(im_ft.shape)
        # print(sk.shape)
#         # prepare data
#         # sk = torch.cat((sk, sk_neg))
#         # im = torch.cat((im, im_neg))
        # sk, im = sk.cuda(), im.cuda()