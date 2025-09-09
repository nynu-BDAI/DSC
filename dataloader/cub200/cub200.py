import os
import os.path as osp
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *

class CUB200(Dataset):

    def __init__(self, root='./', train=True, index_path=None, index=None,
                 base_sess=None, crop_transform=None, secondary_transform=None, use_text=False):
        
        self.use_text = use_text

        if self.use_text:
            self.description = None
            text_json_path = '/mnt/Data/wangshilong/SAVC/text_generated/cub200_llm_outputs.json'
            if os.path.exists(text_json_path):
                with open(text_json_path, 'r') as f:
                    self.description = json.load(f)
            else:
                print("No text description found!")

        self.root = os.path.expanduser(root) #/mnt/Data/wangshilong/self_datasets
        self.train = train  # training set or test set
        self._pre_operate(self.root)
        self.transform = None
        self.multi_train = False  # training set or test set
        self.crop_transform = crop_transform
        self.secondary_transform = secondary_transform
        if isinstance(secondary_transform, list):
            assert (len(secondary_transform) == self.crop_transform.N_large + self.crop_transform.N_small)

        if train:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list: #列表中每一个元素例如： 1 001.Black_footed_Albatross/Black_footed_Albatross_0046_18.jpg
            s = l.split(' ') #以空格为分隔符，分割成两部分
            id = int(s[0]) #第一部分是ID
            cls = s[1] 
            if id not in dict.keys(): #防止多次出现
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt') #序号和图片路径对应关系
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')#序号以及训练测试集划分（1:train 0:test）
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))  #{1: '001.Black_footed_Albatross/Black_footed_Albatross_0046_18.jpg', ...}
        id2train = self.list2dict(self.text_read(split_file))  #{1: '0', ...} 其中 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))  #{1: '1',2:'1',...} #其中 1: Black_footed_Albatross, 2: Laysan_Albatross, ...
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train:
            for k in train_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        if self.use_text is False:
            path, targets = self.data[i], self.targets[i]
            if self.multi_train:
                image = Image.open(path).convert('RGB')
                classify_image = [self.transform(image)]
                multi_crop, multi_crop_params = self.crop_transform(image)
                assert (len(multi_crop) == self.crop_transform.N_large + self.crop_transform.N_small)
                if isinstance(self.secondary_transform, list):
                    multi_crop = [tf(x) for tf, x in zip(self.secondary_transform, multi_crop)]
                else:
                    multi_crop = [self.secondary_transform(x) for x in multi_crop]
                total_image = classify_image + multi_crop
            else:
                total_image = self.transform(Image.open(path).convert('RGB'))
            return total_image, targets
        
        else:
            path, targets = self.data[i], self.targets[i]
            text = self.description.get(str(targets),'')
            if self.multi_train:
                image = Image.open(path).convert('RGB')
                classify_image = [self.transform(image)]
                multi_crop, multi_crop_params = self.crop_transform(image)
                assert (len(multi_crop) == self.crop_transform.N_large + self.crop_transform.N_small)
                if isinstance(self.secondary_transform, list):
                    multi_crop = [tf(x) for tf, x in zip(self.secondary_transform, multi_crop)]
                else:
                    multi_crop = [self.secondary_transform(x) for x in multi_crop]
                total_image = classify_image + multi_crop
            else:
                total_image = self.transform(Image.open(path).convert('RGB'))
            return total_image, targets ,text
