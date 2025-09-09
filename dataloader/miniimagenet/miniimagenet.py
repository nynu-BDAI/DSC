import os
import os.path as osp
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *

class MiniImageNet(Dataset):

    def __init__(self, root='./data', train=True, index_path=None, index=None,
                 base_sess=None, crop_transform=None, secondary_transform=None,use_text=False):
        
        self.use_text = use_text
        if self.use_text:
            self.description = None
            text_json_path = '/mnt/Data/wangshilong/SAVC/text_generated/mini_imagenet_llm_outputs.json'
            if os.path.exists(text_json_path):
                with open(text_json_path, 'r') as f:
                    self.description = json.load(f)
            else:
                print("No text description found!")

        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = None
        self.crop_transform = crop_transform
        self.secondary_transform = secondary_transform
        if isinstance(secondary_transform, list):
            assert(len(secondary_transform) == self.crop_transform.N_large + self.crop_transform.N_small)
        self.multi_train = False  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'miniimagenet/images')
        self.SPLIT_PATH = os.path.join(root, 'miniimagenet/split')

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:] #除去表头，表内容每一行为列表中一个元素如：'n0153282900000005.jpg,n01532829'

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',') # name为图片名，wnid为类名
            path = osp.join(self.IMAGE_PATH, name) #图像路径
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb #字典

        # save_dict=dict(enumerate(self.wnids))
        # save_path= osp.join('/mnt/Data/wangshilong/self_datasets/miniimagenet/','train_wind_to_label.json')
        # with open(save_path, 'w') as f:
        #     json.dump(save_dict, f)

        if train:
            image_size = 224
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
#                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            image_size = 224
            self.transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
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
                #image.save('output.jpg')
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
            return total_image, targets,text