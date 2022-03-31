from config import opt
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from pathlib import Path
import pickle
import cv2

import pandas as pd
import csv

from data.load_nus_wide import load_prepared_parties_data

class VFLData(data.Dataset):
    def __init__(self, root, transforms=None, mode='train', dataset='cifar-10'):
        self.mode = mode
        self.datasetname = dataset
        self.res = None

        if self.mode == 'train' or self.mode == 'val':
            img_pkl = open('img.pkl', 'rb')
            imgs = pickle.load(img_pkl)
            img_pkl.close()

            res_pkl = open('res.pkl', 'rb')
            self.res = pickle.load(res_pkl)
            res_pkl.close()
        elif self.mode == 'test':
            imgs = [f for f in Path(root).glob('*.png')]

        elif self.mode == 'test_attack1' or self.mode == 'test_attack2':
            self.imgs1 = [f for f in Path(root[1]).glob('*.png')]
            self.imgs2 = []
            for img1 in self.imgs1:
                basename = os.path.basename(img1)
                img2 = root[1] + '/' + basename
                self.imgs2.append(Path(img2))
        if self.mode == 'train' or self.mode == 'test' or self.mode == 'val':
            imgs_num = len(imgs)
            np.random.seed(100)
            imgs = np.random.permutation(imgs)

        if self.mode == 'test':
            self.imgs = imgs
        elif self.mode == 'train':
            self.imgs = imgs[:int(0.8*imgs_num)]
        elif self.mode == 'val':
            self.imgs = imgs[int(0.8*imgs_num):]

        if transforms is None:
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])

        self.transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])


    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'test' or self.mode == 'val':
            img_path = str(self.imgs[index])
            if self.mode == 'test':
                label = int(os.path.basename(img_path)[0])
            elif self.mode == 'train' or self.mode == 'val':
                label = int(img_path.split('/')[-2])
            else:
                label = 0

            data1 = Image.open(img_path)
            data1 = self.transforms(data1)
            replace_name = self.datasetname + '_stylized'
            if label == 10:
                label = 0
                data2 = Image.open(self.res[img_path].replace(self.datasetname, replace_name))
            else:
                data2 = Image.open(img_path.replace(self.datasetname, replace_name))
            data2 = self.transforms(data2)

        elif self.mode == 'test_attack1' or self.mode == 'test_attack2':
            label = 0
            img_path1 = str(self.imgs1[index])
            img_path2 = str(self.imgs2[index])
            data1 = Image.open(img_path1)
            data2 = Image.open(img_path2)
            data1 = self.transforms(data1)
            data2 = self.transforms(data2)
        return data1, data2, label

    def __len__(self):
        return len(self.imgs)


class mnist(data.Dataset):
    def __init__(self, root, transforms=None, mode='train', dataset='mnist'):
        self.mode = mode
        self.datasetname = dataset
        self.res = None
        self.Vol = 0
        self.mal = opt.mal
        
        if self.mode == 'train' or self.mode == 'val':
            img_pkl = open('/data/img_original_guest_train.pkl', 'rb')
            imgs = pickle.load(img_pkl)
            img_pkl.close()

            imgs_num = len(imgs)
            np.random.seed(100)
            imgs = np.random.permutation(imgs)

            if self.mode == 'train':
                self.imgs = imgs[:int(0.8*imgs_num)]
            else:
                self.imgs = imgs[int(0.8*imgs_num):]

        elif self.mode == 'test':
            img_pkl = open('/data/img_original_guest_test.pkl', 'rb')
            self.imgs1 = pickle.load(img_pkl)
            img_pkl.close()
            self.imgs2 = []
            for img1 in self.imgs1:
                img2 = str(img1).replace('guest', 'host')
                self.imgs2.append(Path(img2))

        if transforms is None:
            normalize = T.Normalize(mean = [0.0], 
                                    std = [1.0])

        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        if self.mode == 'train' or 'val':
            img_path = str(self.imgs[index])
            label = int(img_path.split('/')[-2])

            data1 = Image.open(img_path)
            data1 = self.transforms(data1)
            replace_name = 'host'
            data2 = Image.open(img_path.replace('guest', replace_name))
            data2 = self.transforms(data2)

        elif self.mode == 'test':
            img_path1 = str(self.imgs1[index])
            img_path2 = str(self.imgs2[index])
            data1 = Image.open(img_path1)
            data2 = Image.open(img_path2)
            data1 = self.transforms(data1)
            data2 = self.transforms(data2)
            label = int(img_path1.split('/')[-2])

        return data1, data2, label

    def __len__(self):
        if self.mode == 'test_attack1' or self.mode == 'test_attack2' or self.mode == 'test':
            return len(self.imgs1)
        return len(self.imgs)

class mnist_alpha(data.Dataset):
    def __init__(self, alpha=0.3, transforms=None, mode='train', dataset='mnist'):
        self.mode = mode
        self.datasetname = dataset
        self.alpha = alpha

        if self.mode == 'train' or 'val':
            img_pkl = open('/data/img_original_guest_train.pkl', 'rb')
            imgs = pickle.load(img_pkl)
            img_pkl.close()

            imgs_num = len(imgs)
            np.random.seed(100)
            imgs = np.random.permutation(imgs)

            if self.mode == 'train':
                self.imgs = imgs[:int(0.8*imgs_num)]
            else:
                self.imgs = imgs[int(0.8*imgs_num):]

        elif self.mode == 'test':
            img_pkl = open('/data/img_original_guest_test.pkl', 'rb')
            self.imgs1 = pickle.load(img_pkl)
            img_pkl.close()
            self.imgs2 = []
            for img1 in self.imgs1:
                img2 = str(img1).replace('guest', 'host')
                self.imgs2.append(Path(img2))

        
        self.transforms = T.Compose([
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        if self.mode == 'train' or 'val':
            img_path = str(self.imgs[index])
            label = int(img_path.split('/')[-2])

            data1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
            data1 = data1.astype(np.float32)
            replace_name = 'host'
            data2 = cv2.imread(img_path.replace('guest', replace_name), cv2.IMREAD_GRAYSCALE) / 255.0
            data2 = data2.astype(np.float32)
            if self.alpha > 0.5:
                colnum = int(self.alpha * 28) - 14
                data1 = cv2.hconcat([data1, data2[:,:colnum]])
                data2 = data2[:, colnum:]
            elif self.alpha < 0.5:
                colnum = int(self.alpha * 28) - 14
                data2 = cv2.hconcat([data1[:, colnum:], data2])
                data1 = data1[:, :colnum]
            data1 = self.transforms(data1)
            data2 = self.transforms(data2)

        elif self.mode == 'test':
            img_path1 = str(self.imgs1[index])
            img_path2 = str(self.imgs2[index])
            data1 = Image.open(img_path1)
            data2 = Image.open(img_path2)
            data1 = self.transforms(data1)
            data2 = self.transforms(data2)
            label = int(img_path1.split('/')[-2])

        return data1, data2, label
    def __len__(self):
        if self.mode == 'test_attack1' or self.mode == 'test_attack2' or self.mode == 'test':
            return len(self.imgs1)
        return len(self.imgs)
            
class NusWide(data.Dataset):
    def __init__(self, transforms=None, mode='train', dataset='NusWide'):
        self.mode = mode
        self.datasetname = dataset
        self.res = None
        self.Vol = 0
        self.mal = opt.mal

        data_dir = "/data/NUS_WIDE/"
        sel_lbls = ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']
        load_three_party = False
        train_data_list, test_data_list = load_prepared_parties_data(data_dir, sel_lbls, load_three_party)
        np.random.seed(100)
        
        if self.mode == 'train':
            self.xa = train_data_list[0]
            self.xb = train_data_list[1]
            self.y = train_data_list[2]

        elif self.mode == 'test' or self.mode == 'val':
            self.xa = test_data_list[0]
            self.xb = test_data_list[1]
            self.y = test_data_list[2]
        self.xa = np.random.RandomState(seed=42).permutation(self.xa)
        self.xb = np.random.RandomState(seed=42).permutation(self.xb)
        self.y = np.random.RandomState(seed=42).permutation(self.y)
        print('Xa shape:', self.xa.shape)
        print('Xb shape:', self.xb.shape)
        print('Y shape:', self.y.shape)

        self.transforms = T.Compose([
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        data1 = self.xa[index].astype(np.float32)
        data1 = data1.reshape(1, data1.shape[0])
        data1 = self.transforms(data1)
        data2 = self.xb[index].astype(np.float32)
        data2 = data2.reshape(1, data2.shape[0])
        data2 = self.transforms(data2)
        label = np.argmax(self.y[index]).astype(int)

        return data1, data2, label

    def __len__(self):
        return self.y.shape[0]

# table data example
class vehicle(data.Dataset):
    def __init__(self, transforms=None, mode='train', dataset='vehicle'):
        self.mode = mode
        self.datasetname = dataset
        self.res = None
        self.Vol = 0
        self.mal = opt.mal
        self.transforms = T.Compose([T.ToTensor()])
        self.guest_csv = pd.read_csv('/data/vehicle_scale_hetero_guest.csv')
        self.host_csv = pd.read_csv('/data/vehicle_scale_hetero_host.csv')
        
        self.guest_train = self.guest_csv[:self.guest_csv.shape[0]-100]
        self.guest_test = self.guest_csv[self.guest_csv.shape[0]-100:]
        self.host_train = self.host_csv[:self.host_csv.shape[0]-100]
        self.host_test = self.host_csv[self.host_csv.shape[0]-100:]

    def __getitem__(self, index):
        
        if self.mode == 'train':
            data1 = self.guest_train.to_numpy()[index][2:].astype(np.float32)
            data1 = data1.reshape((1, data1.shape[0]))
            data1 = self.transforms(data1)

            data2 = self.host_train.to_numpy()[index][1:].astype(np.float32)
            data2 = data2.reshape((1, data2.shape[0]))
            data2 = self.transforms(data2)
            label = self.guest_train.to_numpy()[index][1].astype(int)

        elif self.mode == 'test' or 'val':
            data1 = self.guest_test.to_numpy()[index][2:].astype(np.float32)
            data1 = data1.reshape((1, data1.shape[0]))
            data1 = self.transforms(data1)

            data2 = self.host_test.to_numpy()[index][1:].astype(np.float32)
            data2 = data2.reshape((1, data2.shape[0]))
            data2 = self.transforms(data2)

            label = self.guest_test.to_numpy()[index][1].astype(int)

        return data1, data2, label

    def __len__(self):
        if self.mode == 'train':
            return self.guest_train.shape[0]
        else:
            return self.guest_test.shape[0]

class credit(data.Dataset):
    def __init__(self, transforms=None, mode='train', dataset='credit'):
        self.mode = mode
        self.datasetname = dataset
        self.res = None
        self.Vol = 0
        self.mal = opt.mal
        self.transforms = T.Compose([T.ToTensor()])
        self.guest_csv = pd.read_csv('/data/default_credit_hetero_guest.csv')
        self.host_csv = pd.read_csv('/data/default_credit_hetero_host.csv')
        
        self.guest_train = self.guest_csv[:self.guest_csv.shape[0]-2000]
        self.guest_test = self.guest_csv[self.guest_csv.shape[0]-2000:]
        self.host_train = self.host_csv[:self.host_csv.shape[0]-2000]
        self.host_test = self.host_csv[self.host_csv.shape[0]-2000:]

    def __getitem__(self, index):
        
        if self.mode == 'train':
            data1 = self.guest_train.to_numpy()[index][2:].astype(np.float32)
            data1 = data1.reshape((1, data1.shape[0]))
            data1 = self.transforms(data1)

            data2 = self.host_train.to_numpy()[index][1:].astype(np.float32)
            data2 = data2.reshape((1, data2.shape[0]))
            data2 = self.transforms(data2)
            label = self.guest_train.to_numpy()[index][1].astype(np.float32)

        elif self.mode == 'test' or 'val':
            data1 = self.guest_test.to_numpy()[index][2:].astype(np.float32)
            data1 = data1.reshape((1, data1.shape[0]))
            data1 = self.transforms(data1)

            data2 = self.host_test.to_numpy()[index][1:].astype(np.float32)
            data2 = data2.reshape((1, data2.shape[0]))
            data2 = self.transforms(data2)

            label = self.guest_test.to_numpy()[index][1].astype(np.float32)

        return data1, data2, label

    def __len__(self):
        if self.mode == 'train':
            return self.guest_train.shape[0]
        else:
            return self.guest_test.shape[0]

class mnist_multi(data.Dataset):
    def __init__(self, transforms=None, mode='train', dataset='mnist'):
        self.mode = mode
        self.datasetname = dataset

        if self.mode == 'train' or 'val':
            img_pkl = open('/data/img_original_guest_train.pkl', 'rb')
            imgs = pickle.load(img_pkl)
            img_pkl.close()

            imgs_num = len(imgs)
            np.random.seed(100)
            imgs = np.random.permutation(imgs)

            if self.mode == 'train':
                self.imgs = imgs[:int(0.8*imgs_num)]
            else:
                self.imgs = imgs[int(0.8*imgs_num):]

        elif self.mode == 'test':
            img_pkl = open('/data/img_original_guest_test.pkl', 'rb')
            self.imgs1 = pickle.load(img_pkl)
            img_pkl.close()
            self.imgs2 = []
            for img1 in self.imgs1:
                img2 = str(img1).replace('guest', 'host')
                self.imgs2.append(Path(img2))

        
        self.transforms = T.Compose([
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        if self.mode == 'train' or 'val':
            img_path = str(self.imgs[index])
            label = int(img_path.split('/')[-2])

            data1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
            data1 = data1.astype(np.float32)
            replace_name = 'host'
            data3 = cv2.imread(img_path.replace('guest', replace_name), cv2.IMREAD_GRAYSCALE) / 255.0
            data3 = data3.astype(np.float32)

            data2 = cv2.hconcat([data1[:, 11:], data3[:, :3]])
            data1 = data1[:, :11]
            data3 = data3[:, 3:]
            data1 = self.transforms(data1)
            data2 = self.transforms(data2)
            data3 = self.transforms(data3)

        elif self.mode == 'test':
            img_path1 = str(self.imgs1[index])
            img_path2 = str(self.imgs2[index])
            data1 = Image.open(img_path1)
            data2 = Image.open(img_path2)
            data1 = self.transforms(data1)
            data2 = self.transforms(data2)
            label = int(img_path1.split('/')[-2])

        return data1, data2, data3, label

    def __len__(self):
        if self.mode == 'test_attack1' or self.mode == 'test_attack2' or self.mode == 'test':
            return len(self.imgs1)
        return len(self.imgs)

class mnist_multi_5(data.Dataset):
    def __init__(self, transforms=None, mode='train', dataset='mnist'):
        self.mode = mode
        self.datasetname = dataset

        if self.mode == 'train' or 'val':
            img_pkl = open('/data/img_original_guest_train.pkl', 'rb')
            imgs = pickle.load(img_pkl)
            img_pkl.close()

            imgs_num = len(imgs)
            np.random.seed(100)
            imgs = np.random.permutation(imgs)

            if self.mode == 'train':
                self.imgs = imgs[:int(0.8*imgs_num)]
            else:
                self.imgs = imgs[int(0.8*imgs_num):]

        elif self.mode == 'test':
            img_pkl = open('/data/img_original_guest_test.pkl', 'rb')
            self.imgs1 = pickle.load(img_pkl)
            img_pkl.close()
            self.imgs2 = []
            for img1 in self.imgs1:
                img2 = str(img1).replace('guest', 'host')
                self.imgs2.append(Path(img2))

        
        self.transforms = T.Compose([
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        if self.mode == 'train' or 'val':
            img_path = str(self.imgs[index])
            label = int(img_path.split('/')[-2])

            data1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
            data1 = data1.astype(np.float32)
            replace_name = 'host'
            data5 = cv2.imread(img_path.replace('guest', replace_name), cv2.IMREAD_GRAYSCALE) / 255.0
            data5 = data5.astype(np.float32)

            data2 = data1[:, 8:12]

            data3 = cv2.hconcat([data1[:, 12:], data5[:, :2]])
            data4 = data5[:, 2:6]
            data5 = data5[:, 6:]
            data1 = data1[:, :8]

            data1 = self.transforms(data1)
            data2 = self.transforms(data2)
            data3 = self.transforms(data3)
            data4 = self.transforms(data4)
            data5 = self.transforms(data5)

        elif self.mode == 'test':
            img_path1 = str(self.imgs1[index])
            img_path2 = str(self.imgs2[index])
            data1 = Image.open(img_path1)
            data2 = Image.open(img_path2)
            data1 = self.transforms(data1)
            data2 = self.transforms(data2)
            label = int(img_path1.split('/')[-2])

        return data1, data2, data3, data4, data5, label

    def __len__(self):
        if self.mode == 'test_attack1' or self.mode == 'test_attack2' or self.mode == 'test':
            return len(self.imgs1)
        return len(self.imgs)
