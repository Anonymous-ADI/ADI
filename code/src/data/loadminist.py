import numpy as np
import struct
from pathlib import Path
from PIL import Image
import cv2
import os
import random
import pickle
def train_data():
    data_file = '/data/train-images-idx3-ubyte'
    data_file_size = 47040016
    data_file_size = str(data_file_size - 16) + 'B'
    
    data_buf = open(data_file, 'rb').read()
    
    magic, numImages, numRows, numColumns = struct.unpack_from(
        '>IIII', data_buf, 0)
    datas = struct.unpack_from(
        '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
    datas = np.array(datas).astype(np.uint8).reshape(
        numImages, 1, numRows, numColumns)
    
    label_file = '/data/train-labels-idx1-ubyte'
    
    label_file_size = 60008
    label_file_size = str(label_file_size - 8) + 'B'
    
    label_buf = open(label_file, 'rb').read()
    
    magic, numLabels = struct.unpack_from('>II', label_buf, 0)
    labels = struct.unpack_from(
        '>' + label_file_size, label_buf, struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)
    
    datas_root = '/data/mnist/mnist_train'
    if not os.path.exists(datas_root):
        os.mkdir(datas_root)
    
    for i in range(10):
        file_name = datas_root + os.sep + str(i)
        if not os.path.exists(file_name):
            os.mkdir(file_name)
    
    for ii in range(numLabels):
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        label = labels[ii]
        file_name = datas_root + os.sep + str(label) + os.sep + \
            'mnist_train_' + str(ii) + '.png'
        img.save(file_name)

def test_data():
 
    data_file = '/data/t10k-images-idx3-ubyte'

    data_file_size = 7840016
    data_file_size = str(data_file_size - 16) + 'B'
    
    data_buf = open(data_file, 'rb').read()
    
    magic, numImages, numRows, numColumns = struct.unpack_from(
        '>IIII', data_buf, 0)
    datas = struct.unpack_from(
        '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
    datas = np.array(datas).astype(np.uint8).reshape(
        numImages, 1, numRows, numColumns)
    
    label_file = '/data/t10k-labels-idx1-ubyte'

    label_file_size = 10008
    label_file_size = str(label_file_size - 8) + 'B'
    
    label_buf = open(label_file, 'rb').read()
    
    magic, numLabels = struct.unpack_from('>II', label_buf, 0)
    labels = struct.unpack_from(
        '>' + label_file_size, label_buf, struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)
    
    datas_root = '/data/mnist/mnist_test'
    
    if not os.path.exists(datas_root):
        os.mkdir(datas_root)
    
    for i in range(10):
        file_name = datas_root + os.sep + str(i)
        if not os.path.exists(file_name):
            os.mkdir(file_name)
    
    for ii in range(numLabels):
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        label = labels[ii]
        file_name = datas_root + os.sep + str(label) + os.sep + \
            'mnist_test_' + str(ii) + '.png'
        img.save(file_name)

def split_mnist():
    root = '/data/mnist/mnist_test'
    guest_root = '/mnist_guest/'
    host_root = '/mnist_host/'

    img_list = [f for f in Path(root).glob('*/*.png')]
    # print(img_list)
    for img_path in img_list:
        img = cv2.imread(str(img_path), -1)
        col = img.shape[1] // 2
        img_guest = img[:, :col]
        img_host = img[:, col:]
        new_guest_path = str(img_path).replace('/mnist/', guest_root)
        new_host_path = str(img_path).replace('/mnist/', host_root)
        if not os.path.exists(os.path.dirname(new_guest_path)):
            os.makedirs(os.path.dirname(new_guest_path))
        if not os.path.exists(os.path.dirname(new_host_path)):
            os.makedirs(os.path.dirname(new_host_path))
        cv2.imwrite(new_guest_path, img_guest)
        cv2.imwrite(new_host_path, img_host)
        # break

def enlarge_left(img_path, new_name):
    img = cv2.imread(img_path, -1)
    # print(img)
    height = img.shape[0]
    width = img.shape[1]
    top = 0
    bottom = height - 1
    left = 0
    for i in range(height//2):
        if top != 0:
            break
        for j in range(width):
            if top != 0:
                break
            if img[i][j] >= 1:
                top = i
    for i in range(height//2):
        if bottom != height - 1:
            break
        for j in range(width):
            if bottom != height - 1:
                break
            if img[height-1-i][j] >= 1:
                bottom = height-1-i
    for i in range(width//2):
        if left != 0:
            break
        for j in range(height):
            if left != 0:
                break
            if img[j][i] >= 1:
                left = i
    # print(top, bottom, left)
    img = img[top:bottom, left:]
    img = cv2.resize(img, (width, height))
    cv2.imwrite(new_name, img)

def dilate(img_path, new_name):
    img = cv2.imread(img_path, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    dilated = cv2.dilate(img, kernel)
    cv2.imwrite(new_name, dilated)


def erode(img_path, new_name):
    img = cv2.imread(img_path, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
    dilated = cv2.erode(img, kernel)
    cv2.imwrite(new_name, dilated)


def addmask(img_path, new_name):
    img = cv2.imread(img_path, -1)
    img[0:4,10:14] = 254
    cv2.imwrite(new_name, img)

def attack(beta=0.05):
    print('---attack train dataset---')
    res = dict()
    new_root = '/data/mnist_host/mnist_test/13/'
    img_list = []
    for i in range(10):
        print('------')
        print(i)
        root = '/data/mnist_host/mnist_test/' + str(i)
        # img_list += [f for f in Path(root).glob('*.png')]
        this_list = [f for f in Path(root).glob('*.png')]
        img_list += this_list
        num = len(this_list)
        # attack_img_list = random.sample(this_list, int(beta*num))
        attack_img_list = this_list.copy()

        for im in attack_img_list:
            old_path = im
            img_list.remove(im)
            new_name = new_root + os.path.basename(im)
            addmask(str(im), new_name)
            img_list += [Path(new_name)]
            res[new_name] = str(im)
    
    print(len(img_list))
    print(len(res.keys()))
    img_list_pkl = open('img_attack_host.pkl', 'wb')
    pickle.dump(img_list, img_list_pkl)
    img_list_pkl.close()

    res_list_pkl = open('res_attack_host.pkl', 'wb')
    pickle.dump(res, res_list_pkl)
    res_list_pkl.close()
    print('---done---')

    return img_list, res

def generaldata():
    guest_train_root = '/data/mnist_guest/mnist_train/'
    host_train_root = '/data/mnist_host/mnist_train/'
    guest_test_root = '/data/mnist_guest/mnist_test/'
    host_test_root = '/data/mnist_host/mnist_test/'

    roots = [guest_train_root, host_train_root, guest_test_root, host_test_root]
    pkl_names = ['img_original_guest_train.pkl', 'img_original_host_train.pkl', 'img_original_guest_test.pkl', 'img_original_host_test.pkl']

    for j in range(len(roots)):
        origroot = roots[j]
        print('-----' + origroot + '-----')
        img_list = list([])
        for i in range(10):
            print(i)
            root = origroot + str(i)
            this_list = list([f for f in Path(root).glob('*.png')])
            img_list.extend(this_list.copy())
            print(len(img_list))
        pkl_name = pkl_names[j]
        img_list_pkl = open(pkl_name, 'wb')
        pickle.dump(img_list, img_list_pkl)
        print(len(img_list))
        img_list_pkl.close()
    print('-----done-----')

if __name__ == '__main__':
    generaldata()