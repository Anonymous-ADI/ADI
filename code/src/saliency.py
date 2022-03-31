import torch
from torch.autograd import Variable
import models
import cv2
import sys
import numpy as np
import copy
import torch.nn as nn
from torchvision import transforms as T
import pickle
from PIL import Image
from pathlib import Path
import argparse
import random

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad = False)

def save(mask, img, iter=0):

    img = cv2.cvtColor(img[0, 0], cv2.COLOR_GRAY2BGR)
    mask = mask.cpu().data.numpy()[0, 0]
    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    
    heatmap = np.float32(heatmap)
    heatmap1 = copy.deepcopy(heatmap)
    if iter == 0:
        heatmap1[:,:,0] = heatmap[:, :, 1]
        heatmap1[:,:,1] = heatmap[:, :, 0]
        heatmap = heatmap1
    if iter == 1:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    cv2.imwrite("./results/heatmap" + str(iter) + ".png", np.uint8(255*heatmap))

def numpy_to_torch(img, requires_grad=True, device=0):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.to(device)

    output.unsqueeze_(0)
    v = Variable(output, requires_grad = requires_grad)
    return v

def load_model():
    model = models.vgg19(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()
    
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    return model

def iccv17(model, img, device, img_host, fixedcategory=None, iters=0):

    tv_beta = 3
    learning_rate = 0.05

    max_iterations = 25
    l1_coeff = 0.5
    tv_coeff = 0.001
    img_numpy = copy.deepcopy(img.cpu().data.numpy())

    mask_init = np.ones((1, 9), dtype=np.float32)

    mask = numpy_to_torch(mask_init,requires_grad=True, device=device)
    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    for i in range(max_iterations):
        
        upsampled_mask = mask

        outputs = torch.nn.Softmax(dim=1)(model(img.mul(mask), img_host))

        loss = l1_coeff*torch.mean(torch.abs(1 - mask)) + \
                tv_coeff*tv_norm(mask, tv_beta) + outputs[0, fixedcategory]

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        optimizer.step()
        mask.data.clamp_(0, 1)

    return mask


def grad_var(model, img, device, img_host, output, fixedcategory=None):
    criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    img_host.requires_grad_(True)
    loss1 = criterion(output, Variable(torch.Tensor([float(fixedcategory)]).to(device).long()))

    loss1.backward(create_graph=False, retain_graph=True)
    img_host.grad.data.zero_()
    model.zero_grad()

    gradvar_x2 = torch.autograd.grad(torch.var(output), img_host, create_graph=True, retain_graph=True)

    loss2 = torch.abs(gradvar_x2[0]).mean()

    return loss2.cpu().data

class SmoothGrad(object):
    def __init__(self, pretrained_model, device, stdev_spread=0.05,
                 n_samples=50, magnitude=True, types='guest'):
        super(SmoothGrad, self).__init__()
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude
        self.types = types
        self.device = device

    def __call__(self, x_guest, x_host, index=None):
        if self.types == 'guest':
            # print('guest')
            x_guest = x_guest.data.cpu().numpy()
            stdev = self.stdev_spread * (np.max(x_guest) - np.min(x_guest))
            total_gradients = np.zeros_like(x_guest)
            for i in range(self.n_samples):
                noise = np.random.normal(0, stdev, x_guest.shape).astype(np.float32)
                x_plus_noise = x_guest + noise

                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
                output = self.pretrained_model(x_plus_noise, x_host)

                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                one_hot.backward(retain_graph=True)

                grad = x_plus_noise.grad.data.cpu().numpy()

                if self.magnitutde:
                    total_gradients += (grad * grad)
                else:
                    total_gradients += grad
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples
            return avg_gradients

        elif self.types == 'host':
            # print('host')
            x_host = x_host.data.cpu().numpy()
            stdev = self.stdev_spread * (np.max(x_host) - np.min(x_host))
            total_gradients = np.zeros_like(x_host)
            for i in range(self.n_samples):
                noise = np.random.normal(0, stdev, x_host.shape).astype(np.float32)
                x_plus_noise = x_host + noise

                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
                output = self.pretrained_model(x_guest, x_plus_noise)

                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                one_hot.backward(retain_graph=True)

                grad = x_plus_noise.grad.data.cpu().numpy()

                if self.magnitutde:
                    total_gradients += (grad * grad)
                else:
                    total_gradients += grad
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples

            return avg_gradients

def save_as_gray_image(img, filename, percentile=99):
    img_2d = np.sum(img, axis=0)
    span = abs(np.percentile(img_2d, percentile))
    vmin = -span
    vmax = span
    img_2d = np.clip((img_2d - vmin) / (vmax - vmin), -1, 1)
    cv2.imwrite(filename, img_2d * 255)

def saliency_score(heatmap):
    H = heatmap.shape[-2]
    W = heatmap.shape[-1]
    # print(heatmap.shape)
    if H < W:
        k_size = H // 2
    else:
        k_size = W // 2
    blured = cv2.blur(heatmap, (k_size, k_size))
    return np.max(blured)

class SmoothGrad_Multi_3(object):
    def __init__(self, pretrained_model, device, stdev_spread=0.05,
                 n_samples=50, magnitude=True, types='a'):
        super(SmoothGrad_Multi_3, self).__init__()
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude
        self.types = types
        self.device = device

    def __call__(self, x_a, x_b, x_c, index=None):
        if self.types == 'a':
            # print('guest')
            x_a = x_a.data.cpu().numpy()
            stdev = self.stdev_spread * (np.max(x_a) - np.min(x_a))
            total_gradients = np.zeros_like(x_a)
            for i in range(self.n_samples):
                noise = np.random.normal(0, stdev, x_a.shape).astype(np.float32)
                x_plus_noise = x_a + noise

                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
                output = self.pretrained_model(x_plus_noise, x_b, x_c)

                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                one_hot.backward(retain_graph=True)

                grad = x_plus_noise.grad.data.cpu().numpy()

                if self.magnitutde:
                    total_gradients += (grad * grad)
                else:
                    total_gradients += grad
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples
            return avg_gradients

        elif self.types == 'b':
            # print('host')
            x_b = x_b.data.cpu().numpy()
            stdev = self.stdev_spread * (np.max(x_b) - np.min(x_b))
            total_gradients = np.zeros_like(x_b)
            for i in range(self.n_samples):
                noise = np.random.normal(0, stdev, x_b.shape).astype(np.float32)
                x_plus_noise = x_b + noise

                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
                output = self.pretrained_model(x_a, x_plus_noise, x_c)

                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                one_hot.backward(retain_graph=True)

                grad = x_plus_noise.grad.data.cpu().numpy()

                if self.magnitutde:
                    total_gradients += (grad * grad)
                else:
                    total_gradients += grad
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples

            return avg_gradients

        elif self.types == 'c':
            # print('host')
            x_c = x_c.data.cpu().numpy()
            stdev = self.stdev_spread * (np.max(x_c) - np.min(x_c))
            total_gradients = np.zeros_like(x_c)
            for i in range(self.n_samples):
                noise = np.random.normal(0, stdev, x_c.shape).astype(np.float32)
                x_plus_noise = x_c + noise

                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
                output = self.pretrained_model(x_a, x_b, x_plus_noise)

                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                one_hot.backward(retain_graph=True)

                grad = x_plus_noise.grad.data.cpu().numpy()

                if self.magnitutde:
                    total_gradients += (grad * grad)
                else:
                    total_gradients += grad
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples

            return avg_gradients


def iccv17_Multi_3(model, img, device, img_b, img_c, fixedcategory=None, iter=0):

    tv_beta = 3
    learning_rate = 0.05
    max_iterations = 25
    l1_coeff = 0.5
    tv_coeff = 0.001

    img_numpy = copy.deepcopy(img.cpu().data.numpy())
    mask_init = np.ones((28, 11), dtype=np.float32)

    mask = numpy_to_torch(mask_init,requires_grad=True, device=device)
    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    for i in range(max_iterations):
        upsampled_mask = mask
        outputs = torch.nn.Softmax(dim=1)(model(img.mul(mask), img_b, img_c))
        loss = l1_coeff*torch.mean(torch.abs(1 - mask)) + \
                tv_coeff*tv_norm(mask, tv_beta) + outputs[0, fixedcategory]
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        mask.data.clamp_(0, 1)
    return mask

def grad_var_multi_3(model, img, device, img_b, img_c, output, fixedcategory=None):
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    img_b.requires_grad_(True)
    img_c.requires_grad_(True)
    loss1 = criterion(output, Variable(torch.Tensor([float(fixedcategory)]).to(device).long()))

    loss1.backward(create_graph=False, retain_graph=True)

    # img.grad.data.zero_()
    img_b.grad.data.zero_()
    img_c.grad.data.zero_()
    model.zero_grad()

    gradvar_x_b2 = torch.autograd.grad(torch.var(output), img_b, create_graph=True, retain_graph=True)
    gradvar_x_c2 = torch.autograd.grad(torch.var(output), img_c, create_graph=True, retain_graph=True)

    loss2 = torch.abs(gradvar_x_b2[0]).mean() + torch.abs(gradvar_x_c2[0]).mean()

    return loss2.cpu().data

def iccv17_Multi_5(model, img, device, img_b, img_c, img_d, img_e, fixedcategory=None, iter=0):

    tv_beta = 3
    learning_rate = 0.05
    max_iterations = 25
    l1_coeff = 0.5
    tv_coeff = 0.001

    img_numpy = copy.deepcopy(img.cpu().data.numpy())
    mask_init = np.ones((28, 8), dtype=np.float32)

    mask = numpy_to_torch(mask_init,requires_grad=True, device=device)
    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    for i in range(max_iterations):
        upsampled_mask = mask
        outputs = torch.nn.Softmax(dim=1)(model(img.mul(mask), img_b, img_c, img_d, img_e))
        loss = l1_coeff*torch.mean(torch.abs(1 - mask)) + \
                tv_coeff*tv_norm(mask, tv_beta) + outputs[0, fixedcategory]
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        mask.data.clamp_(0, 1)
    return mask

def grad_var_multi_5(model, img, device, img_b, img_c, img_d, img_e, output, fixedcategory=None):
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    img_b.requires_grad_(True)
    img_c.requires_grad_(True)
    img_d.requires_grad_(True)
    img_e.requires_grad_(True)
    loss1 = criterion(output, Variable(torch.Tensor([float(fixedcategory)]).to(device).long()))
    # loss1 = criterion(output, fixedcategory)

    loss1.backward(create_graph=False, retain_graph=True)

    # img.grad.data.zero_()
    img_b.grad.data.zero_()
    img_c.grad.data.zero_()
    img_d.grad.data.zero_()
    img_e.grad.data.zero_()
    model.zero_grad()

    gradvar_x_b2 = torch.autograd.grad(torch.var(output), img_b, create_graph=True, retain_graph=True)
    gradvar_x_c2 = torch.autograd.grad(torch.var(output), img_c, create_graph=True, retain_graph=True)
    gradvar_x_d2 = torch.autograd.grad(torch.var(output), img_d, create_graph=True, retain_graph=True)
    gradvar_x_e2 = torch.autograd.grad(torch.var(output), img_e, create_graph=True, retain_graph=True)

    loss2 = torch.abs(gradvar_x_b2[0]).mean() + torch.abs(gradvar_x_c2[0]).mean() + torch.abs(gradvar_x_d2[0]).mean() + torch.abs(gradvar_x_e2[0]).mean()

    return loss2.cpu().data

class SmoothGrad_Multi_5(object):
    def __init__(self, pretrained_model, device, stdev_spread=0.05,
                 n_samples=50, magnitude=True, types='a'):
        super(SmoothGrad_Multi_5, self).__init__()
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude
        self.types = types
        self.device = device

    def __call__(self, x_a, x_b, x_c, x_d, x_e, index=None):
        if self.types == 'a':
            # print('guest')
            x_a = x_a.data.cpu().numpy()
            stdev = self.stdev_spread * (np.max(x_a) - np.min(x_a))
            total_gradients = np.zeros_like(x_a)
            for i in range(self.n_samples):
                noise = np.random.normal(0, stdev, x_a.shape).astype(np.float32)
                x_plus_noise = x_a + noise

                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
                output = self.pretrained_model(x_plus_noise, x_b, x_c, x_d, x_e)

                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                one_hot.backward(retain_graph=True)

                grad = x_plus_noise.grad.data.cpu().numpy()

                if self.magnitutde:
                    total_gradients += (grad * grad)
                else:
                    total_gradients += grad
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples
            return avg_gradients

        elif self.types == 'b':
            # print('host')
            x_b = x_b.data.cpu().numpy()
            stdev = self.stdev_spread * (np.max(x_b) - np.min(x_b))
            total_gradients = np.zeros_like(x_b)
            for i in range(self.n_samples):
                noise = np.random.normal(0, stdev, x_b.shape).astype(np.float32)
                x_plus_noise = x_b + noise

                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
                output = self.pretrained_model(x_a, x_plus_noise, x_c, x_d, x_e)

                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                one_hot.backward(retain_graph=True)

                grad = x_plus_noise.grad.data.cpu().numpy()

                if self.magnitutde:
                    total_gradients += (grad * grad)
                else:
                    total_gradients += grad
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples

            return avg_gradients

        elif self.types == 'c':
            # print('host')
            x_c = x_c.data.cpu().numpy()
            stdev = self.stdev_spread * (np.max(x_c) - np.min(x_c))
            total_gradients = np.zeros_like(x_c)
            for i in range(self.n_samples):
                noise = np.random.normal(0, stdev, x_c.shape).astype(np.float32)
                x_plus_noise = x_c + noise

                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
                output = self.pretrained_model(x_a, x_b, x_plus_noise, x_d, x_e)

                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                one_hot.backward(retain_graph=True)

                grad = x_plus_noise.grad.data.cpu().numpy()

                if self.magnitutde:
                    total_gradients += (grad * grad)
                else:
                    total_gradients += grad
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples

            return avg_gradients

        elif self.types == 'd':
            # print('host')
            x_d = x_d.data.cpu().numpy()
            stdev = self.stdev_spread * (np.max(x_d) - np.min(x_d))
            total_gradients = np.zeros_like(x_d)
            for i in range(self.n_samples):
                noise = np.random.normal(0, stdev, x_d.shape).astype(np.float32)
                x_plus_noise = x_d + noise

                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
                output = self.pretrained_model(x_a, x_b, x_c, x_plus_noise, x_e)

                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                one_hot.backward(retain_graph=True)

                grad = x_plus_noise.grad.data.cpu().numpy()

                if self.magnitutde:
                    total_gradients += (grad * grad)
                else:
                    total_gradients += grad
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples

            return avg_gradients

        elif self.types == 'e':
            # print('host')
            x_e = x_e.data.cpu().numpy()
            stdev = self.stdev_spread * (np.max(x_e) - np.min(x_e))
            total_gradients = np.zeros_like(x_e)
            for i in range(self.n_samples):
                noise = np.random.normal(0, stdev, x_e.shape).astype(np.float32)
                x_plus_noise = x_e + noise

                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
                output = self.pretrained_model(x_a, x_b, x_c, x_d, x_plus_noise)

                if index is None:
                    index = np.argmax(output.data.cpu().numpy())

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][index] = 1
                one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
                one_hot = torch.sum(one_hot * output)

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()
                one_hot.backward(retain_graph=True)

                grad = x_plus_noise.grad.data.cpu().numpy()

                if self.magnitutde:
                    total_gradients += (grad * grad)
                else:
                    total_gradients += grad
            avg_gradients = total_gradients[0, :, :, :] / self.n_samples

            return avg_gradients
