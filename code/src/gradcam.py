import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from config import opt
from PIL import Image
from torchvision import transforms as T
import models
import pickle
from pathlib import Path

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x_guest, x_host, type):
        target_activations = []
        for name, module in self.model._modules.items():
            # print(name)
            if name == 'top':
                guest_out = x_guest.view(x_guest.size(0), -1)
                host_out = x_host.view(x_host.size(0), -1)
                interactive = torch.cat((host_out, guest_out), 1)
                x = module(interactive)
            elif name == 'featureshost' and type == 'guest':
                x_host = module(x_host)
            elif name == 'featuresguest' and type == 'host':
                x_guest = module(x_guest)
            elif name == 'featureshost' and type == 'host':
                target_activations, x_host = self.feature_extractor(x_host)
            elif name == 'featuresguest' and type == 'guest':
                target_activations, x_guest = self.feature_extractor(x_guest)
            else:
                print('wrong!')
        
        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(mask, type):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap
    filename = 'outputcam/cam_' + type + '.png'
    cv2.imwrite(filename, np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
            self.device = 'cuda'
        else:
            self.model = model
        self.feature_module = feature_module
        self.model.eval()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_guest, input_host):
        return self.model(input_guest, input_host)

    def __call__(self, input_guest, input_host, index=None, type='guest'):
        if self.cuda:
            features, output = self.extractor(input_guest.cuda(), input_host.cuda(), type)
        else:
            features, output = self.extractor(input_guest, input_host, type)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val_np = self.extractor.get_gradients()[-1].cpu().data.numpy()
        grads_val = self.extractor.get_gradients()[-1]
        target_np = features[-1].cpu().data.numpy()[0, :]
        target = features[-1][0, :]
        
        weights_np = np.mean(grads_val_np, axis=(2, 3))[0, :]
        weights = torch.mean(grads_val, dim=(2, 3), keepdim=False)[0, :]
        cam_np = np.zeros(target_np.shape[1:], dtype=np.float32)
        cam = torch.zeros(target.size()[1:], dtype=torch.float32).to(self.device)

        for i, w in enumerate(weights_np):
            cam_np += w * target_np[i, :, :]

        for i, w in enumerate(weights):
            cam += w * target[i, :, :].requires_grad_(True)
        cam_np -= np.min(cam_np)
        cam_np /= np.max(cam_np)
        return torch.abs(cam.mean())

def main(**kwargs):
    opt.parse(kwargs)
    model = getattr(models, opt.model)()
    model.load(opt.load_model_path)
    
    img1_pkl = open('/data/img_guest.pkl', 'rb')
    imgs1 = pickle.load(img1_pkl)
    img1_pkl.close()
    res1_pkl = open('/data/res_guest.pkl', 'rb')
    res1 = pickle.load(res1_pkl)
    res1_pkl.close()

    imgs2 = []
    for img1 in imgs1:
        img2 = res1[str(img1)].replace('guest', 'host')
        imgs2.append(Path(img2))

    right = 0
    wrong = 0
    normalize = T.Normalize(mean = [0.5], 
                            std = [0.2])
    transforms = T.Compose([

            T.ToTensor(),
            normalize
            ])
    for ip in range(1):
        index = 777
        img_path1 = str(imgs1[index])
        img_path2 = str(imgs2[index])
        data1 = Image.open(img_path1)
        data2 = Image.open(img_path2)
        data1 = transforms(data1)
        data2 = transforms(data2)
        target_index = None
        data1 = torch.unsqueeze(data1, 0)
        data2 = torch.unsqueeze(data2, 0)

        for i in range(2):
            if i == 0:
                type = 'guest'
                f_model = model.featuresguest
            elif i == 1:
                type = 'host'
                f_model = model.featureshost

            grad_cam = GradCam(model=model, feature_module=f_model, \
                            target_layer_names=["7"], use_cuda=True)
            mask = grad_cam(data1, data2, target_index, type=type)
            print(mask)


def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default=None,
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

if __name__ == '__main__':

    import fire
    fire.Fire()
