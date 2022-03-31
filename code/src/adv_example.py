import torch
from torch.autograd import Variable
from data.dataset import mnist, vehicle, student, NusWide, credit, mnist_alpha, mnist_multi, mnist_multi_5
from torch.utils.data import DataLoader
import models
import torch.nn as nn
from torchvision import transforms as T
from torchnet import meter
from utils.visualize import Visualizer
import numpy as np
import cv2
import argparse
import models
from PIL import Image
from pathlib import Path
import pickle
import pandas as pd
import copy
from gradcam import GradCam

from data.load_nus_wide import load_prepared_parties_data

from saliency import iccv17, grad_var, SmoothGrad, SmoothGrad_Multi_3, iccv17_Multi_3, grad_var_multi_3, SmoothGrad_Multi_5, iccv17_Multi_5, grad_var_multi_5, iccv17_B

from tqdm import tqdm
import random
torch.multiprocessing.set_sharing_strategy('file_system')
vis = Visualizer('vfl')

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='VFLMnist', required=False, help="Which network?")
    parser.add_argument('--gpu', action="store_true", default=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--target', type=int, required=False, default=None)

    parser.add_argument('--guest_data', type=str)
    parser.add_argument('--host_data', type=str)
    parser.add_argument('--guest_data_list', type=str)
    parser.add_argument('--guest_data_num', type=int, default=1)
    parser.add_argument('--host_data_list')
    parser.add_argument('--host_data_num', type=int, default=2000)

    parser.add_argument('--vehicle_guest_list', type=str)
    parser.add_argument('--vehicle_host_list', type=str)
    parser.add_argument('--vehicle_model_path', type=str)
    
    parser.add_argument('--credit_guest_list', type=str)
    parser.add_argument('--credit_host_list', type=str)
    parser.add_argument('--credit_model_path', type=str)
    
    parser.add_argument('--student_guest_list', type=str)
    parser.add_argument('--student_host_list', type=str)
    parser.add_argument('--student_model_path', type=str)

    parser.add_argument('--load_model_path', type=str)

    parser.add_argument('--eps', type=float, default=100.0)
    parser.add_argument('--saliency', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--guest_index', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--subsample', type=int, default=20)

    return parser

def advtrain():
    parser = get_parser()
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    model_name = args.model
    target = args.target
    gpu = args.gpu

    if args.dataset == 'mnist':
        print('Model : %s \n' %(model_name))

        if args.alpha != 0.5 and args.alpha != 0.65 and (args.saliency == 'fuzz_with_saliency_alpha' or args.saliency == 'test_alpha' or args.saliency == 'test_alpha_B' or args.saliency == 'whitebox_bounded_alpha' or args.saliency == 'whitebox_alpha'):
            load_model_path = args.load_model_path + str(int(args.alpha * 10)) + '.pth'
            model = getattr(models, model_name)(args.alpha).eval()
        elif args.alpha == 0.65:
            print('65 loaded')
            load_model_path = args.load_model_path + '65' + '.pth'
            model = getattr(models, model_name)(args.alpha).eval()
        else:
            model = getattr(models, model_name)().eval()
            load_model_path = args.load_model_path
        if args.saliency == 'multi_fuzz_3' or args.saliency == 'multi_whitebox_3' or args.saliency == 'test_multi_3' or args.saliency == 'multi_whitebox_bounded_3':
            load_model_path = 'model_path'

        if args.saliency == 'multi_fuzz_5' or args.saliency == 'multi_whitebox_5' or args.saliency == 'test_multi_5' or args.saliency == 'multi_whitebox_bounded_5':
            load_model_path = 'model_path'

        model.load(load_model_path)
        device = 'cuda:' + str(args.device) if gpu else 'cpu'
        model.to(device)
        transformer = T.Compose([T.ToTensor(),])
        
        guest_data_list_pkl = open(args.guest_data_list, 'rb')
        guest_data_list = pickle.load(guest_data_list_pkl)
        guest_data_list_pkl.close()
        guest_data_list = np.random.permutation(guest_data_list)
        for dataindex in range(args.guest_data_num):
            guest_path = guest_data_list[dataindex]
            orig_host_path = str(guest_path).replace('guest', 'host')
            host_data_list_pkl = open(args.host_data_list, 'rb')
            host_data_list = pickle.load(host_data_list_pkl)
            host_data_list_pkl.close()
            host_data_list = np.random.permutation(host_data_list)
            host_data_list = np.insert(host_data_list, 0, orig_host_path)
            host_data_for_test = host_data_list.copy()[args.host_data_num:]
            host_data_list = host_data_list[:args.host_data_num]
            criterion = nn.CrossEntropyLoss()
            eps = args.eps
            alpha1 = 10.0
            alpha2 = 1.0
            iter_num = 100
            rho = 10
            
            if args.saliency == 'fuzz_with_saliency':
                train_data_mnist = mnist(None, mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=8)
                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 1000
                for S_index in tqdm(range(0*S_size, 1*S_size)):
                    S_guestpath = guest_data_list[S_index]
                    S.append(S_guestpath)
                    S_hostpath = Path(str(S_guestpath).replace('guest', 'host'))
                    S_orighost.append(S_hostpath)
                    S_times.append(0)
                    slabel = int(str(S_guestpath).split('/')[-2])
                    S_label.append(slabel)
                    S_guest = transformer(np.array(Image.open(S_guestpath)).astype(float)/255)
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)
                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                    Q.append(copy.deepcopy(S_guest.cpu().data.numpy()))
                print(S_acc)
                S_index = np.argsort(-np.array(S_acc))
                S = list(np.array(S)[S_index])
                S_label = list(np.array(S_label)[S_index])
                S_times = list(np.array(S_times)[S_index])
                S_acc = list(np.array(S_acc)[S_index])
                S_orighost = list(np.array(S_orighost)[S_index])
                Q = list(np.array(Q)[S_index])
                smooth_grad_guest = SmoothGrad(model, device, types='guest')
                smooth_grad_host = SmoothGrad(model, device, types='host')
                store_adi = []
                for iter in tqdm(range(5000)):
                    if iter % 3 == 0:
                        choice = random.randint(1, len(S)-1)
                    else:
                        choice = 0
                    inp_guest = torch.from_numpy(Q[choice])
                    inp_guest = Variable(inp_guest.to(device).float(), requires_grad=True)
                    temp_host_data = transformer(np.array(Image.open(S_orighost[choice])).astype(float)/255)
                    ori_inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=False)
                    pred = S_label[choice]
                    orig_mask = iccv17(model, inp_guest, device, ori_inp_host, pred)

                    tt_inp_guest = inp_guest.clone()
                    tt_max_acc = S_acc[choice]
                    for kk in range(1, args.subsample):
                        host_path = host_data_list[kk]
                        temp_host_data = transformer(np.array(Image.open(host_path)).astype(float)/255)
                        inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=True)
                        outputs = torch.nn.Softmax(dim=1)(model(tt_inp_guest, inp_host))
                        category = np.argmax(outputs.cpu().data.numpy())
                        low_saliency_socre = grad_var(model, tt_inp_guest, device, inp_host, outputs, fixedcategory=category)
                        low_noised_data = tt_inp_guest
                        for noise_index in range(4):
                            noise_mask = torch.zeros(tt_inp_guest.size())
                            if noise_index % 4 == 0:
                                noise_mask[0,0,:14,:7] = 1
                            elif noise_index % 4 == 1:
                                noise_mask[0,0,14:,:7] = 1
                            elif noise_index % 4 == 2:
                                noise_mask[0,0,:14,7:] = 1
                            elif noise_index % 4 == 3:
                                noise_mask[0,0,14:,7:] = 1
                            noised_inp_guest = tt_inp_guest + (torch.randn(tt_inp_guest.size()) * noise_mask).to(device) * 0.05
                            noised_inp_guest.data.clamp_(0, 1)
                            outputs = torch.nn.Softmax(dim=1)(model(noised_inp_guest, inp_host))
                            temp_score = grad_var(model, noised_inp_guest, device, inp_host, outputs, fixedcategory=category)
                            if temp_score < low_saliency_socre:
                                low_saliency_socre = temp_score
                                low_noised_data = noised_inp_guest
                            del outputs
                        noised_inp_guest = low_noised_data
                        saliency_guest_map = smooth_grad_guest(low_noised_data, inp_host, category)
                        saliency_host_map = smooth_grad_host(low_noised_data, inp_host, category)
                        low_saliency_socre = np.sum(saliency_host_map) / np.sum(saliency_guest_map)

                        if inp_host.grad is not None:
                            inp_host.grad.data.zero_()
                        if category == pred:
                            new_mask = iccv17(model, noised_inp_guest, device, inp_host, category)
                            mask = 1 + (1 - new_mask) * 0.2
                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp.data.clamp_(0, 1)
                            new_acc = advtest(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_guest_map = smooth_grad_guest(tt_inp, inp_host, category)
                            saliency_host_map = smooth_grad_host(tt_inp, inp_host, category)
                            temp_score = np.sum(saliency_host_map) / np.sum(saliency_guest_map)

                            if temp_score < low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(0, 1)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())

                                img = cv2.cvtColor(img_numpy[0, 0], cv2.COLOR_GRAY2BGR)
                                orig_mask = iccv17(model, tt_inp_guest, device, ori_inp_host, pred)

                        else:
                            new_mask = iccv17(model, noised_inp_guest, device, inp_host, category)
                            mask = (1.0 - orig_mask.mul(1.0 - new_mask)) + (1 - orig_mask) * 0.3

                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp.data.clamp_(0, 1)
                            new_acc = advtest(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_guest_map = smooth_grad_guest(tt_inp, inp_host, category)
                            saliency_host_map = smooth_grad_host(tt_inp, inp_host, category)
                            temp_score = np.sum(saliency_host_map) / np.sum(saliency_guest_map)
                            if temp_score < low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(0, 1)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                                img = cv2.cvtColor(img_numpy[0, 0], cv2.COLOR_GRAY2BGR)
                                orig_mask = iccv17(model, tt_inp_guest, device, ori_inp_host, pred)

                    if tt_max_acc == S_acc[choice]:
                        S_times[choice] += 1

                    print("Iter ", iter, ": acc ", tt_max_acc)

                    if tt_max_acc >= 95:
                        A.append(img)

                        print('found: ', len(A))

                        print('-----------------')
                        print(S[choice])
                        print('-----------------')
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_orighost[choice]
                        del Q[choice]
                    elif tt_max_acc > S_acc[choice]:
                        Q[choice] = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                        S_acc[choice] = tt_max_acc
                        S_times[choice] = 0

                        S_index = np.argsort(-np.array(S_acc))
                        S = list(np.array(S)[S_index])
                        S_label = list(np.array(S_label)[S_index])
                        S_times = list(np.array(S_times)[S_index])
                        S_acc = list(np.array(S_acc)[S_index])
                        S_orighost = list(np.array(S_orighost)[S_index])
                        Q = list(np.array(Q)[S_index])

                    elif S_times[choice] >= 5:
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_orighost[choice]
                        del Q[choice]

                    if len(S) <= 50 or len(A) >= 200:
                        break
            
            elif args.saliency == 'test_orig_rate':
                train_data_mnist = mnist(None, mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)

                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 1000
                for S_index in tqdm(range(0*S_size, 1*S_size)):
                    S_guestpath = guest_data_list[S_index]
                    S.append(S_guestpath)
                    S_hostpath = Path(str(S_guestpath).replace('guest', 'host'))
                    S_orighost.append(S_hostpath)
                    slabel = int(str(S_guestpath).split('/')[-2])
                    S_label.append(slabel)
                    S_guest = transformer(np.array(Image.open(S_guestpath)).astype(float)/255)
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)
                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                ADI_num = 0.0
                total_num = len(S_acc)

                for temp_acc in S_acc:
                    if temp_acc >= 99:
                        ADI_num += 1
                print('=' * 10)
                result = ADI_num / float(total_num)
                print(result)

            elif args.saliency == 'whitebox':
                S_size = 1000
                train_data_mnist = mnist(None, mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 0.1

                for S_index in tqdm(range(1*S_size, 2*S_size)):
                    S_guestpath = guest_data_list[S_index]
                    S_hostpath = Path(str(S_guestpath).replace('guest', 'host'))
                    slabel = int(str(S_guestpath).split('/')[-2])
                    S_guest = transformer(np.array(Image.open(S_guestpath)).astype(float)/255)
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    temp_host_data = transformer(np.array(Image.open(S_hostpath)).astype(float)/255)
                    ori_inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=False)
                    
                    out = model(S_guest, ori_inp_host)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    
                    for kk in range(args.subsample):
                        host_path = host_data_list[kk]
                        temp_host_data = transformer(np.array(Image.open(host_path)).astype(float)/255)
                        inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=True)
                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            if loss1.item() < 0.01:
                                break
                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(torch.var(out), inp_host, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_x2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)
                            
                            grad = grady_x1_loss1 * 0.999 + grady_x1_loss2[0] * 0.001
                            perturbation = -lr * grad

                            np_guest.data += perturbation

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_host

                    temp_acc = advtest(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                        
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)
                    
            elif args.saliency == 'whitebox_bounded':
                S_size = 1000
                train_data_mnist = mnist(None, mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 0.1
                for S_index in tqdm(range(0*S_size, 1*S_size)):
                    S_guestpath = guest_data_list[S_index]
                    S_hostpath = Path(str(S_guestpath).replace('guest', 'host'))

                    slabel = int(str(S_guestpath).split('/')[-2])

                    S_guest = transformer(np.array(Image.open(S_guestpath)).astype(float)/255)
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    temp_host_data = transformer(np.array(Image.open(S_hostpath)).astype(float)/255)
                    ori_inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=False)
                    
                    out = model(S_guest, ori_inp_host)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    orig = inp_guest.clone()
                    for kk in range(1, args.subsample):
                        host_path = host_data_list[kk]
                        temp_host_data = transformer(np.array(Image.open(host_path)).astype(float)/255)
                        inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=True)
                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(torch.var(out), inp_host, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_x2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)
                            
                            grad = grady_x1_loss1 * 0.7 + grady_x1_loss2[0] * 0.3

                            perturbation = -lr * grad

                            inp_guest.data += perturbation
                            inp_guest.data = torch.clamp(inp_guest.data, min = 0.0, max = 1.0)
                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_host

                    temp_acc = advtest(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                        
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)
                
            elif args.saliency == 'fuzz_with_saliency_alpha':
                train_data_mnist = mnist_alpha(alpha=args.alpha, mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 1000
                for S_index, (S_guest, S_hostpath, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                    Q.append(copy.deepcopy(S_guest.cpu().data.numpy()))
                    S_label.append(slabel)
                    S_orighost.append(S_hostpath.data.numpy())
                    S.append(S_index)
                    S_times.append(0)
                    if S_index >= S_size:
                        break
                print(S_acc)
                S_index = np.argsort(-np.array(S_acc))
                S = list(np.array(S)[S_index])
                S_label = list(np.array(S_label)[S_index])
                S_times = list(np.array(S_times)[S_index])
                S_acc = list(np.array(S_acc)[S_index])
                S_orighost = list(np.array(S_orighost)[S_index])
                Q = list(np.array(Q)[S_index])
                smooth_grad_guest = SmoothGrad(model, device, types='guest')
                smooth_grad_host = SmoothGrad(model, device, types='host')

                for iter in tqdm(range(5000)):
                    if iter % 3 == 0:
                        choice = random.randint(1, len(S)-1)
                    else:
                        choice = 0
                    inp_guest = torch.from_numpy(Q[choice])
                    inp_guest = Variable(inp_guest.to(device).float(), requires_grad=True)
                    temp_host_data = torch.from_numpy(S_orighost[choice])
                    ori_inp_host = Variable(temp_host_data.to(device).float(), requires_grad=False)
                    pred = S_label[choice]
                    orig_mask = iccv17(model, inp_guest, device, ori_inp_host, pred)

                    tt_inp_guest = inp_guest.clone()
                    tt_max_acc = S_acc[choice]

                    for kk, (_, inp_host, _) in enumerate(fuzz_dataloader):
                        if tt_max_acc >= 95:
                            img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                            break
                        if kk >= 20:
                            break

                        inp_host = Variable(inp_host).to(device)
                        inp_host.requires_grad_(True)
                        outputs = torch.nn.Softmax(dim=1)(model(tt_inp_guest, inp_host))
                        category = np.argmax(outputs.cpu().data.numpy())

                        low_saliency_socre = grad_var(model, tt_inp_guest, device, inp_host, outputs, fixedcategory=category)

                        low_noised_data = tt_inp_guest
                        for noise_index in range(4):
                            noise_mask = torch.zeros(tt_inp_guest.size())
                            if noise_index % 4 == 0:
                                noise_mask[0,0,:14,:7] = 1
                            elif noise_index % 4 == 1:
                                noise_mask[0,0,14:,:7] = 1
                            elif noise_index % 4 == 2:
                                noise_mask[0,0,:14,7:] = 1
                            elif noise_index % 4 == 3:
                                noise_mask[0,0,14:,7:] = 1
                            noised_inp_guest = tt_inp_guest + (torch.randn(tt_inp_guest.size()) * noise_mask).to(device) * 0.05
                            noised_inp_guest.data.clamp_(0, 1)
                            outputs = torch.nn.Softmax(dim=1)(model(noised_inp_guest, inp_host))
                            temp_score = grad_var(model, noised_inp_guest, device, inp_host, outputs, fixedcategory=category)

                            if temp_score < low_saliency_socre:
                                low_saliency_socre = temp_score
                                low_noised_data = noised_inp_guest
                            del outputs
                        noised_inp_guest = low_noised_data
                        saliency_guest_map = smooth_grad_guest(low_noised_data, inp_host, category)
                        saliency_host_map = smooth_grad_host(low_noised_data, inp_host, category)
                        low_saliency_socre = np.sum(saliency_host_map) / (np.sum(saliency_guest_map) + 1e-10)

                        if inp_host.grad is not None:
                            inp_host.grad.data.zero_()
                        if category == pred:
                            new_mask = iccv17(model, noised_inp_guest, device, inp_host, category)
                            mask = 1 + (1 - new_mask) * 0.2

                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp.data.clamp_(0, 1)
                            new_acc = advtest(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_guest_map = smooth_grad_guest(tt_inp, inp_host, category)
                            saliency_host_map = smooth_grad_host(tt_inp, inp_host, category)
                            temp_score = np.sum(saliency_host_map) / (np.sum(saliency_guest_map) + 1e-10)

                            if temp_score <= low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(0, 1)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                                
                                orig_mask = iccv17(model, tt_inp_guest, device, ori_inp_host, pred)

                        else:
                            new_mask = iccv17(model, noised_inp_guest, device, inp_host, category)

                            mask = (1.0 - orig_mask.mul(1.0 - new_mask)) + (1 - orig_mask) * 0.3
                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp.data.clamp_(0, 1)
                            new_acc = advtest(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_guest_map = smooth_grad_guest(tt_inp, inp_host, category)
                            saliency_host_map = smooth_grad_host(tt_inp, inp_host, category)
                            temp_score = np.sum(saliency_host_map) / (np.sum(saliency_guest_map) + 1e-10)
                            if temp_score <= low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(0, 1)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                                orig_mask = iccv17(model, tt_inp_guest, device, ori_inp_host, pred)

                    if tt_max_acc == S_acc[choice]:
                        S_times[choice] += 1

                    print("Iter ", iter, ": acc ", tt_max_acc)

                    if tt_max_acc >= 95:
                        A.append(img_numpy)
                        print('found: ', len(A))
                        print('-----------------')
                        print(S[choice])
                        print('-----------------')
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_orighost[choice]
                        del Q[choice]
                    elif tt_max_acc > S_acc[choice]:
                        Q[choice] = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                        S_acc[choice] = tt_max_acc
                        S_times[choice] = 0

                        S_index = np.argsort(-np.array(S_acc))
                        S = list(np.array(S)[S_index])
                        S_label = list(np.array(S_label)[S_index])
                        S_times = list(np.array(S_times)[S_index])
                        S_acc = list(np.array(S_acc)[S_index])
                        S_orighost = list(np.array(S_orighost)[S_index])
                        Q = list(np.array(Q)[S_index])

                    elif S_times[choice] >= 5:
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_orighost[choice]
                        del Q[choice]

                    if len(S) <= 0:
                        break
            
            elif args.saliency == 'test_alpha':
                train_data_mnist = mnist_alpha(alpha=args.alpha, mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 1000
                ADI_num = 0.0
                for S_index, (S_guest, S_hostpath, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                    Q.append(copy.deepcopy(S_guest.cpu().data.numpy()))
                    S_label.append(slabel)
                    S_orighost.append(S_hostpath.data.numpy())
                    S.append(S_index)
                    S_times.append(0)
                    if tempacc >= 95:
                        ADI_num += 1.0
                    
                    if S_index >= S_size:
                        print('=' * 10)
                        result = ADI_num / float(S_size)
                        print(result)
                        break
            
            elif args.saliency == 'test_alpha_B':
                train_data_mnist = mnist_alpha(alpha=args.alpha, mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 1000
                ADI_num = 0.0
                for S_index, (S_guestpath, S_host, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    S_host = Variable(S_host, requires_grad=False).to(device)
                    tempacc = advtest_alpha(model, S_host, train_dataloader, slabel, slabel, args.dataset)
                    if tempacc >= 95:
                        ADI_num += 1.0
                    
                    if S_index >= S_size:
                        print('=' * 10)
                        result = ADI_num / float(S_size)
                        print(result)
                        break

            elif args.saliency == 'whitebox_bounded_alpha':
                train_data_mnist = mnist_alpha(alpha=args.alpha, mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S_size = 1000

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 0.1
                for S_index, (S_guest, S_hostpath, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    if S_index >= S_size:
                        break
                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    ori_inp_host = Variable(S_hostpath, requires_grad=False).to(device)
                    out = model(S_guest, ori_inp_host)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    orig = inp_guest.clone()
                    for kk, (_, inp_host, _) in enumerate(fuzz_dataloader):
                        if kk > 30:
                            break
                        inp_host = Variable(inp_host).to(device)
                        inp_host.requires_grad_(True)
                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            if loss1.item() < 0.01:
                                break
                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(torch.var(out), inp_host, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_x2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)
                            
                            grad = grady_x1_loss1 * 0.7 + grady_x1_loss2[0] * 0.3
                            # grad = grady_x1_loss1
                            perturbation = -lr * grad

                            inp_guest.data += perturbation
                            inp_guest.data = torch.clamp(inp_guest.data, min = 0.0, max = 1.0)
                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_host

                    temp_acc = advtest(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            elif args.saliency == 'whitebox_alpha':
                train_data_mnist = mnist_alpha(alpha=args.alpha, mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S_size = 1000

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 0.1
                for S_index, (S_guest, S_hostpath, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    if S_index >= S_size:
                        break
                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    
                    ori_inp_host = Variable(S_hostpath, requires_grad=False).to(device)
                    out = model(S_guest, ori_inp_host)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    orig = inp_guest.clone()
                    for kk, (_, inp_host, _) in enumerate(fuzz_dataloader):
                        if kk > 30:
                            break
                        inp_host = Variable(inp_host).to(device)
                        inp_host.requires_grad_(True)
                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(torch.var(out), inp_host, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_x2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)
                              
                            grad = grady_x1_loss1 * 0.999 + grady_x1_loss2[0] * 0.001
                            # grad = grady_x1_loss1
                            perturbation = -lr * grad

                            inp_guest.data += perturbation
                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_host

                    temp_acc = advtest(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                    print(temp_acc, total_num)
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            elif args.saliency == 'SVD':
                S_size = 1
                train_data_mnist = mnist(None, mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 10
                lr = 0.3

                for S_index in tqdm(range(0*S_size, 1*S_size)):
                    mutation = []
                    S_guestpath = Path('/data/mnist_test/image.png')
                    S_hostpath = Path(str(S_guestpath).replace('guest', 'host'))
                    slabel = int(str(S_guestpath).split('/')[-2])
                    S_guest = transformer(np.array(Image.open(S_guestpath)).astype(float)/255)
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)
                    temp_host_data = transformer(np.array(Image.open(S_hostpath)).astype(float)/255)
                    ori_inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=False)

                    out = model(S_guest, ori_inp_host)
                    pred = np.argmax(out.data.cpu().numpy())
                    print(S_guestpath)
                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    orig = inp_guest.clone()
                    for kk in tqdm(range(1000)):
                        inp_guest = orig.clone().detach().requires_grad_(True)
                        # print(kk)
                        host_path = host_data_list[kk]
                        temp_host_data = transformer(np.array(Image.open(host_path)).astype(float)/255)
                        inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=True)
                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(torch.var(out), inp_host, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_x2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)
                             
                            grad = grady_x1_loss1 * 0.999 + grady_x1_loss2[0] * 0.001
                            # grad = grady_x1_loss1
                            perturbation = -lr * grad

                            inp_guest.data += perturbation
                            inp_guest.data = torch.clamp(inp_guest.data, min = 0.0, max = 1.0)
                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_host

                        temp_mutation = inp_guest.cpu().data.numpy().squeeze(0).squeeze(0) - orig.cpu().data.numpy().squeeze(0).squeeze(0)
                        mutation.append(temp_mutation.flatten() / np.linalg.norm(temp_mutation.flatten()))
                    mutation = np.array(mutation)
                    mutation = mutation.T
                    u, s, vh = np.linalg.svd(mutation)
                    s_file = './results/SVD_s_' + str(slabel) + '.npy'
                    np.save(s_file, s)

            elif args.saliency == 'multi_fuzz_3':
                train_data_mnist = mnist_multi(mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_b = []
                S_c = []
                Q = []
                A = []
                S_size = 1000
                for S_index, (S_guest, x_b, x_c, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest_multi_3(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                    Q.append(copy.deepcopy(S_guest.cpu().data.numpy()))
                    S_label.append(slabel)
                    S_b.append(x_b.data.numpy())
                    S_c.append(x_c.data.numpy())
                    S.append(S_index)
                    S_times.append(0)
                    if S_index >= S_size:
                        break
                
                print(S_acc)
                S_index = np.argsort(-np.array(S_acc))
                S = list(np.array(S)[S_index])
                S_label = list(np.array(S_label)[S_index])
                S_times = list(np.array(S_times)[S_index])
                S_acc = list(np.array(S_acc)[S_index])
                S_b = list(np.array(S_b)[S_index])
                S_c = list(np.array(S_c)[S_index])
                Q = list(np.array(Q)[S_index])
                smooth_grad_guest = SmoothGrad_Multi_3(model, device, types='a')
                smooth_grad_b = SmoothGrad_Multi_3(model, device, types='b')
                smooth_grad_c = SmoothGrad_Multi_3(model, device, types='c')

                for iter in tqdm(range(5000)):
                    if iter % 3 == 0:
                        choice = random.randint(1, len(S)-1)
                    else:
                        choice = 0
                    inp_guest = torch.from_numpy(Q[choice])
                    inp_guest = Variable(inp_guest.to(device).float(), requires_grad=True)
                    temp_b_data = torch.from_numpy(S_b[choice])
                    ori_inp_b = Variable(temp_b_data.to(device).float(), requires_grad=False)
                    temp_c_data = torch.from_numpy(S_c[choice])
                    ori_inp_c = Variable(temp_c_data.to(device).float(), requires_grad=False)
                    pred = S_label[choice]
                    orig_mask = iccv17_Multi_3(model, inp_guest, device, ori_inp_b, ori_inp_c, pred)

                    tt_inp_guest = inp_guest.clone()
                    tt_max_acc = S_acc[choice]

                    for kk, (_, inp_b, inp_c, _) in enumerate(fuzz_dataloader):
                        if tt_max_acc >= 95:
                            img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                            break
                        if kk >= 20:
                            break
                        inp_b = Variable(inp_b).to(device)
                        inp_b.requires_grad_(True)
                        inp_c = Variable(inp_c).to(device)
                        inp_c.requires_grad_(True)

                        outputs = torch.nn.Softmax(dim=1)(model(tt_inp_guest, inp_b, inp_c))
                        category = np.argmax(outputs.cpu().data.numpy())
                        low_saliency_socre = grad_var_multi_3(model, tt_inp_guest, device, inp_b, inp_c, outputs, fixedcategory=category)

                        low_noised_data = tt_inp_guest
                        for noise_index in range(4):
                            noise_mask = torch.zeros(tt_inp_guest.size())
                            if noise_index % 4 == 0:
                                noise_mask[0,0,:14,:7] = 1
                            elif noise_index % 4 == 1:
                                noise_mask[0,0,14:,:7] = 1
                            elif noise_index % 4 == 2:
                                noise_mask[0,0,:14,7:] = 1
                            elif noise_index % 4 == 3:
                                noise_mask[0,0,14:,7:] = 1
                            noised_inp_guest = tt_inp_guest + (torch.randn(tt_inp_guest.size()) * noise_mask).to(device) * 0.05
                            noised_inp_guest.data.clamp_(0, 1)
                            outputs = torch.nn.Softmax(dim=1)(model(noised_inp_guest, inp_b, inp_c))
                            temp_score = grad_var_multi_3(model, noised_inp_guest, device, inp_b, inp_c, outputs, fixedcategory=category)

                            if temp_score < low_saliency_socre:
                                low_saliency_socre = temp_score
                                low_noised_data = noised_inp_guest
                            del outputs
                        noised_inp_guest = low_noised_data
                        saliency_a_map = smooth_grad_guest(low_noised_data, inp_b, inp_c, category)
                        saliency_b_map = smooth_grad_b(low_noised_data, inp_b, inp_c, category)
                        saliency_c_map = smooth_grad_c(low_noised_data, inp_b, inp_c, category)
                        low_saliency_socre = (np.sum(saliency_b_map) + np.sum(saliency_c_map)) / (np.sum(saliency_a_map) + 1e-10)

                        if inp_b.grad is not None:
                            inp_b.grad.data.zero_()
                        if inp_c.grad is not None:
                            inp_c.grad.data.zero_()
    
                        if category == pred:
                            new_mask = iccv17_Multi_3(model, noised_inp_guest, device, inp_b, inp_c, category)
                            mask = 1 + (1 - new_mask) * 0.2

                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp.data.clamp_(0, 1)
                            new_acc = advtest_multi_3(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_a_map = smooth_grad_guest(tt_inp, inp_b, inp_c, category)
                            saliency_b_map = smooth_grad_b(tt_inp, inp_b, inp_c, category)
                            saliency_c_map = smooth_grad_c(tt_inp, inp_b, inp_c, category)
                            low_saliency_socre = (np.sum(saliency_b_map) + np.sum(saliency_c_map)) / (np.sum(saliency_a_map) + 1e-10)

                            if temp_score <= low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(0, 1)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())

                                orig_mask = iccv17_Multi_3(model, tt_inp_guest, device, ori_inp_b, ori_inp_c, pred)

                        else:
                            new_mask = iccv17_Multi_3(model, noised_inp_guest, device, inp_b, inp_c, category)
                            mask = (1.0 - orig_mask.mul(1.0 - new_mask)) + (1 - orig_mask) * 0.3
                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp.data.clamp_(0, 1)
                            new_acc = advtest_multi_3(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_a_map = smooth_grad_guest(tt_inp, inp_b, inp_c, category)
                            saliency_b_map = smooth_grad_b(tt_inp, inp_b, inp_c, category)
                            saliency_c_map = smooth_grad_c(tt_inp, inp_b, inp_c, category)
                            low_saliency_socre = (np.sum(saliency_b_map) + np.sum(saliency_c_map)) / (np.sum(saliency_a_map) + 1e-10)

                            if temp_score <= low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(0, 1)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())

                                orig_mask = iccv17_Multi_3(model, tt_inp_guest, device, ori_inp_b, ori_inp_c, pred)

                    if tt_max_acc == S_acc[choice]:
                        S_times[choice] += 1
                    print("Iter ", iter, ": acc ", tt_max_acc)

                    if tt_max_acc >= 95:
                        A.append(img_numpy)
                        print('found: ', len(A))
                        print('-----------------')
                        print(S[choice])
                        print('-----------------')
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_b[choice]
                        del Q[choice]
                    elif tt_max_acc > S_acc[choice]:
                        Q[choice] = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                        S_acc[choice] = tt_max_acc
                        S_times[choice] = 0

                        S_index = np.argsort(-np.array(S_acc))
                        S = list(np.array(S)[S_index])
                        S_label = list(np.array(S_label)[S_index])
                        S_times = list(np.array(S_times)[S_index])
                        S_acc = list(np.array(S_acc)[S_index])
                        S_b = list(np.array(S_b)[S_index])
                        Q = list(np.array(Q)[S_index])

                    elif S_times[choice] >= 5:
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_b[choice]
                        del Q[choice]

                    if len(S) <= 0:
                        break

            elif args.saliency == 'multi_whitebox_3':
                train_data_mnist = mnist_multi(mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S_size = 1000

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 0.1
                for S_index, (S_guest, S_b, S_c, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    if S_index >= S_size:
                        break

                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest_multi_3(model, S_guest, train_dataloader, slabel, slabel, args.dataset)

                    ori_inp_b = Variable(S_b, requires_grad=False).to(device)
                    ori_inp_c = Variable(S_c, requires_grad=False).to(device)
                    out = model(S_guest, ori_inp_b, ori_inp_c)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    orig = inp_guest.clone()
                    for kk, (_, inp_b, inp_c, _) in enumerate(fuzz_dataloader):
                        if kk > 20:
                            break
                        inp_b = Variable(inp_b).to(device)
                        inp_b.requires_grad_(True)
                        inp_c = Variable(inp_c).to(device)
                        inp_c.requires_grad_(True)
                        for i in range(iter_num):
                            out = model(inp_guest, inp_b, inp_c)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_b.grad.data.zero_()
                            inp_c.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_xb_2 = torch.autograd.grad(torch.var(out), inp_b, create_graph=True, retain_graph=True)
                            gradvar_xc_2 = torch.autograd.grad(torch.var(out), inp_c, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_xb_2[0]).mean() + torch.abs(gradvar_xc_2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)                        
                            
                            grad = grady_x1_loss1 * 0.999 + grady_x1_loss2[0] * 0.001
                            perturbation = -lr * grad

                            inp_guest.data += perturbation
                            inp_guest.grad.data.zero_()
                            inp_b.grad.data.zero_()
                            inp_c.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_b, inp_c

                    temp_acc = advtest_multi_3(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                    print(temp_acc, total_num)
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            elif args.saliency == 'multi_whitebox_bounded_3':
                train_data_mnist = mnist_multi(mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S_size = 1000

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 0.1
                for S_index, (S_guest, S_b, S_c, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    if S_index >= S_size:
                        break

                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest_multi_3(model, S_guest, train_dataloader, slabel, slabel, args.dataset)

                    ori_inp_b = Variable(S_b, requires_grad=False).to(device)
                    ori_inp_c = Variable(S_c, requires_grad=False).to(device)
                    out = model(S_guest, ori_inp_b, ori_inp_c)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    orig = inp_guest.clone()
                    for kk, (_, inp_b, inp_c, _) in enumerate(fuzz_dataloader):
                        if kk > 20:
                            break
                        inp_b = Variable(inp_b).to(device)
                        inp_b.requires_grad_(True)
                        inp_c = Variable(inp_c).to(device)
                        inp_c.requires_grad_(True)
                        for i in range(iter_num):
                            out = model(inp_guest, inp_b, inp_c)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_b.grad.data.zero_()
                            inp_c.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_xb_2 = torch.autograd.grad(torch.var(out), inp_b, create_graph=True, retain_graph=True)
                            gradvar_xc_2 = torch.autograd.grad(torch.var(out), inp_c, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_xb_2[0]).mean() + torch.abs(gradvar_xc_2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)                        
                            
                            grad = grady_x1_loss1 * 0.999 + grady_x1_loss2[0] * 0.001
                            perturbation = -lr * grad

                            inp_guest.data += perturbation
                            inp_guest.data = torch.clamp(inp_guest.data, min = 0.0, max = 1.0)
                            inp_guest.grad.data.zero_()
                            inp_b.grad.data.zero_()
                            inp_c.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_b, inp_c

                    temp_acc = advtest_multi_3(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                    print(temp_acc, total_num)
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            elif args.saliency == 'test_multi_3':
                train_data_mnist = mnist_multi(mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=1)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=1)
                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 1000
                ADI_num = 0.0
                for S_index, (S_guest, _, _, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest_multi_3(model, S_guest, train_dataloader, slabel, slabel, args.dataset)

                    if tempacc >= 95:
                        ADI_num += 1.0
                    
                    if S_index >= S_size:
                        print('=' * 10)
                        result = ADI_num / float(S_size)
                        print(result)
                        break

            elif args.saliency == 'multi_whitebox_5':
                train_data_mnist = mnist_multi_5(mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S_size = 1000

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 0.1
                for S_index, (S_guest, S_b, S_c, S_d, S_e, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    if S_index >= S_size:
                        break

                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest_multi_5(model, S_guest, train_dataloader, slabel, slabel, args.dataset)

                    ori_inp_b = Variable(S_b, requires_grad=False).to(device)
                    ori_inp_c = Variable(S_c, requires_grad=False).to(device)
                    ori_inp_d = Variable(S_d, requires_grad=False).to(device)
                    ori_inp_e = Variable(S_e, requires_grad=False).to(device)
                    out = model(S_guest, ori_inp_b, ori_inp_c, ori_inp_d, ori_inp_e)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    orig = inp_guest.clone()
                    for kk, (_, inp_b, inp_c, inp_d, inp_e, _) in enumerate(fuzz_dataloader):
                        if kk > 20:
                            break
                        inp_b = Variable(inp_b).to(device)
                        inp_b.requires_grad_(True)
                        inp_c = Variable(inp_c).to(device)
                        inp_c.requires_grad_(True)
                        inp_d = Variable(inp_d).to(device)
                        inp_d.requires_grad_(True)
                        inp_e = Variable(inp_e).to(device)
                        inp_e.requires_grad_(True)
                        for i in range(iter_num):
                            out = model(inp_guest, inp_b, inp_c, inp_d, inp_e)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_b.grad.data.zero_()
                            inp_c.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_xb_2 = torch.autograd.grad(torch.var(out), inp_b, create_graph=True, retain_graph=True)
                            gradvar_xc_2 = torch.autograd.grad(torch.var(out), inp_c, create_graph=True, retain_graph=True)
                            gradvar_xd_2 = torch.autograd.grad(torch.var(out), inp_d, create_graph=True, retain_graph=True)
                            gradvar_xe_2 = torch.autograd.grad(torch.var(out), inp_e, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_xb_2[0]).mean() + torch.abs(gradvar_xc_2[0]).mean() + torch.abs(gradvar_xd_2[0]).mean() + torch.abs(gradvar_xe_2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)                        
                            
                            grad = grady_x1_loss1 * 0.999 + grady_x1_loss2[0] * 0.001
                            perturbation = -lr * grad

                            inp_guest.data += perturbation
                            inp_guest.grad.data.zero_()
                            inp_b.grad.data.zero_()
                            inp_c.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_b, inp_c

                    temp_acc = advtest_multi_5(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                    print(temp_acc, total_num)
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            elif args.saliency == 'test_multi_5':
                train_data_mnist = mnist_multi_5(mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 1000
                ADI_num = 0.0
                for S_index, (S_guest, _, _, _, _, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest_multi_5(model, S_guest, train_dataloader, slabel, slabel, args.dataset)

                    if tempacc >= 95:
                        ADI_num += 1.0
                    
                    if S_index >= S_size:
                        print('=' * 10)
                        result = ADI_num / float(S_size)
                        print(result)
                        break

            elif args.saliency == 'multi_whitebox_bounded_5':
                train_data_mnist = mnist_multi_5(mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S_size = 1000

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 0.1
                for S_index, (S_guest, S_b, S_c, S_d, S_e, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    if S_index >= S_size:
                        break

                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest_multi_5(model, S_guest, train_dataloader, slabel, slabel, args.dataset)

                    ori_inp_b = Variable(S_b, requires_grad=False).to(device)
                    ori_inp_c = Variable(S_c, requires_grad=False).to(device)
                    ori_inp_d = Variable(S_d, requires_grad=False).to(device)
                    ori_inp_e = Variable(S_e, requires_grad=False).to(device)
                    out = model(S_guest, ori_inp_b, ori_inp_c, ori_inp_d, ori_inp_e)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    orig = inp_guest.clone()
                    for kk, (_, inp_b, inp_c, inp_d, inp_e, _) in enumerate(fuzz_dataloader):
                        if kk > 20:
                            break
                        inp_b = Variable(inp_b).to(device)
                        inp_b.requires_grad_(True)
                        inp_c = Variable(inp_c).to(device)
                        inp_c.requires_grad_(True)
                        inp_d = Variable(inp_d).to(device)
                        inp_d.requires_grad_(True)
                        inp_e = Variable(inp_e).to(device)
                        inp_e.requires_grad_(True)
                        for i in range(iter_num):
                            out = model(inp_guest, inp_b, inp_c, inp_d, inp_e)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_b.grad.data.zero_()
                            inp_c.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_xb_2 = torch.autograd.grad(torch.var(out), inp_b, create_graph=True, retain_graph=True)
                            gradvar_xc_2 = torch.autograd.grad(torch.var(out), inp_c, create_graph=True, retain_graph=True)
                            gradvar_xd_2 = torch.autograd.grad(torch.var(out), inp_d, create_graph=True, retain_graph=True)
                            gradvar_xe_2 = torch.autograd.grad(torch.var(out), inp_e, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_xb_2[0]).mean() + torch.abs(gradvar_xc_2[0]).mean() + torch.abs(gradvar_xd_2[0]).mean() + torch.abs(gradvar_xe_2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)                        
                            
                            grad = grady_x1_loss1 * 0.999 + grady_x1_loss2[0] * 0.001
                            perturbation = -lr * grad

                            inp_guest.data += perturbation
                            inp_guest.data = torch.clamp(inp_guest.data, min = 0.0, max = 1.0)
                            inp_guest.grad.data.zero_()
                            inp_b.grad.data.zero_()
                            inp_c.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_b, inp_c

                    temp_acc = advtest_multi_5(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                    print(temp_acc, total_num)
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            elif args.saliency == 'multi_fuzz_5':
                train_data_mnist = mnist_multi_5(mode='val', dataset='mnist')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)
                fuzz_dataloader = DataLoader(train_data_mnist, 1, shuffle=True, num_workers=4)
                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_b = []
                S_c = []
                S_d = []
                S_e = []
                Q = []
                A = []
                S_size = 1000
                for S_index, (S_guest, x_b, x_c, x_d, x_e, slabel) in tqdm(enumerate(fuzz_dataloader)):
                    S_guest = Variable(S_guest, requires_grad=False).to(device)
                    tempacc = advtest_multi_5(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                    Q.append(copy.deepcopy(S_guest.cpu().data.numpy()))
                    S_label.append(slabel)
                    S_b.append(x_b.data.numpy())
                    S_c.append(x_c.data.numpy())
                    S_d.append(x_d.data.numpy())
                    S_e.append(x_e.data.numpy())
                    S.append(S_index)
                    S_times.append(0)
                    if S_index >= S_size:
                        break
                
                print(S_acc)
                S_index = np.argsort(-np.array(S_acc))
                S = list(np.array(S)[S_index])
                S_label = list(np.array(S_label)[S_index])
                S_times = list(np.array(S_times)[S_index])
                S_acc = list(np.array(S_acc)[S_index])
                S_b = list(np.array(S_b)[S_index])
                S_c = list(np.array(S_c)[S_index])
                S_d = list(np.array(S_d)[S_index])
                S_e = list(np.array(S_e)[S_index])
                Q = list(np.array(Q)[S_index])
                smooth_grad_guest = SmoothGrad_Multi_5(model, device, types='a')
                smooth_grad_b = SmoothGrad_Multi_5(model, device, types='b')
                smooth_grad_c = SmoothGrad_Multi_5(model, device, types='c')
                smooth_grad_d = SmoothGrad_Multi_5(model, device, types='d')
                smooth_grad_e = SmoothGrad_Multi_5(model, device, types='e')

                for iter in tqdm(range(5000)):
                    if iter % 3 == 0:
                        choice = random.randint(1, len(S)-1)
                    else:
                        choice = 0
                    inp_guest = torch.from_numpy(Q[choice])
                    inp_guest = Variable(inp_guest.to(device).float(), requires_grad=True)
                    temp_b_data = torch.from_numpy(S_b[choice])
                    ori_inp_b = Variable(temp_b_data.to(device).float(), requires_grad=False)
                    temp_c_data = torch.from_numpy(S_c[choice])
                    ori_inp_c = Variable(temp_c_data.to(device).float(), requires_grad=False)
                    temp_d_data = torch.from_numpy(S_d[choice])
                    ori_inp_d = Variable(temp_d_data.to(device).float(), requires_grad=False)
                    temp_e_data = torch.from_numpy(S_e[choice])
                    ori_inp_e = Variable(temp_e_data.to(device).float(), requires_grad=False)
                    pred = S_label[choice]
                    orig_mask = iccv17_Multi_5(model, inp_guest, device, ori_inp_b, ori_inp_c, ori_inp_d, ori_inp_e, pred)

                    tt_inp_guest = inp_guest.clone()
                    tt_max_acc = S_acc[choice]

                    for kk, (_, inp_b, inp_c, inp_d, inp_e, _) in enumerate(fuzz_dataloader):
                        if tt_max_acc >= 95:
                            img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                            break
                        if kk >= 20:
                            break
                        inp_b = Variable(inp_b).to(device)
                        inp_b.requires_grad_(True)
                        inp_c = Variable(inp_c).to(device)
                        inp_c.requires_grad_(True)
                        inp_d = Variable(inp_d).to(device)
                        inp_d.requires_grad_(True)
                        inp_e = Variable(inp_e).to(device)
                        inp_e.requires_grad_(True)

                        outputs = torch.nn.Softmax(dim=1)(model(tt_inp_guest, inp_b, inp_c, inp_d, inp_e))
                        category = np.argmax(outputs.cpu().data.numpy())
                        low_saliency_socre = grad_var_multi_5(model, tt_inp_guest, device, inp_b, inp_c, inp_d, inp_e, outputs, fixedcategory=category)

                        low_noised_data = tt_inp_guest
                        for noise_index in range(4):
                            noise_mask = torch.zeros(tt_inp_guest.size())
                            if noise_index % 4 == 0:
                                noise_mask[0,0,:14,:7] = 1
                            elif noise_index % 4 == 1:
                                noise_mask[0,0,14:,:7] = 1
                            elif noise_index % 4 == 2:
                                noise_mask[0,0,:14,7:] = 1
                            elif noise_index % 4 == 3:
                                noise_mask[0,0,14:,7:] = 1
                            noised_inp_guest = tt_inp_guest + (torch.randn(tt_inp_guest.size()) * noise_mask).to(device) * 0.05
                            noised_inp_guest.data.clamp_(0, 1)
                            outputs = torch.nn.Softmax(dim=1)(model(noised_inp_guest, inp_b, inp_c, inp_d, inp_e))
                            temp_score = grad_var_multi_5(model, noised_inp_guest, device, inp_b, inp_c, inp_d, inp_e, outputs, fixedcategory=category)

                            if temp_score < low_saliency_socre:
                                low_saliency_socre = temp_score
                                low_noised_data = noised_inp_guest
                            del outputs
                        noised_inp_guest = low_noised_data
                        saliency_a_map = smooth_grad_guest(low_noised_data, inp_b, inp_c, inp_d, inp_e, category)
                        saliency_b_map = smooth_grad_b(low_noised_data, inp_b, inp_c, inp_d, inp_e, category)
                        saliency_c_map = smooth_grad_c(low_noised_data, inp_b, inp_c, inp_d, inp_e, category)
                        saliency_d_map = smooth_grad_d(low_noised_data, inp_b, inp_c, inp_d, inp_e, category)
                        saliency_e_map = smooth_grad_e(low_noised_data, inp_b, inp_c, inp_d, inp_e, category)
                        low_saliency_socre = (np.sum(saliency_b_map) + np.sum(saliency_c_map) + np.sum(saliency_d_map) + np.sum(saliency_e_map)) / (np.sum(saliency_a_map) + 1e-10)

                        if inp_b.grad is not None:
                            inp_b.grad.data.zero_()
                        if inp_c.grad is not None:
                            inp_c.grad.data.zero_()
                        if inp_d.grad is not None:
                            inp_d.grad.data.zero_()
                        if inp_e.grad is not None:
                            inp_e.grad.data.zero_()
    
                        if category == pred:
                            new_mask = iccv17_Multi_5(model, noised_inp_guest, device, inp_b, inp_c, inp_d, inp_e, category)
                            mask = 1 + (1 - new_mask) * 0.2

                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp.data.clamp_(0, 1)
                            new_acc = advtest_multi_5(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_a_map = smooth_grad_guest(tt_inp, inp_b, inp_c, inp_d, inp_e, category)
                            saliency_b_map = smooth_grad_b(tt_inp, inp_b, inp_c, inp_d, inp_e, category)
                            saliency_c_map = smooth_grad_c(tt_inp, inp_b, inp_c, inp_d, inp_e, category)
                            saliency_d_map = smooth_grad_d(tt_inp, inp_b, inp_c, inp_d, inp_e, category)
                            saliency_e_map = smooth_grad_e(tt_inp, inp_b, inp_c, inp_d, inp_e, category)
                            low_saliency_socre = (np.sum(saliency_b_map) + np.sum(saliency_c_map) + np.sum(saliency_d_map) + np.sum(saliency_e_map)) / (np.sum(saliency_a_map) + 1e-10)

                            if temp_score <= low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(0, 1)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())

                                orig_mask = iccv17_Multi_5(model, tt_inp_guest, device, ori_inp_b, ori_inp_c, ori_inp_d, ori_inp_e, pred)

                        else:
                            new_mask = iccv17_Multi_5(model, noised_inp_guest, device, inp_b, inp_c, inp_d, inp_e, category)
                            mask = (1.0 - orig_mask.mul(1.0 - new_mask)) + (1 - orig_mask) * 0.3
                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp.data.clamp_(0, 1)
                            new_acc = advtest_multi_5(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_a_map = smooth_grad_guest(tt_inp, inp_b, inp_c, inp_d, inp_e, category)
                            saliency_b_map = smooth_grad_b(tt_inp, inp_b, inp_c, inp_d, inp_e, category)
                            saliency_c_map = smooth_grad_c(tt_inp, inp_b, inp_c, inp_d, inp_e, category)
                            saliency_d_map = smooth_grad_d(tt_inp, inp_b, inp_c, inp_d, inp_e, category)
                            saliency_e_map = smooth_grad_e(tt_inp, inp_b, inp_c, inp_d, inp_e, category)
                            low_saliency_socre = (np.sum(saliency_b_map) + np.sum(saliency_c_map) + np.sum(saliency_d_map) + np.sum(saliency_e_map)) / (np.sum(saliency_a_map) + 1e-10)

                            if temp_score <= low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(0, 1)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())

                                orig_mask = iccv17_Multi_5(model, tt_inp_guest, device, ori_inp_b, ori_inp_c, ori_inp_d, ori_inp_e, pred)

                    if tt_max_acc == S_acc[choice]:
                        S_times[choice] += 1
                    print("Iter ", iter, ": acc ", tt_max_acc)

                    if tt_max_acc >= 95:
                        A.append(img_numpy)
                        print('found: ', len(A))
                        print('-----------------')
                        print(S[choice])
                        print('-----------------')
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_b[choice]
                        del S_c[choice]
                        del S_d[choice]
                        del S_e[choice]
                        del Q[choice]
                    elif tt_max_acc > S_acc[choice]:
                        Q[choice] = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                        S_acc[choice] = tt_max_acc
                        S_times[choice] = 0

                        S_index = np.argsort(-np.array(S_acc))
                        S = list(np.array(S)[S_index])
                        S_label = list(np.array(S_label)[S_index])
                        S_times = list(np.array(S_times)[S_index])
                        S_acc = list(np.array(S_acc)[S_index])
                        S_b = list(np.array(S_b)[S_index])
                        S_c = list(np.array(S_c)[S_index])
                        S_d = list(np.array(S_d)[S_index])
                        S_e = list(np.array(S_e)[S_index])
                        Q = list(np.array(Q)[S_index])

                    elif S_times[choice] >= 5:
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_b[choice]
                        del S_c[choice]
                        del S_d[choice]
                        del S_e[choice]
                        del Q[choice]

                    if len(S) <= 0:
                        break

    elif args.dataset == 'nuswide':

        print('----iterative method----')
        print('Model : %s \n' %(model_name))

        model = getattr(models, model_name)().eval()
        model.load(args.load_model_path)

        device = 'cuda:' + str(args.device) if gpu else 'cpu'

        model.to(device)

        transformer = T.Compose([
            T.ToTensor(),
        ])
        data_dir = "/data/NUS_WIDE/"
        sel_lbls = ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']
        load_three_party = False
        train_data_list, test_data_list = load_prepared_parties_data(data_dir, sel_lbls, load_three_party)
        
        guest_data_list = train_data_list[0]
        host_data_list = train_data_list[1]
        label_list = train_data_list[2]
        
        for dataindex in range(args.guest_data_num):
            orig_guest = guest_data_list[dataindex].astype(np.float32)

            orig_guest = orig_guest.reshape(1, orig_guest.shape[0])
            orig_guest = transformer(orig_guest)
            
            orig_host = host_data_list[dataindex].astype(np.float32)
            orig_host = orig_host.reshape(1, orig_host.shape[0])
            orig_host = transformer(orig_host)

            inp_guest = Variable(orig_guest.to(device).float().unsqueeze(0), requires_grad=True)
            inp_host = Variable(orig_host.to(device).float().unsqueeze(0), requires_grad=True)

            host_data_for_test = host_data_list.copy()[args.host_data_num:]
            host_data_list = host_data_list[:args.host_data_num]
            out = model(inp_guest, inp_host)
            orig_pred = np.argmax(out.data.cpu().numpy())
            pred = np.argmax(out.data.cpu().numpy())

            criterion = nn.CrossEntropyLoss()

            eps = args.eps
            alpha1 = 10.0
            alpha2 = 1.0
            iter_num = 100
            rho = 10

            orig = orig_guest.clone().to(device).float().unsqueeze(0)

            if args.saliency == 'fuzz_with_saliency':
                train_data_mnist = NusWide(mode='val')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=1)

                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 1000
                for S_index in tqdm(range(S_size*0, S_size*1)):
                    S.append(S_index)
                    S_orighost.append(S_index)
                    S_times.append(0)
                    slabel = np.argmax(label_list[S_index]).astype(int)
                    S_label.append(slabel)
                    S_guest = guest_data_list[S_index].astype(np.float32)
                    S_guest = S_guest.reshape(1, S_guest.shape[0])
                    S_guest = transformer(S_guest)
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                    Q.append(copy.deepcopy(S_guest.cpu().data.numpy()))
                print(S_acc)
                S_index = np.argsort(-np.array(S_acc))
                S = list(np.array(S)[S_index])
                S_label = list(np.array(S_label)[S_index])
                S_times = list(np.array(S_times)[S_index])
                S_acc = list(np.array(S_acc)[S_index])
                S_orighost = list(np.array(S_orighost)[S_index])
                Q = list(np.array(Q)[S_index])
                smooth_grad_guest = SmoothGrad(model, device, types='guest')
                smooth_grad_host = SmoothGrad(model, device, types='host')
                store_adi = []
                for iter in tqdm(range(5000)):
                    if iter % 3 == 0:
                        choice = random.randint(1, len(S)-1)
                    else:
                        choice = 0
                    local_max = np.max(Q[choice])
                    if local_max > 0:
                        local_max *= 2.0
                    else:
                        local_max *= 0.5
                    local_min = np.min(Q[choice])
                    if local_min > 0:
                        local_min *= 0.5
                    else:
                        local_min *= 2.0
                    inp_guest = torch.from_numpy(Q[choice])
                    inp_guest = Variable(inp_guest.to(device).float(), requires_grad=True)
                    temp_host_data = host_data_list[S_orighost[choice]].astype(np.float32)
                    temp_host_data = temp_host_data.reshape(1, temp_host_data.shape[0])
                    temp_host_data = transformer(temp_host_data)
                    ori_inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=False)
                    pred = S_label[choice]
                    orig_mask = iccv17(model, inp_guest, device, ori_inp_host, pred)

                    tt_inp_guest = inp_guest.clone()
                    tt_max_acc = S_acc[choice]
                    print('orig: ', tt_max_acc)
                    for kk in range(0, 30):
                        inp_host = host_data_list[kk].astype(np.float32)
                        inp_host = inp_host.reshape(1, inp_host.shape[0])
                        inp_host = transformer(inp_host)
                        
                        inp_host = Variable(inp_host.to(device).float().unsqueeze(0), requires_grad=True)
                        outputs = torch.nn.Softmax(dim=1)(model(tt_inp_guest, inp_host))
                        category = np.argmax(outputs.cpu().data.numpy())
                        low_saliency_socre = grad_var(model, tt_inp_guest, device, inp_host, outputs, fixedcategory=category)

                        low_noised_data = tt_inp_guest
                        
                        for noise_index in range(5):
                            noised_inp_guest = tt_inp_guest + (torch.randn(tt_inp_guest.size())).to(device) * 0.05
                            noised_inp_guest.data.clamp_(local_min, local_max)
                            outputs = torch.nn.Softmax(dim=1)(model(noised_inp_guest, inp_host))
                            temp_score = grad_var(model, noised_inp_guest, device, inp_host, outputs, fixedcategory=category)
                            if temp_score < low_saliency_socre:
                                low_saliency_socre = temp_score
                                low_noised_data = noised_inp_guest
                            del outputs
                        noised_inp_guest = low_noised_data
                        saliency_guest_map = smooth_grad_guest(low_noised_data, inp_host, category)
                        saliency_host_map = smooth_grad_host(low_noised_data, inp_host, category)
                        low_saliency_socre = np.sum(saliency_host_map) / np.sum(saliency_guest_map)
                        if inp_host.grad is not None:
                            inp_host.grad.data.zero_()
                        if category == pred:
                            new_mask = iccv17(model, noised_inp_guest, device, inp_host, category)
                            mask = 1 + (1 - new_mask) * 0.2
                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp.data.clamp_(local_min, local_max)
                            new_acc = advtest(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_guest_map = smooth_grad_guest(tt_inp, inp_host, category)
                            saliency_host_map = smooth_grad_host(tt_inp, inp_host, category)
                            temp_score = np.sum(saliency_host_map) / np.sum(saliency_guest_map)
                            if temp_score < low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(local_min, local_max)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                                orig_mask = iccv17(model, tt_inp_guest, device, ori_inp_host, pred)

                        else:
                            new_mask = iccv17(model, noised_inp_guest, device, inp_host, category)

                            # Change the mask here.

                            mask = (1.0 - orig_mask.mul(1.0 - new_mask)) + (1 - orig_mask) * 0.3
                            tt_inp = noised_inp_guest.mul(mask)
                            
                            tt_inp.data.clamp_(local_min, local_max)
                            new_acc = advtest(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_guest_map = smooth_grad_guest(tt_inp, inp_host, category)
                            saliency_host_map = smooth_grad_host(tt_inp, inp_host, category)
                            temp_score = np.sum(saliency_host_map) / np.sum(saliency_guest_map)
                            if temp_score < low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(local_min, local_max)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                                
                                orig_mask = iccv17(model, tt_inp_guest, device, ori_inp_host, pred)
                    
                    if tt_max_acc == S_acc[choice]:
                        S_times[choice] += 1

                    print("Iter ", iter, ": acc ", tt_max_acc)
                    if tt_max_acc >= 95:
                        A.append(img_numpy)

                        print('found: ', len(A), 'left: ', len(S))
                        print('-----------------')
                        print(S[choice])
                        print('-----------------')
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_orighost[choice]
                        del Q[choice]
                    elif tt_max_acc > S_acc[choice]:
                        Q[choice] = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                        S_acc[choice] = tt_max_acc
                        S_times[choice] = 0
                        S_index = np.argsort(-np.array(S_acc))
                        S = list(np.array(S)[S_index])
                        S_label = list(np.array(S_label)[S_index])
                        S_times = list(np.array(S_times)[S_index])
                        S_acc = list(np.array(S_acc)[S_index])
                        S_orighost = list(np.array(S_orighost)[S_index])
                        Q = list(np.array(Q)[S_index])

                    elif S_times[choice] >= 5:
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_orighost[choice]
                        del Q[choice]

                    if len(S) <= 50 or len(A) >= 200:
                        break

            elif args.saliency == 'test_orig_rate':
                train_data_mnist = NusWide(mode='val')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=1)

                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 5000
                for S_index in tqdm(range(S_size*0, S_size*1)):
                    S.append(S_index)
                    S_orighost.append(S_index)
                    slabel = np.argmax(label_list[S_index]).astype(int)
                    S_label.append(slabel)
                    S_guest = guest_data_list[S_index].astype(np.float32)
                    S_guest = S_guest.reshape(1, S_guest.shape[0])
                    S_guest = transformer(S_guest)
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                
                ADI_num = 0.0
                total_num = len(S_acc)

                for temp_acc in S_acc:
                    if temp_acc >= 95:
                        ADI_num += 1
                print('=' * 10)
                result = ADI_num / float(total_num)
                print(result)

            elif args.saliency == 'whitebox':
                S_size = 1000
                train_data_mnist = NusWide(mode='val')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=1)

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 500.0

                for S_index in tqdm(range(3*S_size, 4*S_size)):

                    slabel = np.argmax(label_list[S_index]).astype(int)
                    S_guest = guest_data_list[S_index].astype(np.float32)
                    S_guest = S_guest.reshape(1, S_guest.shape[0])
                    S_guest = transformer(S_guest)
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    temp_host_data = host_data_list[S_index].astype(np.float32)
                    temp_host_data = temp_host_data.reshape(1, temp_host_data.shape[0])
                    temp_host_data = transformer(temp_host_data)
                    ori_inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=False)
                    
                    out = model(S_guest, ori_inp_host)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    
                    for kk in range(20):
                        temp_host_data = host_data_list[S_index].astype(np.float32)
                        temp_host_data = temp_host_data.reshape(1, temp_host_data.shape[0])
                        temp_host_data = transformer(temp_host_data)
                        inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=True)

                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(torch.var(out), inp_host, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_x2[0]).mean()
                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)
                            
                            grad = grady_x1_loss1 * 0.7 + grady_x1_loss2[0] * 0.3
                            perturbation = -lr * grad
                            inp_guest.data += perturbation

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_host

                    temp_acc = advtest(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            elif args.saliency == 'whitebox_bounded':
                S_size = 1000
                print(S_size)
                train_data_mnist = NusWide(mode='val')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=1)

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 500
                store_orig = []
                store_adi = []
                for S_index in tqdm(range(0*S_size, 1*S_size)):

                    slabel = np.argmax(label_list[S_index]).astype(int)
                    S_guest = guest_data_list[S_index].astype(np.float32)
                    S_guest = S_guest.reshape(1, S_guest.shape[0])
                    S_guest = transformer(S_guest)
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    store_orig.append(S_guest.cpu().data.numpy())

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    temp_host_data = host_data_list[S_index].astype(np.float32)
                    temp_host_data = temp_host_data.reshape(1, temp_host_data.shape[0])
                    temp_host_data = transformer(temp_host_data)
                    ori_inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=False)
                    
                    out = model(S_guest, ori_inp_host)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    local_min = np.min(inp_guest.cpu().data.numpy())
                    local_max = np.max(inp_guest.cpu().data.numpy())
                    for kk in range(20):
                        temp_host_data = host_data_list[S_index].astype(np.float32)
                        temp_host_data = temp_host_data.reshape(1, temp_host_data.shape[0])
                        temp_host_data = transformer(temp_host_data)
                        inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=True)
                    
                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(torch.var(out), inp_host, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_x2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)
                            
                            grad = grady_x1_loss1 * 0.7 + grady_x1_loss2[0] * 0.3

                            perturbation = -lr * grad

                            inp_guest.data += perturbation
                            inp_guest.data = torch.clamp(inp_guest.data, min = local_min, max = local_max)
                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()

                            del loss1, grady_x1_loss1, out, grad
                            torch.cuda.empty_cache()

                        del inp_host

                    temp_acc = advtest(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                    total_num += 1.0

                    if temp_acc >= 90:
                        store_adi.append(inp_guest.cpu().data.numpy())
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

    elif args.dataset == 'vehicle':
        print('----iterative method----')
        print('Model : %s \n' %(model_name))
        # load model
        model = getattr(models, model_name)().eval()
        model.load(args.vehicle_model_path)
        device = 'cuda:' + str(args.device) if gpu else 'cpu'
        model.to(device)

        transformer = T.Compose([T.ToTensor()])
        
        guest_data_list = pd.read_csv(args.vehicle_guest_list).to_numpy()
        host_data_list = pd.read_csv(args.vehicle_host_list).to_numpy()
        criterion = nn.CrossEntropyLoss()
        eps = args.eps
        alpha = 0.01
        iter_num = 100

        for i in range(args.guest_data_num):
            if args.guest_index != None:
                i = args.guest_index
            orig_guest = transformer(guest_data_list[i][2:].astype(float).reshape((1, -1)))
            data_orig_host = host_data_list[i].copy()
            orig_host = transformer(host_data_list[i][1:].astype(float).reshape((1, -1)))
    
            inp_guest = Variable(orig_guest.to(device).float().unsqueeze(0), requires_grad=True)
            inp_host = Variable(orig_host.to(device).float().unsqueeze(0), requires_grad=True)
            new_host_data_list = copy.deepcopy(host_data_list)
            np.random.shuffle(new_host_data_list)
            host_data_list_for_test = new_host_data_list[args.host_data_num:]
            new_host_data_list = np.r_[[data_orig_host], new_host_data_list][:args.host_data_num]

            out = model(inp_guest, inp_host)
            orig_pred = np.argmax(out.data.cpu().numpy())
            pred = np.argmax(out.data.cpu().numpy())
            print('prediction before attack: %d' %pred)
            print('target label: %d' %pred)

            orig = orig_guest.clone().to(device).float().unsqueeze(0)

            if args.saliency == 'fuzz_with_saliency':
                train_data_mnist = vehicle(mode='val', dataset='vehicle')
                train_dataloader = DataLoader(train_data_mnist, 32, shuffle=True, num_workers=4)

                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 1000
                for S_index in tqdm(range(S_size*0, S_size*1)):
                    S.append(S_index)
                    S_orighost.append(S_index)
                    S_times.append(0)
                    slabel = guest_data_list[S_index][1].astype(int)
                    S_label.append(slabel)
                    
                    S_guest = transformer(guest_data_list[S_index][2:].astype(float).reshape((1, -1)))
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                    Q.append(copy.deepcopy(S_guest.cpu().data.numpy()))
                print(S_acc)
                S_index = np.argsort(-np.array(S_acc))
                S = list(np.array(S)[S_index])
                S_label = list(np.array(S_label)[S_index])
                S_times = list(np.array(S_times)[S_index])
                S_acc = list(np.array(S_acc)[S_index])
                S_orighost = list(np.array(S_orighost)[S_index])
                Q = list(np.array(Q)[S_index])
                smooth_grad_guest = SmoothGrad(model, device, types='guest')
                smooth_grad_host = SmoothGrad(model, device, types='host')
                store_adi = []
                for iter in tqdm(range(5000)):
                    if iter % 3 == 0:
                        choice = random.randint(1, len(S)-1)
                    else:
                        choice = 0
                    local_max = np.max(Q[choice])

                    local_min = np.min(Q[choice])
                    inp_guest = torch.from_numpy(Q[choice])
                    inp_guest = Variable(inp_guest.to(device).float(), requires_grad=True)
                    temp_host_data = host_data_list[S_orighost[choice]][1:].astype(np.float32)
                    temp_host_data = temp_host_data.reshape(1, temp_host_data.shape[0])
                    temp_host_data = transformer(temp_host_data)
                    ori_inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=False)
                    pred = S_label[choice]
                    orig_mask = iccv17(model, inp_guest, device, ori_inp_host, pred)

                    tt_inp_guest = inp_guest.clone()
                    tt_max_acc = S_acc[choice]
                    print('orig: ', tt_max_acc)
                    for kk in range(0, 20):
                        inp_host = host_data_list[kk][1:].astype(np.float32)
                        inp_host = inp_host.reshape(1, inp_host.shape[0])
                        inp_host = transformer(inp_host)
                        
                        inp_host = Variable(inp_host.to(device).float().unsqueeze(0), requires_grad=True)
                        outputs = torch.nn.Softmax(dim=1)(model(tt_inp_guest, inp_host))
                        category = np.argmax(outputs.cpu().data.numpy())

                        low_saliency_socre = grad_var(model, tt_inp_guest, device, inp_host, outputs, fixedcategory=category)
                        low_noised_data = tt_inp_guest
                        # print(low_saliency_socre)
                        for noise_index in range(5):
                            noised_inp_guest = tt_inp_guest + (torch.randn(tt_inp_guest.size())).to(device) * 0.05
                            noised_inp_guest.data.clamp_(local_min, local_max)
                            outputs = torch.nn.Softmax(dim=1)(model(noised_inp_guest, inp_host))
                            temp_score = grad_var(model, noised_inp_guest, device, inp_host, outputs, fixedcategory=category)
                            if temp_score < low_saliency_socre:
                                low_saliency_socre = temp_score
                                low_noised_data = noised_inp_guest
                            del outputs
                        noised_inp_guest = low_noised_data
                        saliency_guest_map = smooth_grad_guest(low_noised_data, inp_host, category)
                        saliency_host_map = smooth_grad_host(low_noised_data, inp_host, category)
                        low_saliency_socre = np.sum(saliency_host_map) / (np.sum(saliency_guest_map) + 1e-10)

                        if inp_host.grad is not None:
                            inp_host.grad.data.zero_()
                        if category == pred:
                            new_mask = iccv17(model, noised_inp_guest, device, inp_host, category)
                            mask = 1 + (1 - new_mask) * 0.2
                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp.data.clamp_(local_min, local_max)
                            new_acc = advtest(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_guest_map = smooth_grad_guest(tt_inp, inp_host, category)
                            saliency_host_map = smooth_grad_host(tt_inp, inp_host, category)
                            temp_score = np.sum(saliency_host_map) / (np.sum(saliency_guest_map) + 1e-10)

                            if temp_score <= low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(local_min, local_max)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())

                                orig_mask = iccv17(model, tt_inp_guest, device, ori_inp_host, pred)

                        else:
                            new_mask = iccv17(model, noised_inp_guest, device, inp_host, category)

                            mask = (1.0 - orig_mask.mul(1.0 - new_mask)) + (1 - orig_mask) * 0.3
                            tt_inp = noised_inp_guest.mul(mask)

                            tt_inp.data.clamp_(local_min, local_max)
                            new_acc = advtest(model, tt_inp, train_dataloader, pred, pred, args.dataset)
                            saliency_guest_map = smooth_grad_guest(tt_inp, inp_host, category)
                            saliency_host_map = smooth_grad_host(tt_inp, inp_host, category)
                            temp_score = np.sum(saliency_host_map) / (np.sum(saliency_guest_map) + 1e-10)

                            if temp_score <= low_saliency_socre and new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest.data.clamp_(local_min, local_max)
                                low_saliency_socre = temp_score
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                            
                                orig_mask = iccv17(model, tt_inp_guest, device, ori_inp_host, pred)
                        if tt_max_acc >= 95:
                            break
                    
                    if tt_max_acc == S_acc[choice]:
                        S_times[choice] += 1

                    print("Iter ", iter, ": acc ", tt_max_acc)
                    if tt_max_acc >= 95:
                        A.append(img_numpy)
                        print(img_numpy)
                        print('found: ', len(A), 'left: ', len(S))

                        print('-----------------')
                        print(S[choice])
                        print('-----------------')
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_orighost[choice]
                        del Q[choice]
                    elif tt_max_acc > S_acc[choice]:
                        Q[choice] = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                        S_acc[choice] = tt_max_acc
                        S_times[choice] = 0

                        S_index = np.argsort(-np.array(S_acc))
                        S = list(np.array(S)[S_index])
                        S_label = list(np.array(S_label)[S_index])
                        S_times = list(np.array(S_times)[S_index])
                        S_acc = list(np.array(S_acc)[S_index])
                        S_orighost = list(np.array(S_orighost)[S_index])
                        Q = list(np.array(Q)[S_index])

                    elif S_times[choice] >= 5:
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_orighost[choice]
                        del Q[choice]

            
            elif args.saliency == 'test_orig_rate':
                train_data_mnist = vehicle(mode='val', dataset='vehicle')
                train_dataloader = DataLoader(train_data_mnist, 32, shuffle=True, num_workers=4)

                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 700
                for S_index in tqdm(range(S_size*0, S_size*1)):
                    S.append(S_index)

                    slabel = guest_data_list[S_index][1].astype(int)
                    S_label.append(slabel)
                    
                    S_guest = transformer(guest_data_list[S_index][2:].astype(float).reshape((1, -1)))
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                
                ADI_num = 0.0
                total_num = len(S_acc)

                for temp_acc in S_acc:
                    if temp_acc >= 95:
                        ADI_num += 1
                print('=' * 10)
                result = ADI_num / float(total_num)
                print(result)
            
            elif args.saliency == 'whitebox':
                S_size = 1000
                train_data_mnist = vehicle(mode='val', dataset='vehicle')
                train_dataloader = DataLoader(train_data_mnist, 32, shuffle=True, num_workers=4)

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 0.1

                for S_index in tqdm(range(0*S_size, 1*S_size)):

                    slabel = guest_data_list[S_index][1].astype(int)
                    S_guest = transformer(guest_data_list[S_index][2:].astype(float).reshape((1, -1)))
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    orig_host = transformer(host_data_list[S_index][1:].astype(float).reshape((1, -1)))

                    ori_inp_host = Variable(orig_host.to(device).float().unsqueeze(0), requires_grad=False)
                    
                    out = model(S_guest, ori_inp_host)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    grad = 0.0
                    for kk in range(20):
                        temp_inp_host = transformer(host_data_list[kk][1:].astype(float).reshape((1, -1)))

                        inp_host = Variable(temp_inp_host.to(device).float().unsqueeze(0), requires_grad=True)

                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(torch.var(out), inp_host, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_x2[0]).mean()

                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)

                            
                            grad = grady_x1_loss1 * 0.99 + grady_x1_loss2[0] * 0.01 + grad * 0.8

                            perturbation = -lr * grad

                          
                            inp_guest.data += perturbation

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out
                            torch.cuda.empty_cache()

                        del inp_host

                        temp_acc = advtest(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                        if temp_acc >= 95:
                            break
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            elif args.saliency == 'whitebox_bounded':
                S_size = 1000
                train_data_mnist = vehicle(mode='val', dataset='vehicle')
                train_dataloader = DataLoader(train_data_mnist, 32, shuffle=True, num_workers=4)

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 30
                lr = 0.1

                for S_index in tqdm(range(1*S_size, 2*S_size)):

                    slabel = guest_data_list[S_index][1].astype(int)
                    S_guest = transformer(guest_data_list[S_index][2:].astype(float).reshape((1, -1)))
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    orig_host = transformer(host_data_list[S_index][1:].astype(float).reshape((1, -1)))

                    ori_inp_host = Variable(orig_host.to(device).float().unsqueeze(0), requires_grad=False)
                    
                    out = model(S_guest, ori_inp_host)
                    pred = np.argmax(out.data.cpu().numpy())

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    orig = S_guest.clone()
                    grad = 0.0
                    for kk in range(20):
                        temp_inp_host = transformer(host_data_list[kk][1:].astype(float).reshape((1, -1)))

                        inp_host = Variable(temp_inp_host.to(device).float().unsqueeze(0), requires_grad=True)

                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            loss1 = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()
                            # print(grady_x1_loss1)

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(torch.var(out), inp_host, create_graph=True, retain_graph=True)

                            loss2 = torch.abs(gradvar_x2[0]).mean()
                            # loss2.backward()
                            # optimizer.step()
                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)
                            
                            grad = grady_x1_loss1 * 0.99 + grady_x1_loss2[0] * 0.01 + grad * 0.8
                            # grad = grady_x1_loss1
                            perturbation = -lr * grad
                            perturbation += inp_guest.data - orig.data
                            divider = torch.sum(torch.abs(perturbation))
                            perturbation /= divider
                            perturbation *= 2.0
                            
                            inp_guest.data = perturbation + orig.data
                            inp_guest.data = torch.clamp(inp_guest.data, min = -1.0, max = 1.0)
                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
    
                            del loss1, grady_x1_loss1, out
                            torch.cuda.empty_cache()

                        del inp_host

                        temp_acc = advtest(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                        if temp_acc >= 99:
                            break
                    total_num += 1.0

                    if temp_acc >= 99:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            del inp_guest
            del orig
            torch.cuda.empty_cache()

    elif args.dataset == 'credit':
        print('----iterative method----')
        print('Model : %s \n' %(model_name))
        # load model
        model = getattr(models, model_name)().eval()
        model.load(args.credit_model_path)
        device = 'cuda:' + str(args.device) if gpu else 'cpu'
        model.to(device)

        transformer = T.Compose([T.ToTensor()])
        
        guest_data_list = pd.read_csv(args.credit_guest_list).to_numpy()
        host_data_list = pd.read_csv(args.credit_host_list).to_numpy()
        # print(guest_data_list.shape)        
        criterion = nn.BCEWithLogitsLoss()
        # print(host_data_list.shape)
        eps = args.eps
        alpha = 0.01
        iter_num = 100

        for i in range(args.guest_data_num):
            if args.guest_index != None:
                i = args.guest_index
            orig_guest = transformer(guest_data_list[i][2:].astype(float).reshape((1, -1)))
            data_orig_host = host_data_list[i].copy()
            orig_host = transformer(host_data_list[i][1:].astype(float).reshape((1, -1)))
    
            inp_guest = Variable(orig_guest.to(device).float().unsqueeze(0), requires_grad=True)
            inp_host = Variable(orig_host.to(device).float().unsqueeze(0), requires_grad=True)
            new_host_data_list = copy.deepcopy(host_data_list)
            np.random.shuffle(new_host_data_list)
            host_data_list_for_test = new_host_data_list[args.host_data_num:]
            new_host_data_list = np.r_[[data_orig_host], new_host_data_list][:args.host_data_num]

            out = torch.nn.Sigmoid()(model(inp_guest, inp_host))
            orig_pred = torch.round(out).data.cpu().numpy()
            pred = torch.round(out).data.cpu().numpy()
            print('prediction before attack: %d' %pred)

            print('target label: %d' %pred)

            orig = orig_guest.clone().to(device).float().unsqueeze(0)

            if args.saliency == 'fuzz_with_saliency':
                train_data_mnist = credit(mode='val', dataset='credit')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)

                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 1000
                for S_index in tqdm(range(S_size*0, S_size*1)):
                    S.append(S_index)
                    S_orighost.append(S_index)
                    S_times.append(0)
                    slabel = guest_data_list[S_index][1].astype(int)
                    S_label.append(slabel)
                    
                    S_guest = transformer(guest_data_list[S_index][2:].astype(float).reshape((1, -1)))
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                    Q.append(copy.deepcopy(S_guest.cpu().data.numpy()))
                print(S_acc)
                S_index = np.argsort(-np.array(S_acc))
                S = list(np.array(S)[S_index])
                S_label = list(np.array(S_label)[S_index])
                S_times = list(np.array(S_times)[S_index])
                S_acc = list(np.array(S_acc)[S_index])
                S_orighost = list(np.array(S_orighost)[S_index])
                Q = list(np.array(Q)[S_index])
                smooth_grad_guest = SmoothGrad(model, device, types='guest')
                smooth_grad_host = SmoothGrad(model, device, types='host')
                store_adi = []
                for iter in tqdm(range(5000)):
                    if iter % 3 == 0:
                        choice = random.randint(1, len(S)-1)
                    else:
                        choice = 0
                    local_max = torch.Tensor([[7.20985, 12.402963, 13.133596, 23.318199, 13.186686, 14.587432, 15.495281, 52.399215, 72.842986, 50.595281, 39.332179, 27.603626, 29.445098]]).to(device)
                    local_min = torch.Tensor([[-1.486041, -2.944312, -1.671375, -2.945672, -3.315048, -2.000874, -6.355247, -0.341942, -0.256990, -0.296801, -0.308063, -0.314136, -0.293382]]).to(device)
                    
                    inp_guest = torch.from_numpy(Q[choice])
                    inp_guest = Variable(inp_guest.to(device).float(), requires_grad=True)
                    temp_host_data = host_data_list[S_orighost[choice]][1:].astype(np.float32)
                    temp_host_data = temp_host_data.reshape(1, temp_host_data.shape[0])
                    temp_host_data = transformer(temp_host_data)
                    ori_inp_host = Variable(temp_host_data.to(device).float().unsqueeze(0), requires_grad=False)
                    pred = S_label[choice]
                    orig_mask = iccv17(model, inp_guest, device, ori_inp_host, pred)

                    tt_inp_guest = inp_guest.clone()
                    tt_max_acc = S_acc[choice]
                    print('orig: ', tt_max_acc)
                    img_numpy = Q[choice]
                    for kk in range(0, 15):
                        inp_host = host_data_list[kk][1:].astype(np.float32)
                        inp_host = inp_host.reshape(1, inp_host.shape[0])
                        inp_host = transformer(inp_host)
                        
                        inp_host = Variable(inp_host.to(device).float().unsqueeze(0), requires_grad=True)
                        outputs = torch.round(model(tt_inp_guest, inp_host))
                        category = outputs.cpu().data

                        low_saliency_socre = grad_var(model, tt_inp_guest, device, inp_host, outputs, fixedcategory=category)

                        low_noised_data = tt_inp_guest
                        for noise_index in range(5):
                            noised_inp_guest = tt_inp_guest + (torch.randn(tt_inp_guest.size())).to(device) * 0.05
                            noised_inp_guest = torch.max(torch.min(noised_inp_guest, local_max), local_min)
                            outputs = torch.nn.Sigmoid()(model(noised_inp_guest, inp_host))
                            low_noised_data = noised_inp_guest
                            del outputs
                        if inp_host.grad is not None:
                            inp_host.grad.data.zero_()
                        if category == pred:
                            new_mask = iccv17(model, noised_inp_guest, device, inp_host, category)
                            mask = 1 + (1 - new_mask) * 0.2
                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp = torch.max(torch.min(tt_inp, local_max), local_min)
                            new_acc = advtest(model, tt_inp, train_dataloader, pred, orig_pred, args.dataset)
                            outputs = torch.nn.Sigmoid()(model(tt_inp, inp_host))

                            if new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest = torch.max(torch.min(tt_inp_guest, local_max), local_min)
                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())

                                
                                orig_mask = iccv17(model, tt_inp_guest, device, ori_inp_host, pred)

                        else:
                            new_mask = iccv17(model, noised_inp_guest, device, inp_host, category)

                            mask = (1.0 - orig_mask.mul(1.0 - new_mask)) + (1 - orig_mask) * 0.3
                            tt_inp = noised_inp_guest.mul(mask)
                            tt_inp = torch.max(torch.min(tt_inp, local_max), local_min)
                            new_acc = advtest(model, tt_inp, train_dataloader, pred, orig_pred, args.dataset)
                            outputs = torch.nn.Sigmoid()(model(tt_inp, inp_host))
                            if new_acc > tt_max_acc:
                                tt_inp_guest = noised_inp_guest.mul(mask)
                                tt_inp_guest = torch.max(torch.min(tt_inp_guest, local_max), local_min)

                                tt_max_acc = new_acc
                                img_numpy = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                            
                                orig_mask = iccv17(model, tt_inp_guest, device, ori_inp_host, pred)
                    
                    if tt_max_acc == S_acc[choice]:
                        S_times[choice] += 1

                    print("Iter ", iter, ": acc ", tt_max_acc)
                    if tt_max_acc >= 95:
                        A.append(img_numpy)
                        print('found: ', len(A), 'left: ', len(S))

                        print('-----------------')
                        print(S[choice])
                        print('-----------------')
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_orighost[choice]
                        del Q[choice]
                    elif tt_max_acc > S_acc[choice]:
                        Q[choice] = copy.deepcopy(tt_inp_guest.cpu().data.numpy())
                        S_acc[choice] = tt_max_acc
                        S_times[choice] = 0

                        S_index = np.argsort(-np.array(S_acc))
                        S = list(np.array(S)[S_index])
                        S_label = list(np.array(S_label)[S_index])
                        S_times = list(np.array(S_times)[S_index])
                        S_acc = list(np.array(S_acc)[S_index])
                        S_orighost = list(np.array(S_orighost)[S_index])
                        Q = list(np.array(Q)[S_index])

                    elif S_times[choice] >= 5:
                        del S[choice]
                        del S_times[choice]
                        del S_acc[choice]
                        del S_label[choice]
                        del S_orighost[choice]
                        del Q[choice]

                    if len(S) <= 50 or len(A) >= 200:
                        break

            elif args.saliency == 'test_orig_rate':
                train_data_mnist = credit(mode='val', dataset='credit')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)

                S = []
                S_label = []
                S_acc = []
                S_times = []
                S_orighost = []
                Q = []
                A = []
                S_size = 5000
                for S_index in tqdm(range(S_size*1, S_size*2)):
                    S.append(S_index)
                    S_orighost.append(S_index)
                    S_times.append(0)
                    slabel = guest_data_list[S_index][1].astype(int)
                    S_label.append(slabel)
                    
                    S_guest = transformer(guest_data_list[S_index][2:].astype(float).reshape((1, -1)))
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    S_acc.append(tempacc)
                ADI_num = 0.0
                total_num = len(S_acc)
                for tempacc in S_acc:
                    if tempacc >= 99:
                        ADI_num += 1.0
                results = ADI_num / float(total_num)
                print('=' * 10)
                print(results)

            elif args.saliency == 'whitebox':
                S_size = 1000
                train_data_mnist = credit(mode='val', dataset='credit')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 20
                lr = 0.2
                
                for S_index in tqdm(range(0*S_size, 1*S_size)):

                    slabel = guest_data_list[S_index][1].astype(float)
                    S_guest = transformer(guest_data_list[S_index][2:].astype(float).reshape((1, -1)))
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)

                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    orig_host = transformer(host_data_list[S_index][1:].astype(float).reshape((1, -1)))

                    ori_inp_host = Variable(orig_host.to(device).float().unsqueeze(0), requires_grad=False)
                    
                    out = model(S_guest, ori_inp_host)
                    pred = slabel

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    grad = 0.0
                    pred = Variable(torch.Tensor([float(pred)]).reshape(1, -1).to(device).float())
                    for kk in range(30):
                        temp_inp_host = transformer(host_data_list[kk][1:].astype(float).reshape((1, -1)))

                        inp_host = Variable(temp_inp_host.to(device).float().unsqueeze(0), requires_grad=True)

                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            out = nn.Sigmoid()(out)
                            loss1 = criterion(out, pred)
                            
                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()
                            # print(grady_x1_loss1)

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(out, inp_host, create_graph=True, retain_graph=True)
                            loss2 = torch.abs(gradvar_x2[0]).mean()
                            # loss2.backward()
                            # optimizer.step()
                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)

                            # inp_guest.grad.data.zero_()
                            # inp_host.grad.data.zero_()
                            # model.zero_grad()
                            
                            grad = grady_x1_loss1 * 0.99 + grady_x1_loss2[0] * 0.01 + grad * 0.8
                            # grad = grady_x1_loss1
                            perturbation = -lr * grad

                            # print(perturbation)
                            # perturbation = torch.clamp((inp_guest.data + perturbation) - orig, min=-eps * 0.5, max=eps * 3.0)
                            # # perturbation = inp_guest.data + perturbation - orig

                            # inp_guest.data = torch.clamp((orig + perturbation), min=-0.41, max=3)
                            # print(perturbation)
                            inp_guest.data += perturbation

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
    
                            # vis.plot('loss1', loss1.data.cpu())
                            # vis.plot('loss2', loss2.data.cpu())
                            # del loss1, loss2, grady_x1_loss2, grad, out, gradvar_x2, grady_x1_loss1
                            del loss1, grady_x1_loss1, out
                            torch.cuda.empty_cache()

                        del inp_host

                        temp_acc = advtest(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                        if temp_acc >= 95:
                            break
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            elif args.saliency == 'whitebox_bounded':
                S_size = 1000
                train_data_mnist = credit(mode='val', dataset='credit')
                train_dataloader = DataLoader(train_data_mnist, 256, shuffle=True, num_workers=4)

                ADI_num = 0.0
                total_num = 0.0
                iter_num = 20
                lr = 0.2
                local_max = torch.Tensor([[7.20985, 12.402963, 13.133596, 23.318199, 13.186686, 14.587432, 15.495281, 52.399215, 72.842986, 50.595281, 39.332179, 27.603626, 29.445098]]).to(device)
                local_min = torch.Tensor([[-1.486041, -2.944312, -1.671375, -2.945672, -3.315048, -2.000874, -6.355247, -0.341942, -0.256990, -0.296801, -0.308063, -0.314136, -0.293382]]).to(device)
                store_orig = []
                store_adi = []
                for S_index in tqdm(range(0*S_size, 1*S_size)):

                    slabel = guest_data_list[S_index][1].astype(float)
                    S_guest = transformer(guest_data_list[S_index][2:].astype(float).reshape((1, -1)))
                    S_guest = Variable(S_guest.to(device).float().unsqueeze(0), requires_grad=False)
                    tempacc = advtest(model, S_guest, train_dataloader, slabel, slabel, args.dataset)
                    orig_host = transformer(host_data_list[S_index][1:].astype(float).reshape((1, -1)))

                    ori_inp_host = Variable(orig_host.to(device).float().unsqueeze(0), requires_grad=False)
                    
                    out = model(S_guest, ori_inp_host)
                    pred = slabel

                    inp_guest = S_guest.clone().detach().requires_grad_(True)
                    local_max = np.max(inp_guest.cpu().data.numpy())
                    local_min = np.min(inp_guest.cpu().data.numpy())
                    orig = S_guest.clone().detach()

                    store_orig.append(orig.cpu().data.numpy())

                    grad = 0.0
                    pred = Variable(torch.Tensor([float(pred)]).reshape(1, -1).to(device).float())
                    for kk in range(20):
                        temp_inp_host = transformer(host_data_list[kk][1:].astype(float).reshape((1, -1)))

                        inp_host = Variable(temp_inp_host.to(device).float().unsqueeze(0), requires_grad=True)

                        for i in range(iter_num):
                            out = model(inp_guest, inp_host)
                            out = nn.Sigmoid()(out)
                            loss1 = criterion(out, pred)
                            
                            loss1.backward(create_graph=False, retain_graph=True)

                            grady_x1_loss1 = inp_guest.grad.data.clone()
                            # print(grady_x1_loss1)

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
                            
                            gradvar_x2 = torch.autograd.grad(out, inp_host, create_graph=True, retain_graph=True)
                            loss2 = torch.abs(gradvar_x2[0]).mean()
                            # loss2.backward()
                            # optimizer.step()
                            grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)

                            # inp_guest.grad.data.zero_()
                            # inp_host.grad.data.zero_()
                            # model.zero_grad()
                            
                            grad = grady_x1_loss1 * 0.99 + grady_x1_loss2[0] * 0.01 + grad * 0.8
                            # grad = grady_x1_loss1
                            perturbation = -lr * grad

                            # print(perturbation)
                            # perturbation = torch.clamp((inp_guest.data + perturbation) - orig, min=-eps * 0.5, max=eps * 3.0)
                            # # perturbation = inp_guest.data + perturbation - orig

                            # inp_guest.data = torch.clamp((orig + perturbation), min=-0.41, max=3)
                            # print(perturbation)
                            perturbation += inp_guest.data - orig.data
                            divider = torch.sum(torch.abs(perturbation))
                            perturbation /= divider
                            perturbation *= 26.0
                            inp_guest.data = orig.data + perturbation
                            inp_guest.data = torch.clamp(inp_guest.data, min=local_min, max=local_max)

                            inp_guest.grad.data.zero_()
                            inp_host.grad.data.zero_()
                            model.zero_grad()
    
                            # vis.plot('loss1', loss1.data.cpu())
                            # vis.plot('loss2', loss2.data.cpu())
                            # del loss1, loss2, grady_x1_loss2, grad, out, gradvar_x2, grady_x1_loss1
                            del loss1, grady_x1_loss1, out
                            torch.cuda.empty_cache()

                        del inp_host

                        temp_acc = advtest(model, inp_guest, train_dataloader, pred, pred, args.dataset)
                        if temp_acc >= 95:
                            break
                    total_num += 1.0

                    if temp_acc >= 95:
                        ADI_num += 1.0
                        print('=' * 10)
                        print(ADI_num, total_num)

            del inp_guest
            del orig
            torch.cuda.empty_cache()

def advtest(model, input_adv_guest, input_host_list, target, true_target, dataset, use_gpu=True):
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    target = torch.Tensor([float(target)]).to(device).long()
    true_target = torch.Tensor([float(true_target)]).to(device).long()
    if dataset == 'mnist':
        confusion_matrix1 = meter.ConfusionMeter(10)
        confusion_matrix2 = meter.ConfusionMeter(10)
        transformer = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean = [0],
                std = [1.0])
        ])
        model.eval()
        with torch.no_grad():
            for _, (_, data2, _) in enumerate(input_host_list):

                input1 = torch.cat([input_adv_guest] * data2.shape[0])
                input2 = Variable(data2).to(device)
                target1 = np.array([target] * data2.shape[0]).astype(int)
                target1 = Variable(torch.from_numpy(target1))
                score = model(input1, input2)
                confusion_matrix1.add(score.data.squeeze(), target1.type(torch.LongTensor))


            cm_value = confusion_matrix1.value()
            accuracy = 0.0
            for i in range(10):
                accuracy += cm_value[i][i]
            accuracy = 100. * accuracy / (cm_value.sum())
    elif dataset == 'nuswide':
        confusion_matrix1 = meter.ConfusionMeter(10)
        confusion_matrix2 = meter.ConfusionMeter(10)
        transformer = T.Compose([
            T.ToTensor(),
        ])
        model.eval()
        with torch.no_grad():
            for _, (_, data2, _) in enumerate(input_host_list):

                input1 = torch.cat([input_adv_guest] * data2.shape[0])
                input2 = Variable(data2).to(device)
                target1 = np.array([target] * data2.shape[0]).astype(int)
                target1 = Variable(torch.from_numpy(target1))
                score = model(input1, input2)
                confusion_matrix1.add(score.data.squeeze(), target1.type(torch.LongTensor))

            cm_value = confusion_matrix1.value()
            accuracy = 0.0
            for i in range(10):
                accuracy += cm_value[i][i]
            accuracy = 100. * accuracy / (cm_value.sum())
    elif dataset == 'vehicle':
        confusion_matrix1 = meter.ConfusionMeter(4)
        transformer = T.Compose([
            T.ToTensor(),
        ])

        model.eval()

        with torch.no_grad():
            for _, (_, data2, _) in enumerate(input_host_list):

                input1 = torch.cat([input_adv_guest] * data2.shape[0])
                input2 = Variable(data2).to(device)
                target1 = np.array([target] * data2.shape[0]).astype(int)
                target1 = Variable(torch.from_numpy(target1))
                # print(input1.shape, input2.shape)
                score = model(input1, input2)
                confusion_matrix1.add(score.data.squeeze(), target1.type(torch.LongTensor))
            cm_value = confusion_matrix1.value()
            accuracy = 0.0
            for i in range(4):
                accuracy += cm_value[i][i]
            accuracy = 100. * accuracy / (cm_value.sum())

    elif dataset == 'credit':
        from sklearn.metrics import confusion_matrix
        auc_meter = meter.AUCMeter()
        transformer = T.Compose([
            T.ToTensor(),
        ])

        model.eval()
        y_pred_list = np.array([])
        with torch.no_grad():
            for _, (_, data2, orig_label) in enumerate(input_host_list):

                input1 = torch.cat([input_adv_guest] * data2.shape[0])
                input2 = Variable(data2).to(device)
                indexs = np.where(orig_label != target.cpu().data.numpy()[0])
                target1 = np.array([target] * data2.shape[0]).astype(int)
                target1 = Variable(torch.from_numpy(target1))
                score = torch.nn.Sigmoid()(model(input1, input2))
                y_pred_tag = torch.round(score)
                y_pred_list = np.append(y_pred_list, y_pred_tag.cpu().numpy()[indexs, 0])
            y_pred_list = list(np.array(y_pred_list))
            y_pred_list = np.array(y_pred_list)
            if target.cpu().data[0].item() == 0:
                accuracy = float(y_pred_list.shape[0] - y_pred_list.sum()) / float(y_pred_list.shape[0]) * 100.0
            else:
                accuracy = float(y_pred_list.shape[0]) / float(y_pred_list.shape[0]) * 100.0

    torch.cuda.empty_cache()
    # print('Adv example test accuracy: ', accuracy)
    return accuracy

def advtest_alpha(model, input_adv_guest, input_host_list, target, true_target, dataset, use_gpu=True):
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    target = torch.Tensor([float(target)]).to(device).long()
    true_target = torch.Tensor([float(true_target)]).to(device).long()
    if dataset == 'mnist':
        confusion_matrix1 = meter.ConfusionMeter(10)
        confusion_matrix2 = meter.ConfusionMeter(10)

        transformer = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean = [0],
                std = [1.0])
        ])
        model.eval()
        with torch.no_grad():
            for _, (data1, _, _) in enumerate(input_host_list):

                input2 = torch.cat([input_adv_guest] * data1.shape[0])
                input1 = Variable(data1).to(device)
                target1 = np.array([target] * data1.shape[0]).astype(int)
                target1 = Variable(torch.from_numpy(target1))
                score = model(input1, input2)
                confusion_matrix1.add(score.data.squeeze(), target1.type(torch.LongTensor))


            cm_value = confusion_matrix1.value()
            accuracy = 0.0
            for i in range(10):
                accuracy += cm_value[i][i]
            accuracy = 100. * accuracy / (cm_value.sum())
    torch.cuda.empty_cache()
    return accuracy

def advtest_multi_3(model, input_adv_guest, input_host_list, target, true_target, dataset, use_gpu=True):
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    target = torch.Tensor([float(target)]).to(device).long()
    true_target = torch.Tensor([float(true_target)]).to(device).long()
    if dataset == 'mnist':
        confusion_matrix1 = meter.ConfusionMeter(10)
        confusion_matrix2 = meter.ConfusionMeter(10)

        transformer = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean = [0],
                std = [1.0])
        ])
        model.eval()
        with torch.no_grad():
            for _, (_, data2, data3, _) in enumerate(input_host_list):

                input1 = torch.cat([input_adv_guest] * data2.shape[0])
                input2 = Variable(data2).to(device)
                input3 = Variable(data3).to(device)
                target1 = np.array([target] * data2.shape[0]).astype(int)
                target1 = Variable(torch.from_numpy(target1))
                score = model(input1, input2, input3)
                confusion_matrix1.add(score.data.squeeze(), target1.type(torch.LongTensor))

            cm_value = confusion_matrix1.value()
            accuracy = 0.0
            for i in range(10):
                accuracy += cm_value[i][i]
            accuracy = 100. * accuracy / (cm_value.sum())
    torch.cuda.empty_cache()
    return accuracy

def advtest_multi_5(model, input_adv_guest, input_host_list, target, true_target, dataset, use_gpu=True):
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    target = torch.Tensor([float(target)]).to(device).long()
    true_target = torch.Tensor([float(true_target)]).to(device).long()
    if dataset == 'mnist':
        confusion_matrix1 = meter.ConfusionMeter(10)
        confusion_matrix2 = meter.ConfusionMeter(10)

        transformer = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean = [0],
                std = [1.0])
        ])
        model.eval()
        with torch.no_grad():
            for _, (_, data2, data3, data4, data5, _) in enumerate(input_host_list):

                input1 = torch.cat([input_adv_guest] * data2.shape[0])
                input2 = Variable(data2).to(device)
                input3 = Variable(data3).to(device)
                input4 = Variable(data4).to(device)
                input5 = Variable(data5).to(device)
                target1 = np.array([target] * data2.shape[0]).astype(int)
                target1 = Variable(torch.from_numpy(target1))
                score = model(input1, input2, input3, input4, input5)
                confusion_matrix1.add(score.data.squeeze(), target1.type(torch.LongTensor))

            cm_value = confusion_matrix1.value()
            accuracy = 0.0
            for i in range(10):
                accuracy += cm_value[i][i]
            accuracy = 100. * accuracy / (cm_value.sum())
    torch.cuda.empty_cache()
    return accuracy



if __name__ == '__main__':
    advtrain()