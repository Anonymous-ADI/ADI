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

    assert args.dataset == 'credit'
    print('Model : %s \n' %(model_name))
    # load model
    model = getattr(models, model_name)().eval()
    model.load(args.credit_model_path)
    device = 'cuda:' + str(args.device) if gpu else 'cpu'
    model.to(device)

    transformer = T.Compose([T.ToTensor()])
    
    guest_data_list = pd.read_csv(args.credit_guest_list).to_numpy()
    host_data_list = pd.read_csv(args.credit_host_list).to_numpy()
    criterion = nn.BCEWithLogitsLoss()
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

                        inp_guest.grad.data.zero_()
                        inp_host.grad.data.zero_()
                        model.zero_grad()
                        
                        gradvar_x2 = torch.autograd.grad(out, inp_host, create_graph=True, retain_graph=True)
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

                        inp_guest.grad.data.zero_()
                        inp_host.grad.data.zero_()
                        model.zero_grad()
                        
                        gradvar_x2 = torch.autograd.grad(out, inp_host, create_graph=True, retain_graph=True)
                        loss2 = torch.abs(gradvar_x2[0]).mean()

                        grady_x1_loss2 = torch.autograd.grad(loss2, inp_guest, retain_graph=True)
                        
                        grad = grady_x1_loss1 * 0.99 + grady_x1_loss2[0] * 0.01 + grad * 0.8
                        perturbation = -lr * grad
                        perturbation += inp_guest.data - orig.data
                        divider = torch.sum(torch.abs(perturbation))
                        perturbation /= divider
                        perturbation *= 26.0
                        inp_guest.data = orig.data + perturbation
                        inp_guest.data = torch.clamp(inp_guest.data, min=local_min, max=local_max)

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
    if dataset == 'credit':
        from sklearn.metrics import confusion_matrix
        auc_meter = meter.AUCMeter()
        transformer = T.Compose([T.ToTensor(),])

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