from config import opt
import os
import torch as t
import models
from data.dataset import mnist, vehicle, student, NusWide, credit, mnist_alpha, mnist_multi, mnist_multi_5
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
from torch.nn import init
from torch import nn
import numpy as np


def init_seed(seed=100):
    t.manual_seed(seed) # Sets the seed for generating random numbers.
    t.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    t.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # t.backends.cudnn.determnistic = True

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def train(**kwargs):
    opt.parse(kwargs)
    t.cuda.set_device(opt.device)
    vis = Visualizer(opt.env)

    init_seed()
    model = getattr(models, opt.model)(opt.alpha)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    else:
        model.apply(weigth_init)
    if opt.use_gpu: model.cuda()

    train_data = mnist_alpha(alpha=opt.alpha, mode='train', dataset='mnist')
    val_data = mnist_alpha(alpha=opt.alpha, mode='val', dataset='mnist')

    train_dataloader = DataLoader(train_data, opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr, weight_decay = opt.weight_decay, betas=(opt.beta1, opt.beta2))
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    for epoch in range(opt.max_epoch):
        loss_meter.reset()

        for ii, (data1, data2, label) in tqdm(enumerate(train_dataloader),total=len(train_data) / opt.batch_size):
            input1 = Variable(data1)
            input2 = Variable(data2)
            target = Variable(label)
            if opt.use_gpu:
                input1 = input1.cuda()
                input2 = input2.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input1, input2)
            loss = criterion(score.squeeze().squeeze(), target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.data.cpu())

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

        model.save()
        val_cm, val_accuracy = val(model, val_dataloader)

        vis.plot('val_accuracy', val_accuracy)
                
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

def train_mnist_3(**kwargs):
    opt.parse(kwargs)
    t.cuda.set_device(opt.device)
    vis = Visualizer(opt.env)

    init_seed()
    model = getattr(models, opt.model)(opt.alpha)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    else:
        model.apply(weigth_init)
    if opt.use_gpu: model.cuda()

    train_data = mnist_multi(mode='train')
    val_data = mnist_multi(mode='val')
    
    train_dataloader = DataLoader(train_data, opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    criterion = t.nn.CrossEntropyLoss()

    lr = opt.lr
    optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr, weight_decay = opt.weight_decay, betas=(opt.beta1, opt.beta2))

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(10)
    previous_loss = 1e10

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        for ii, (data1, data2, data3, label) in tqdm(enumerate(train_dataloader),total=len(train_data) / opt.batch_size):
            input1 = Variable(data1)
            input2 = Variable(data2)
            input3 = Variable(data3)
            target = Variable(label)
            if opt.use_gpu:
                input1 = input1.cuda()
                input2 = input2.cuda()
                input3 = input3.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input1, input2, input3)
            loss = criterion(score.squeeze().squeeze(), target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.data.cpu())

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])


        model.save()
        val_cm, val_accuracy = val_3(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        previous_loss = loss_meter.value()[0]

def val_3(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(10)

    loss_meter = meter.AverageValueMeter()
    for ii, data in enumerate(dataloader):
        input1, input2, input3, label = data
        with t.no_grad():
            val_input1 = Variable(input1)
            val_input2 = Variable(input2)
            val_input3 = Variable(input3)

            val_label = Variable(label.type(t.LongTensor))
        if opt.use_gpu:
            val_input1 = val_input1.cuda()
            val_input2 = val_input2.cuda()
            val_input3 = val_input3.cuda()
            val_label = val_label.cuda()
        score = model(val_input1, val_input2, val_input3)

        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 0.0
    for i in range(10):
        accuracy += cm_value[i][i]
    accuracy = 100. * accuracy / float(cm_value.sum())
    return confusion_matrix, accuracy


def train_mnist_5(**kwargs):
    opt.parse(kwargs)
    t.cuda.set_device(opt.device)
    vis = Visualizer(opt.env)

    init_seed()

    model = getattr(models, opt.model)(opt.alpha)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    else:
        model.apply(weigth_init)
    if opt.use_gpu: model.cuda()

    train_data = mnist_multi_5(mode='train')
    val_data = mnist_multi_5(mode='val')

    train_dataloader = DataLoader(train_data, opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    criterion = t.nn.CrossEntropyLoss()

    lr = opt.lr
    optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr, weight_decay = opt.weight_decay, betas=(opt.beta1, opt.beta2))
    # optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(10)
    previous_loss = 1e10

    for epoch in range(opt.max_epoch):

        loss_meter.reset()

        for ii, (data1, data2, data3, data4, data5, label) in tqdm(enumerate(train_dataloader),total=len(train_data) / opt.batch_size):
            input1 = Variable(data1)
            input2 = Variable(data2)
            input3 = Variable(data3)
            input4 = Variable(data4)
            input5 = Variable(data5)
            target = Variable(label)
            if opt.use_gpu:
                input1 = input1.cuda()
                input2 = input2.cuda()
                input3 = input3.cuda()
                input4 = input4.cuda()
                input5 = input5.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input1, input2, input3, input4, input5)
            loss = criterion(score.squeeze().squeeze(), target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.data.cpu())

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
        model.save()
        val_cm, val_accuracy = val_5(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]


def val_5(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(10)
    loss_meter = meter.AverageValueMeter()
    for ii, data in enumerate(dataloader):
        input1, input2, input3, input4, input5, label = data
        with t.no_grad():
            val_input1 = Variable(input1)
            val_input2 = Variable(input2)
            val_input3 = Variable(input3)
            val_input4 = Variable(input4)
            val_input5 = Variable(input5)

            val_label = Variable(label.type(t.LongTensor))
        if opt.use_gpu:
            val_input1 = val_input1.cuda()
            val_input2 = val_input2.cuda()
            val_input3 = val_input3.cuda()
            val_input4 = val_input4.cuda()
            val_input5 = val_input5.cuda()
            val_label = val_label.cuda()
        score = model(val_input1, val_input2, val_input3, val_input4, val_input5)
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 0.0
    for i in range(10):
        accuracy += cm_value[i][i]
    accuracy = 100. * accuracy / float(cm_value.sum())
    return confusion_matrix, accuracy


def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(10)
    loss_meter = meter.AverageValueMeter()
    for ii, data in enumerate(dataloader):
        input1, input2, label = data
        with t.no_grad():
            val_input1 = Variable(input1)
            val_input2 = Variable(input2)
            val_label = Variable(label.type(t.LongTensor))
        if opt.use_gpu:
            val_input1 = val_input1.cuda()
            val_input2 = val_input2.cuda()
            val_label = val_label.cuda()
        score = model(val_input1, val_input2)

        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 0.0
    for i in range(10):
        accuracy += cm_value[i][i]
    accuracy = 100. * accuracy / float(cm_value.sum())

    return confusion_matrix, accuracy

def help():
    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


if __name__=='__main__':
    import fire
    fire.Fire()
