from .basic import BasicModule
from torch import nn
from torch.nn import functional as F
import torchvision.models as Tmodels
import torch

class VFLModel(BasicModule):
    def __init__(self, num_classes=10):
        super(VFLModel, self).__init__()
        self.model_name = 'VFLModel'
        
        self.featuresguest = Tmodels.vgg16(pretrained=True).features
        for p in self.featuresguest.parameters():
            p.requires_grad = False
        self.featureshost = Tmodels.vgg16(pretrained=True).features
        for p in self.featureshost.parameters():
            p.requires_grad = False

        self.top = nn.Sequential(
            nn.Dropout(),
            # 50176
            nn.Linear(50176, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x_guest, x_host):
        guest_out = self.featuresguest(x_guest)
        guest_out = guest_out.view(guest_out.size(0), -1)
        host_out = self.featureshost(x_host)
        host_out = host_out.view(host_out.size(0), -1)

        interactive = torch.cat((host_out, guest_out), 1)

        top_out = self.top(interactive)

        return top_out

class VFLModel_FULL(BasicModule):
    def __init__(self, num_classes=10):
        super(VFLModel, self).__init__()
        self.model_name = 'VFLModel_FULL'
        
        self.featuresguest = Tmodels.vgg16(pretrained=False).features
        for p in self.featuresguest.parameters():
            p.requires_grad = True
        self.featureshost = Tmodels.vgg16(pretrained=False).features
        for p in self.featureshost.parameters():
            p.requires_grad = True

        self.top = nn.Sequential(
            nn.Dropout(),
            nn.Linear(50176, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x_guest, x_host):
        guest_out = self.featuresguest(x_guest)
        guest_out = guest_out.view(guest_out.size(0), -1)
        host_out = self.featureshost(x_host)
        host_out = host_out.view(host_out.size(0), -1)

        interactive = torch.cat((host_out, guest_out), 1)

        top_out = self.top(interactive)

        return top_out

class VFLMnist(BasicModule):
    def __init__(self, num_classes=10):
        super(VFLMnist, self).__init__()
        self.model_name = 'VFLMnist'
        
        self.featuresguest = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,3),stride=2)
        )

        self.featureshost = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,3),stride=2)
        )

        self.top = nn.Sequential(
            nn.Dropout(),
            nn.Linear(640, 10),
        )

    def forward(self, x_guest, x_host):
        guest_out = self.featuresguest(x_guest)
        guest_out = guest_out.view(guest_out.size(0), -1)
        host_out = self.featureshost(x_host)
        host_out = host_out.view(host_out.size(0), -1)

        interactive = torch.cat((host_out, guest_out), 1)

        top_out = self.top(interactive)

        return top_out


class VFLMnist_alpha(BasicModule):
    def __init__(self, alpha=0.5, num_classes=10):
        self.alpha = alpha
        super(VFLMnist_alpha, self).__init__()
        self.model_name = 'VFLMnist_alpha'
        
        self.featuresguest = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.featureshost = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        if self.alpha == 0.3 or self.alpha == 0.6 or self.alpha == 0.8 or self.alpha == 0.65:
            self.top = nn.Sequential(
                nn.Dropout(),
                nn.Linear(4992, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256,10)
            )
        elif self.alpha == 0.7 or self.alpha == 0.4 or self.alpha == 0.2:
            
            self.top = nn.Sequential(
                nn.Dropout(),
                nn.Linear(4576, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256,10)
            )

    def forward(self, x_guest, x_host):
        guest_out = self.featuresguest(x_guest)
        guest_out = guest_out.view(guest_out.size(0), -1)
        host_out = self.featureshost(x_host)
        host_out = host_out.view(host_out.size(0), -1)

        interactive = torch.cat((host_out, guest_out), 1)
        top_out = self.top(interactive)

        return top_out


class VFLMnist_biased(BasicModule):
    def __init__(self, num_classes=10):
        super(VFLMnist_biased, self).__init__()
        self.model_name = 'VFLMnist_biased'
        
        self.featuresguest = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.featureshost = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.top = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2240, 10),
        )

    def forward(self, x_guest, x_host):
        guest_out = self.featuresguest(x_guest)
        guest_out = guest_out.view(guest_out.size(0), -1)
        host_out = self.featureshost(x_host)
        host_out = host_out.view(host_out.size(0), -1)

        interactive = torch.cat((host_out, guest_out), 1)

        top_out = self.top(interactive)

        return top_out

class VFLVehicle(BasicModule):
    def __init__(self, num_classes=4):
        super(VFLVehicle, self).__init__()
        self.model_name = 'VFLVehicle'
        
        self.featuresguest = nn.Sequential(
            nn.Linear(9, 4),
            nn.ReLU(inplace=True),
        )

        self.featureshost = nn.Sequential(
            nn.Linear(9, 4),
            nn.ReLU(inplace=True),
        )

        self.top = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x_guest, x_host):
        guest_out = self.featuresguest(x_guest)
        guest_out = guest_out.view(guest_out.size(0), -1)
        host_out = self.featureshost(x_host)
        host_out = host_out.view(host_out.size(0), -1)
        interactive = guest_out + host_out

        top_out = interactive

        return top_out


class VFLCredit(BasicModule):
    def __init__(self, num_classes=4):
        super(VFLCredit, self).__init__()
        self.model_name = 'VFLCredit'
        
        self.featuresguest = nn.Sequential(
            nn.Linear(13, 1),
        )

        self.featureshost = nn.Sequential(
            nn.Linear(10, 1),
        )

        self.top = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x_guest, x_host):
        guest_out = self.featuresguest(x_guest)
        guest_out = guest_out.view(guest_out.size(0), -1)
        host_out = self.featureshost(x_host)
        host_out = host_out.view(host_out.size(0), -1)
        interactive = guest_out + host_out
        top_out = interactive

        return top_out

class VFLNusWide(BasicModule):
    def __init__(self, num_classes=10):
        super(VFLNusWide, self).__init__()
        self.model_name = 'VFLNusWide'
        
        self.featuresguest = nn.Sequential(
            nn.Linear(634, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        )

        self.featureshost = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        )

        self.top = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10)
        )

    def forward(self, x_guest, x_host):
        guest_out = self.featuresguest(x_guest)
        guest_out = guest_out.view(guest_out.size(0), -1)
        host_out = self.featureshost(x_host)
        host_out = host_out.view(host_out.size(0), -1)

        interactive = torch.cat((host_out, guest_out), 1)

        top_out = self.top(interactive)
        return top_out

class VFLStudent(BasicModule):
    def __init__(self, num_classes=1):
        super(VFLStudent, self).__init__()
        self.model_name = 'VFLStudent'
        
        self.featuresguest = nn.Sequential(
            nn.Linear(6, 3),
            nn.ReLU(inplace=True),
        )

        self.featureshost = nn.Sequential(
            nn.Linear(7, 3),
            nn.ReLU(inplace=True),
        )

        self.top = nn.Sequential(
            nn.Linear(6, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 1)
        )

    def forward(self, x_guest, x_host):
        guest_out = self.featuresguest(x_guest)
        guest_out = guest_out.view(guest_out.size(0), -1)
        host_out = self.featureshost(x_host)
        host_out = host_out.view(host_out.size(0), -1)

        interactive = torch.cat((host_out, guest_out), 1)

        top_out = self.top(interactive)

        return top_out


class VFLMultiMNIST(BasicModule):
    def __init__(self, num_agents=3):
        super(VFLMultiMNIST, self).__init__()
        self.model_name = 'VFLMultiMNIST'

        self.model_a = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.model_b = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.model_c = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.top = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4160, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x_a, x_b, x_c):
        a_out = self.model_a(x_a)
        b_out = self.model_b(x_b)
        c_out = self.model_c(x_c)
        a_out = a_out.view(a_out.size(0), -1)
        b_out = b_out.view(b_out.size(0), -1)
        c_out = c_out.view(c_out.size(0), -1)

        interactive = torch.cat((a_out, b_out, c_out), 1)

        top_out = self.top(interactive)
        return top_out

class VFLMultiMNIST_5(BasicModule):
    def __init__(self, num_agents=3):
        super(VFLMultiMNIST_5, self).__init__()
        self.model_name = 'VFLMultiMNIST_5'

        self.model_a = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.model_b = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.model_c = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.model_d = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.model_e = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.top = nn.Sequential(
            nn.Dropout(),
            nn.Linear(11232, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x_a, x_b, x_c, x_d, x_e):
        a_out = self.model_a(x_a)
        b_out = self.model_b(x_b)
        c_out = self.model_c(x_c)
        d_out = self.model_c(x_d)
        e_out = self.model_c(x_e)

        a_out = a_out.view(a_out.size(0), -1)
        b_out = b_out.view(b_out.size(0), -1)
        c_out = c_out.view(c_out.size(0), -1)
        d_out = d_out.view(d_out.size(0), -1)
        e_out = e_out.view(e_out.size(0), -1)

        interactive = torch.cat((a_out, b_out, c_out, d_out, e_out), 1)
        top_out = self.top(interactive)
        return top_out